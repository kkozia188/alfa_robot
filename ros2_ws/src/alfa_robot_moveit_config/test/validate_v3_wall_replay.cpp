// Independent acceptance checker: reconstruct the published scene and interpolate
// the published joints, without calling the demo's planner or collision helpers.
#include <moveit/planning_scene/planning_scene.h>
#include <moveit/robot_model/robot_model.h>
#include <moveit_msgs/msg/collision_object.hpp>
#include <geometric_shapes/shapes.h>
#include <srdfdom/model.h>
#include <urdf_parser/urdf_parser.h>
#include <nlohmann/json.hpp>
#include <cmath>
#include <set>
#include <fstream>
#include <iostream>
#include <stdexcept>

using Json = nlohmann::json;
void require(bool ok, const std::string& message) { if (!ok) throw std::runtime_error(message); }
std::string read(const char* path) { std::ifstream in(path); require(bool(in), path); return {std::istreambuf_iterator<char>(in), {}}; }
Eigen::Vector3d vec(const Json& j) { return {j[0].get<double>(), j[1].get<double>(), j[2].get<double>()}; }

void validateCycle(const Json& task, const moveit::core::RobotModelPtr& model)
{
    require(task.at("success"), "requires a successful cycle");
    const bool release = task.at("release_after_transfer");
    planning_scene::PlanningScene scene(model);
    require(task.at("collision_inset").get<double>() == 0, "box geometry must not be shrunk");
    const auto size = vec(task.at("box_size"));
    const auto center = vec(task.at("box_center"));
    auto add = [&](const std::string& id, const Eigen::Vector3d& p, const Eigen::Vector3d& dimensions) {
      moveit_msgs::msg::CollisionObject obj;
      obj.header.frame_id = model->getModelFrame(); obj.id = id; obj.operation = obj.ADD;
      shape_msgs::msg::SolidPrimitive box; box.type = box.BOX;
      box.dimensions = {dimensions.x(), dimensions.y(), dimensions.z()};
      geometry_msgs::msg::Pose pose; pose.orientation.w = 1;
      pose.position.x = p.x(); pose.position.y = p.y(); pose.position.z = p.z();
      obj.primitives.push_back(box); obj.primitive_poses.push_back(pose);
      require(scene.processCollisionObjectMsg(obj), "cannot add " + id);
    };
    require(task.at("environment").at("enabled"), "environment must remain enabled");
    for (const auto& b : task.at("environment").at("boxes")) add(b.at("id"), vec(b.at("center")), vec(b.at("size")));
    size_t neighbor = 0;
    for (const auto& c : task.at("neighbor_centers")) add("neighbor_" + std::to_string(neighbor++), vec(c), size);
    add("target", center, size);
    const auto names = task.at("joint_names").get<std::vector<std::string>>();
    const auto frames = task.at("frames");
    require(frames.size() > 2 && !frames.front().at("box_attached").get<bool>(), "missing empty-arm start");
    require(frames.front().at("joints") == task.at("initial_joints"), "start teleported from home");
    const std::string tool = task.at("tool_link"), side = task.at("side");
    Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
    offset.translation() = vec(task.at("tool_to_box_center"));
    for (int i = 0; i < 3; ++i) for (int j = 0; j < 3; ++j) offset.linear()(i,j) = task.at("tool_to_box_rotation")[i][j];
    moveit::core::RobotState state(model); state.setToDefaultValues();
    bool attached = false;
    size_t checks = 0, attachments = 0, releases = 0;
    for (size_t f = 0; f < frames.size(); ++f) {
      const auto q = frames[f].at("joints").get<std::vector<double>>();
      require(q.size() == names.size(), "invalid frame joint count");
      const auto prev = f ? frames[f-1].at("joints").get<std::vector<double>>() : q;
      const bool now_attached = frames[f].at("box_attached");
      const std::string stage = frames[f].at("stage");
      require(stage.find("SNAPSHOT") == std::string::npos && stage.find("UNCONNECTED") == std::string::npos, "diagnostics in executable frames");
      if (now_attached != attached) {
        require(q == prev, "attachment/release teleported");
        for (size_t j = 0; j < names.size(); ++j) state.setVariablePosition(names[j], q[j]);
        state.update(true);
        const Eigen::Isometry3d box = state.getGlobalLinkTransform(tool) * offset;
        if (now_attached) {
          require(stage == "attach_box" && releases == 0, "unexpected attachment");
          require((box.translation() - center).norm() < 1e-6 &&
            (box.linear() - Eigen::Matrix3d::Identity()).norm() < 1e-6, "box jumped/rotated at attachment");
          scene.getWorldNonConst()->removeObject("target");
          EigenSTL::vector_Isometry3d poses{offset};
          state.attachBody("target", Eigen::Isometry3d::Identity(), {shapes::ShapeConstPtr(new shapes::Box(size.x(),size.y(),size.z()))},
            poses, std::set<std::string>{tool, side + "_link7"}, tool);
          ++attachments;
        } else {
          require(release && stage == "release_box" && f && frames[f-1].at("stage") == "rear_placement", "unexpected release");
          // Verify all eight corners independently of the planner's extent helper.
          for (int i = 0; i < 8; ++i) {
            const Eigen::Vector3d corner((i & 1) ? size.x()/2 : -size.x()/2,
              (i & 2) ? size.y()/2 : -size.y()/2, (i & 4) ? size.z()/2 : -size.z()/2);
            require((box * corner).x() <= task.at("chassis_rear_x").get<double>() - .01 + 1e-8, "box not fully behind chassis at release");
          }
          state.clearAttachedBody("target"); ++releases;
        }
        attached = now_attached;
      }
      require(frames[f].at("box_visible").get<bool>() == (releases == 0), "box disappeared before release or reappeared");
      size_t steps = 1;
      for (size_t j = 0; j < names.size(); ++j)
        steps = std::max(steps, static_cast<size_t>(std::ceil(std::abs(q[j]-prev[j]) /
          (names[j] == "updown" ? .0025 : .5 * std::acos(-1.) / 180.))));
      for (size_t step = 1; step <= steps; ++step) {
        for (size_t j = 0; j < names.size(); ++j)
          state.setVariablePosition(names[j], prev[j] + (q[j]-prev[j])*step/steps);
        state.update(true);
        require(state.satisfiesBounds(), "joint limit at frame " + std::to_string(f));
        collision_detection::CollisionRequest request; request.contacts = true; request.max_contacts = 10;
        collision_detection::CollisionResult result;
        auto acm = scene.getAllowedCollisionMatrix();
        // Only the true suction contact may touch the target; never exempt neighbors/environment.
        if (!attached && (state.getGlobalLinkTransform(tool) * offset).translation().isApprox(center, 1e-6)) {
          acm.setEntry("target", tool, true); acm.setEntry("target", side + "_link7", true);
        }
        scene.checkCollision(request, result, state, acm); ++checks;
        if (result.collision) {
          std::string pairs;
          for (const auto& contact : result.contacts) pairs += contact.first.first + "<->" + contact.first.second + " ";
          throw std::runtime_error("collision frame=" + std::to_string(f) + " stage=" + stage + " " + pairs);
        }
      }
    }
    require(attachments == 1 && releases == (release ? 1u : 0u) && attached == !release, "wrong payload lifecycle");
    if (release) require(frames.back().at("stage") == "release_box", "cycle must end at rear release, not reset home");
    std::cout << "PASS independent full-scene replay: " << frames.size() << " frames, " << checks
              << " collision/bounds probes (0.5deg / 2.5mm), attachment continuous, payload "
              << (release ? "released behind chassis, final posture retained" : "retained") << "\n";
}

void validateDirectPlacement(const Json& task, const moveit::core::RobotModelPtr& model)
{
  require(task.at("direct_attach"), "requires direct placement");
  const bool successful = task.at("success");
  const std::string placement_pose = task.value("direct_placement_pose", "unloading");
  require(placement_pose == "unloading" || placement_pose == "second_unloading", "invalid placement pose");
  require(task.at("connection_planner") == "shortcut_local_rrt", "wrong connection planner");
  require(task.at("rear_placement_strategy") == "named_unloading", "wrong placement target");
  require(task.at("wall_context") == "target_only" &&
    !task.at("release_after_transfer").get<bool>(), "unexpected wall/release policy");
  require(task.at("collision_inset").get<double>() == 0, "payload geometry was shrunk");
  const auto names = task.at("joint_names").get<std::vector<std::string>>();
  const auto& frames = successful ? task.at("frames") : task.at("diagnostic_frames");
  require(frames.size() > 2 && frames.front().at("stage") == "direct_attach" &&
    frames.back().at("stage") == (successful ? placement_pose : "REJECTED_SHORTCUT_PREVIEW_NOT_EXECUTED"),
    "unexpected direct placement stages");
  require(frames.front().at("joints") == task.at("initial_joints"), "initial state teleported");
  planning_scene::PlanningScene scene(model);
  require(task.at("environment").at("enabled"), "environment disabled");
  for (const auto& obstacle : task.at("environment").at("boxes")) {
    moveit_msgs::msg::CollisionObject object;
    object.header.frame_id = model->getModelFrame();
    object.id = obstacle.at("id");
    object.operation = object.ADD;
    shape_msgs::msg::SolidPrimitive shape;
    shape.type = shape.BOX;
    const auto dimensions = vec(obstacle.at("size"));
    shape.dimensions = {dimensions.x(), dimensions.y(), dimensions.z()};
    geometry_msgs::msg::Pose pose;
    const auto center = vec(obstacle.at("center"));
    pose.position.x = center.x(); pose.position.y = center.y(); pose.position.z = center.z();
    pose.orientation.w = 1;
    object.primitives.push_back(shape);
    object.primitive_poses.push_back(pose);
    require(scene.processCollisionObjectMsg(object), "cannot add " + object.id);
  }
  moveit::core::RobotState state(model);
  state.setToDefaultValues();
  require(state.setToDefaultValues(model->getJointModelGroup("whole_body"),
    task.at("initial_pose").get<std::string>()), "initial named pose missing");
  for (size_t joint = 0; joint < names.size(); ++joint)
    require(std::abs(state.getVariablePosition(names[joint]) -
      frames.front().at("joints").at(joint).get<double>()) < 1e-8,
      "wrong named initial joint " + names[joint]);
  state.update(true);
  const auto size = vec(task.at("box_size"));
  const auto payloads = task.at("direct_attached_boxes");
  require(payloads.size() == 2, "requires two attached boxes");
  std::set<std::string> sides;
  for (const auto& payload : payloads) {
    const std::string side = payload.at("side"), tool = payload.at("tool_link");
    require((side == "left" || side == "right") && sides.insert(side).second &&
      tool == side + "_tool0", "wrong payload side/tool");
    Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
    offset.translation() = vec(payload.at("tool_to_box_center"));
    Eigen::Matrix3d expected_rotation;
    for (int row = 0; row < 3; ++row)
      for (int column = 0; column < 3; ++column) {
        offset.linear()(row, column) = payload.at("tool_to_box_rotation")[row][column];
        expected_rotation(row, column) = payload.at("box_rotation")[row][column];
      }
    const Eigen::Isometry3d world_box = state.getGlobalLinkTransform(tool) * offset;
    require((world_box.translation() - vec(payload.at("box_center"))).norm() < 1e-6 &&
      (world_box.linear() - expected_rotation).norm() < 1e-6,
      "initial box transform mismatch");
    state.attachBody("carried_target_box_" + side, Eigen::Isometry3d::Identity(),
      {shapes::ShapeConstPtr(new shapes::Box(size.x(), size.y(), size.z()))},
      EigenSTL::vector_Isometry3d{offset}, std::set<std::string>{tool, side + "_joint7"}, tool);
  }
  const auto initial = task.at("initial_joints").get<std::vector<double>>();
  auto previous = initial;
  size_t probes = 0;
  bool final_collision_verified = false;
  for (size_t index = 0; index < frames.size(); ++index) {
    const auto& frame = frames.at(index);
    const std::string stage = frame.at("stage");
    require(stage == "direct_attach" || stage == "loaded_transfer" || stage == placement_pose ||
      (!successful && stage == "REJECTED_SHORTCUT_PREVIEW_NOT_EXECUTED"),
      "unexpected grasp/extraction stage " + stage);
    require(frame.at("box_attached") && frame.at("box_visible") &&
      frame.at("carried_boxes").size() == 2, "payload disappeared");
    const auto joints = frame.at("joints").get<std::vector<double>>();
    require(joints.size() == names.size(), "wrong joint count");
    size_t steps = 1;
    for (size_t joint = 0; joint < names.size(); ++joint) {
      if (names[joint] == "updown" || names[joint] == "head_joint" || names[joint] == "head_pitch_joint")
        require(std::abs(joints[joint] - initial[joint]) < 1e-8, "fixed axis changed");
      steps = std::max(steps, static_cast<size_t>(std::ceil(std::abs(joints[joint] - previous[joint]) /
        (names[joint] == "updown" ? .0025 : .5 * std::acos(-1.) / 180.))));
    }
    for (size_t sample = 1; sample <= steps; ++sample) {
      for (size_t joint = 0; joint < names.size(); ++joint)
        state.setVariablePosition(names[joint], previous[joint] +
          (joints[joint] - previous[joint]) * sample / steps);
      state.update(true);
      require(state.satisfiesBounds(), "joint limit at frame " + std::to_string(index));
      collision_detection::CollisionRequest request;
      collision_detection::CollisionResult collision;
      request.contacts = true;
      request.max_contacts = 20;
      scene.checkCollision(request, collision, state);
      ++probes;
      std::string pairs;
      for (const auto& contact : collision.contacts)
        pairs += contact.first.first + "<->" + contact.first.second + " ";
      if (!successful && index + 1 == frames.size()) {
        require(collision.collision, "rejected final sample has no collision");
        final_collision_verified = true;
      } else {
        require(!collision.collision, "collision at frame " + std::to_string(index) + " " + pairs);
      }
    }
    previous = joints;
  }
  moveit::core::RobotState unloading(model);
  unloading.setToDefaultValues();
  for (size_t joint = 0; joint < names.size(); ++joint)
    unloading.setVariablePosition(names[joint], initial[joint]);
  require(unloading.setToDefaultValues(model->getJointModelGroup("dual_arm"), placement_pose),
    "unloading named pose missing");
  for (size_t joint = 0; joint < names.size(); ++joint)
    require(std::abs((successful ? previous[joint] : task.at("goal_joints")[joint].get<double>()) -
      unloading.getVariablePosition(names[joint])) < 1e-8,
      "wrong unloading joint " + names[joint]);
  if (!successful) {
    require(final_collision_verified && task.at("diagnostic").at("diagnostic_only"),
      "failure preview was not verified");
    std::cout << "PASS rejected shortcut diagnostic: " << frames.size() << " frames, " << probes
      << " collision/bounds probes, last sample collides, no executable transfer claimed\n";
    return;
  }
  std::cout << "PASS direct dual-arm placement: " << frames.size() << " frames, " << probes
    << " collision/bounds probes (0.5deg / 2.5mm), exact named endpoints, two payloads retained\n";
}

int main(int argc, char** argv)
{
  try {
    require(argc == 4, "usage: validate_v3_wall_replay result.json robot.urdf robot.srdf");
    const auto task = Json::parse(read(argv[1]));
    auto urdf = urdf::parseURDF(read(argv[2]));
    auto srdf = std::make_shared<srdf::Model>();
    require(urdf && srdf->initString(*urdf, read(argv[3])), "invalid robot descriptions");
    auto model = std::make_shared<moveit::core::RobotModel>(urdf, srdf);
    if (task.value("direct_attach", false)) {
      validateDirectPlacement(task, model);
    } else if (!task.value("sequence", false)) {
      validateCycle(task, model);
    } else {
      require(task.at("success") && task.at("completed_count") == 25, "requires all 25 successful cycles");
      Json previous = task.at("initial_joints");
      std::set<int> removed;
      size_t end = 0;
      for (const auto& box : task.at("boxes")) {
        Json cycle = box;
        const size_t begin = box.at("frame_begin"), next = box.at("frame_end");
        require(begin == end && next > begin, "non-contiguous cycle range");
        require(box.at("initial_joints") == previous, "cycle boundary teleported");
        require(box.at("removed_box_ids").get<std::set<int>>() == removed, "wrong prior boxes removed");
        require(box.at("neighbor_centers").size() == 24 - removed.size(), "missing neighbors");
        cycle["frames"] = Json::array();
        for (size_t f = begin; f < next; ++f) cycle["frames"].push_back(task.at("frames").at(f));
        validateCycle(cycle, model);
        previous = cycle.at("frames").back().at("joints");
        removed.insert(box.at("box_id").get<int>());
        end = next;
      }
      require(removed.size() == 25 && end == task.at("frames").size(), "incomplete sequence");
      require(previous == task.at("final_joints"), "wrong final posture");
      std::cout << "PASS all 25 cycles, unchanged-joint handoff and exact prior-box removal\n";
    }
  } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
