#include "alfa_robot_moveit_config/extract_monitor_state.hpp"

#include <cassert>
#include <string>
#include <vector>

int main()
{
  using alfa_robot::motion::ExtractMonitorFullRunResult;
  using alfa_robot::motion::ExtractMonitorController;
  using alfa_robot::motion::ExtractMonitorPhase;
  using alfa_robot::motion::ExtractMonitorStage;
  using alfa_robot::motion::ExtractMonitorStageCallbacks;
  using alfa_robot::motion::AttachedBoxSpec;
  using alfa_robot::motion::extract_monitor_prefix;
  using alfa_robot::motion::make_extract_monitor_initial_state;
  using alfa_robot::motion::populate_extract_monitor_candidate_states;
  using alfa_robot::motion::summarize_extract_monitor_timings;
  using alfa_robot::motion::summarize_loaded_plan_timings;
  using alfa_robot::motion::extract_monitor_next_phase_after;
  using alfa_robot::motion::extract_monitor_phase_before_running;
  using alfa_robot::motion::extract_monitor_stage_for_phase;
  using alfa_robot::motion::run_extract_monitor_full_sequence;
  using alfa_robot::motion::run_extract_monitor_stage;

  assert(extract_monitor_stage_for_phase(ExtractMonitorPhase::ReadyForIk) == ExtractMonitorStage::Ik);
  assert(extract_monitor_stage_for_phase(ExtractMonitorPhase::ReadyForExtract) == ExtractMonitorStage::Extract);
  assert(extract_monitor_stage_for_phase(ExtractMonitorPhase::ReadyForLoaded) == ExtractMonitorStage::Loaded);
  assert(extract_monitor_stage_for_phase(ExtractMonitorPhase::ReadyForFinal) == ExtractMonitorStage::Final);
  assert(extract_monitor_stage_for_phase(ExtractMonitorPhase::Done) == ExtractMonitorStage::Ik);
  assert(extract_monitor_phase_before_running(ExtractMonitorPhase::Done) == ExtractMonitorPhase::ReadyForIk);
  assert(extract_monitor_phase_before_running(ExtractMonitorPhase::ReadyForLoaded) == ExtractMonitorPhase::ReadyForLoaded);

  assert(extract_monitor_next_phase_after(ExtractMonitorStage::Ik) == ExtractMonitorPhase::ReadyForExtract);
  assert(extract_monitor_next_phase_after(ExtractMonitorStage::Extract) == ExtractMonitorPhase::ReadyForLoaded);
  assert(extract_monitor_next_phase_after(ExtractMonitorStage::Loaded) == ExtractMonitorPhase::ReadyForFinal);
  assert(extract_monitor_next_phase_after(ExtractMonitorStage::Final) == ExtractMonitorPhase::Done);

  std::vector<std::string> calls;
  double last_elapsed = 0.0;
  ExtractMonitorStageCallbacks callbacks;
  callbacks.ik = [&](std::string* message) {
    calls.push_back("ik");
    last_elapsed = 1.0;
    if (message) *message = "ik ok";
    return true;
  };
  callbacks.extract = [&](std::string* message) {
    calls.push_back("extract");
    last_elapsed = 2.0;
    if (message) *message = "extract ok";
    return true;
  };
  callbacks.loaded = [&](std::string* message) {
    calls.push_back("loaded");
    last_elapsed = 3.0;
    if (message) *message = "loaded ok";
    return true;
  };
  callbacks.final = [&](std::string* message) {
    calls.push_back("final");
    last_elapsed = 4.0;
    if (message) *message = "final ok";
    return true;
  };

  std::string message;
  double elapsed = 0.0;
  assert(run_extract_monitor_stage(
    ExtractMonitorStage::Loaded, callbacks, [&] { return last_elapsed; }, &elapsed, &message));
  assert(calls.size() == 1 && calls.back() == "loaded");
  assert(elapsed == 3.0);
  assert(message == "loaded ok");

  calls.clear();
  last_elapsed = 0.0;
  const ExtractMonitorFullRunResult result = run_extract_monitor_full_sequence(callbacks, [&] { return last_elapsed; });
  assert(result.success);
  assert(calls.size() == 4);
  assert(calls[0] == "ik");
  assert(calls[1] == "extract");
  assert(calls[2] == "loaded");
  assert(calls[3] == "final");
  assert(result.stage_elapsed_ms[0] == 1.0);
  assert(result.stage_elapsed_ms[1] == 2.0);
  assert(result.stage_elapsed_ms[2] == 3.0);
  assert(result.stage_elapsed_ms[3] == 4.0);
  assert(result.message.find("完整流程完成") != std::string::npos);

  callbacks.extract = [&](std::string* fail_message) {
    calls.push_back("extract_fail");
    last_elapsed = 20.0;
    if (fail_message) *fail_message = "bad extract";
    return false;
  };
  calls.clear();
  const auto failed = run_extract_monitor_full_sequence(callbacks, [&] { return last_elapsed; });
  assert(!failed.success);
  assert(failed.message.find("完整流程失败在抽离阶段") != std::string::npos);

  ExtractMonitorController controller;
  calls.clear();
  callbacks.extract = [&](std::string* message) {
    calls.push_back("extract");
    last_elapsed = 2.0;
    if (message) *message = "extract ok";
    return true;
  };
  assert(controller.phase() == ExtractMonitorPhase::ReadyForIk);
  assert(controller.runNext(callbacks, [&] { return last_elapsed; }, &message));
  assert(controller.phase() == ExtractMonitorPhase::ReadyForExtract);
  assert(controller.runNext(callbacks, [&] { return last_elapsed; }, &message));
  assert(controller.phase() == ExtractMonitorPhase::ReadyForLoaded);
  assert(controller.runNext(callbacks, [&] { return last_elapsed; }, &message));
  assert(controller.phase() == ExtractMonitorPhase::ReadyForFinal);
  assert(controller.runNext(callbacks, [&] { return last_elapsed; }, &message));
  assert(controller.phase() == ExtractMonitorPhase::Done);
  assert(controller.runNext(callbacks, [&] { return last_elapsed; }, &message));
  assert(controller.phase() == ExtractMonitorPhase::ReadyForExtract);

  controller.reset();
  const auto full_from_controller = controller.runFull(callbacks, [&] { return last_elapsed; });
  assert(full_from_controller.success);
  assert(controller.phase() == ExtractMonitorPhase::ReadyForIk);

  AttachedBoxSpec left_box;
  left_box.id = "left_box";
  AttachedBoxSpec right_box;
  right_box.id = "right_box";
  auto state = make_extract_monitor_initial_state(
    6,
    8,
    left_box,
    right_box,
    {},
    {});
  assert(extract_monitor_prefix(6, 8) == "extract_monitor_L6_R8");
  assert(state.left_box_id == 6);
  assert(state.right_box_id == 8);
  assert(state.prefix == "extract_monitor_L6_R8");
  assert(state.left_box.id == "left_box");
  assert(state.right_box.id == "right_box");
  assert(!state.seed_state);
  assert(!state.loaded_start_state);

  state.legal_candidates.resize(3);
  size_t built_count = 0;
  populate_extract_monitor_candidate_states(
    state,
    [&](const ik_benchmark::UpdownAwareIkCandidate&) {
      ++built_count;
      return moveit::core::RobotStatePtr{};
    });
  assert(built_count == 3);
  assert(state.candidate_states.size() == 3);
  populate_extract_monitor_candidate_states(state, {});
  assert(state.candidate_states.empty());

  std::vector<alfa_robot::motion::ExtractRolloutTiming> timings(4);
  moveit::core::RobotStatePtr fake_final_state(
    reinterpret_cast<moveit::core::RobotState*>(0x1),
    [](moveit::core::RobotState*) {});
  timings[0].success = true;
  timings[0].final_state = fake_final_state;
  timings[1].success = true;
  timings[1].failure_reason = "missing_final_state";
  timings[2].failure_reason = "collision";
  const auto timing_summary = summarize_extract_monitor_timings(timings);
  assert(timing_summary.success_count == 1);
  assert(timing_summary.success_indices.size() == 1);
  assert(timing_summary.success_indices[0] == 0);
  assert(timing_summary.failure_counts.at("missing_final_state") == 1);
  assert(timing_summary.failure_counts.at("collision") == 1);
  assert(timing_summary.failure_counts.at("unknown") == 1);

  timings[0].loaded_plan_attempted = true;
  timings[0].loaded_plan_success = true;
  timings[1].loaded_plan_attempted = true;
  timings[1].loaded_plan_failure_reason = "rrt_failed";
  timings[2].loaded_plan_attempted = true;
  const auto loaded_summary = summarize_loaded_plan_timings(timings, {0, 1, 2, 99});
  assert(loaded_summary.attempted_count == 3);
  assert(loaded_summary.success_count == 1);
  assert(loaded_summary.attempted_indices.size() == 3);
  assert(loaded_summary.success_indices.size() == 1);
  assert(loaded_summary.success_indices[0] == 0);
  assert(loaded_summary.failure_counts.at("rrt_failed") == 1);
  assert(loaded_summary.failure_counts.at("unknown") == 1);

  return 0;
}
