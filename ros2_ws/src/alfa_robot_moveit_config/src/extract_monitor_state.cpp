#include "alfa_robot_moveit_config/extract_monitor_state.hpp"

#include <chrono>
#include <sstream>
#include <utility>

namespace alfa_robot::motion
{

namespace
{

ExtractMonitorStageRunner runner_for_stage(
  ExtractMonitorStage stage,
  const ExtractMonitorStageCallbacks& callbacks)
{
  switch (stage) {
    case ExtractMonitorStage::Ik:
      return callbacks.ik;
    case ExtractMonitorStage::Extract:
      return callbacks.extract;
    case ExtractMonitorStage::Loaded:
      return callbacks.loaded;
    case ExtractMonitorStage::Final:
      return callbacks.final;
  }
  return {};
}

size_t stage_index(ExtractMonitorStage stage)
{
  switch (stage) {
    case ExtractMonitorStage::Ik:
      return 0;
    case ExtractMonitorStage::Extract:
      return 1;
    case ExtractMonitorStage::Loaded:
      return 2;
    case ExtractMonitorStage::Final:
      return 3;
  }
  return 0;
}

bool timing_has_loaded_plan(const ExtractRolloutTiming& timing)
{
  return timing.loaded_plan_success && timing.final_state;
}

}  // namespace

std::string extract_monitor_prefix(int left_box_id, int right_box_id)
{
  return "extract_monitor_L" + std::to_string(left_box_id) +
         "_R" + std::to_string(right_box_id);
}

ExtractMonitorState make_extract_monitor_initial_state(
  int left_box_id,
  int right_box_id,
  AttachedBoxSpec left_box,
  AttachedBoxSpec right_box,
  moveit::core::RobotStatePtr seed_state,
  moveit::core::RobotStatePtr loaded_start_state)
{
  ExtractMonitorState state;
  state.left_box_id = left_box_id;
  state.right_box_id = right_box_id;
  state.left_box = std::move(left_box);
  state.right_box = std::move(right_box);
  state.prefix = extract_monitor_prefix(left_box_id, right_box_id);
  state.seed_state = std::move(seed_state);
  state.loaded_start_state = std::move(loaded_start_state);
  return state;
}

void populate_extract_monitor_candidate_states(
  ExtractMonitorState& state,
  const ExtractMonitorCandidateStateBuilder& state_builder)
{
  state.candidate_states.clear();
  state.candidate_states.reserve(state.legal_candidates.size());
  if (!state_builder) {
    return;
  }
  for (const auto& candidate : state.legal_candidates) {
    state.candidate_states.push_back(state_builder(candidate));
  }
}

ExtractMonitorTimingSummary summarize_extract_monitor_timings(
  const std::vector<ExtractRolloutTiming>& timings)
{
  ExtractMonitorTimingSummary summary;
  for (size_t i = 0; i < timings.size(); ++i) {
    const auto& timing = timings[i];
    if (timing.success && timing.final_state) {
      ++summary.success_count;
      summary.success_indices.push_back(i);
    } else {
      summary.failure_counts[timing.failure_reason.empty() ? "unknown" : timing.failure_reason]++;
    }
  }
  return summary;
}

ExtractMonitorLoadedPlanSummary summarize_loaded_plan_timings(
  const std::vector<ExtractRolloutTiming>& timings,
  const std::vector<size_t>& plan_indices)
{
  ExtractMonitorLoadedPlanSummary summary;
  for (const auto index : plan_indices) {
    if (index >= timings.size()) {
      continue;
    }
    const auto& timing = timings[index];
    if (timing.loaded_plan_attempted) {
      ++summary.attempted_count;
      summary.attempted_indices.push_back(index);
    }
    if (timing.loaded_plan_success && timing.final_state) {
      ++summary.success_count;
      summary.success_indices.push_back(index);
    } else if (timing.loaded_plan_attempted) {
      summary.failure_counts[
        timing.loaded_plan_failure_reason.empty() ? "unknown" : timing.loaded_plan_failure_reason]++;
    }
  }
  return summary;
}

ExtractRolloutTiming* select_extract_monitor_final_timing(
  std::vector<ExtractRolloutTiming>& timings,
  const ExtractMonitorTimingPredicate& preferred_predicate)
{
  if (preferred_predicate) {
    for (auto& timing : timings) {
      if (timing_has_loaded_plan(timing) && preferred_predicate(timing)) {
        return &timing;
      }
    }
  }
  for (auto& timing : timings) {
    if (timing_has_loaded_plan(timing)) {
      return &timing;
    }
  }
  return nullptr;
}

const char* extract_monitor_stage_failure_label(ExtractMonitorStage stage)
{
  switch (stage) {
    case ExtractMonitorStage::Ik:
      return "IK阶段";
    case ExtractMonitorStage::Extract:
      return "抽离阶段";
    case ExtractMonitorStage::Loaded:
      return "负重规划阶段";
    case ExtractMonitorStage::Final:
      return "最终选择阶段";
  }
  return "未知阶段";
}

ExtractMonitorStage extract_monitor_stage_for_phase(ExtractMonitorPhase phase)
{
  switch (phase) {
    case ExtractMonitorPhase::ReadyForIk:
    case ExtractMonitorPhase::Done:
      return ExtractMonitorStage::Ik;
    case ExtractMonitorPhase::ReadyForExtract:
      return ExtractMonitorStage::Extract;
    case ExtractMonitorPhase::ReadyForLoaded:
      return ExtractMonitorStage::Loaded;
    case ExtractMonitorPhase::ReadyForFinal:
      return ExtractMonitorStage::Final;
  }
  return ExtractMonitorStage::Ik;
}

ExtractMonitorPhase extract_monitor_phase_before_running(ExtractMonitorPhase phase)
{
  if (phase == ExtractMonitorPhase::Done) {
    return ExtractMonitorPhase::ReadyForIk;
  }
  return phase;
}

ExtractMonitorPhase extract_monitor_next_phase_after(ExtractMonitorStage stage)
{
  switch (stage) {
    case ExtractMonitorStage::Ik:
      return ExtractMonitorPhase::ReadyForExtract;
    case ExtractMonitorStage::Extract:
      return ExtractMonitorPhase::ReadyForLoaded;
    case ExtractMonitorStage::Loaded:
      return ExtractMonitorPhase::ReadyForFinal;
    case ExtractMonitorStage::Final:
      return ExtractMonitorPhase::Done;
  }
  return ExtractMonitorPhase::ReadyForIk;
}

bool run_extract_monitor_stage(
  ExtractMonitorStage stage,
  const ExtractMonitorStageCallbacks& callbacks,
  const std::function<double()>& last_stage_ms,
  double* elapsed_ms,
  std::string* message)
{
  const auto runner = runner_for_stage(stage, callbacks);
  if (!runner) {
    if (message) {
      *message = "extract monitor: missing stage runner";
    }
    if (elapsed_ms) {
      *elapsed_ms = 0.0;
    }
    return false;
  }
  const bool ok = runner(message);
  if (elapsed_ms) {
    *elapsed_ms = last_stage_ms ? last_stage_ms() : 0.0;
  }
  return ok;
}

ExtractMonitorFullRunResult run_extract_monitor_full_sequence(
  const ExtractMonitorStageCallbacks& callbacks,
  const std::function<double()>& last_stage_ms)
{
  ExtractMonitorFullRunResult result;
  const auto total_start = std::chrono::steady_clock::now();
  constexpr std::array<ExtractMonitorStage, 4> stages{
    ExtractMonitorStage::Ik,
    ExtractMonitorStage::Extract,
    ExtractMonitorStage::Loaded,
    ExtractMonitorStage::Final,
  };

  for (const auto stage : stages) {
    std::string stage_message;
    double elapsed_ms = 0.0;
    if (!run_extract_monitor_stage(stage, callbacks, last_stage_ms, &elapsed_ms, &stage_message)) {
      result.message = std::string("完整流程失败在") + extract_monitor_stage_failure_label(stage) + ": " + stage_message;
      result.success = false;
      return result;
    }
    result.stage_elapsed_ms[stage_index(stage)] = elapsed_ms;
  }

  result.total_elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - total_start).count();
  std::ostringstream out;
  out << "完整流程完成: total=" << result.total_elapsed_ms << "ms"
      << " ik=" << result.stage_elapsed_ms[0] << "ms"
      << " extract=" << result.stage_elapsed_ms[1] << "ms"
      << " loaded=" << result.stage_elapsed_ms[2] << "ms"
      << " final=" << result.stage_elapsed_ms[3] << "ms";
  result.message = out.str();
  result.success = true;
  return result;
}

bool ExtractMonitorController::runNext(
  const ExtractMonitorStageCallbacks& callbacks,
  const std::function<double()>& last_stage_ms,
  std::string* message)
{
  const auto stage = extract_monitor_stage_for_phase(phase_);
  phase_ = extract_monitor_phase_before_running(phase_);
  double elapsed_ms = 0.0;
  const bool ok = run_extract_monitor_stage(stage, callbacks, last_stage_ms, &elapsed_ms, message);
  if (ok) {
    phase_ = extract_monitor_next_phase_after(stage);
  }
  return ok;
}

ExtractMonitorFullRunResult ExtractMonitorController::runFull(
  const ExtractMonitorStageCallbacks& callbacks,
  const std::function<double()>& last_stage_ms)
{
  reset();
  ExtractMonitorFullRunResult result;
  const auto total_start = std::chrono::steady_clock::now();
  constexpr std::array<ExtractMonitorStage, 4> stages{
    ExtractMonitorStage::Ik,
    ExtractMonitorStage::Extract,
    ExtractMonitorStage::Loaded,
    ExtractMonitorStage::Final,
  };

  for (const auto stage : stages) {
    std::string stage_message;
    double elapsed_ms = 0.0;
    if (!run_extract_monitor_stage(stage, callbacks, last_stage_ms, &elapsed_ms, &stage_message)) {
      result.message = std::string("完整流程失败在") + extract_monitor_stage_failure_label(stage) + ": " + stage_message;
      result.success = false;
      return result;
    }
    result.stage_elapsed_ms[stage_index(stage)] = elapsed_ms;
    phase_ = extract_monitor_next_phase_after(stage);
  }

  result.total_elapsed_ms = std::chrono::duration<double, std::milli>(
    std::chrono::steady_clock::now() - total_start).count();
  std::ostringstream out;
  out << "完整流程完成: total=" << result.total_elapsed_ms << "ms"
      << " ik=" << result.stage_elapsed_ms[0] << "ms"
      << " extract=" << result.stage_elapsed_ms[1] << "ms"
      << " loaded=" << result.stage_elapsed_ms[2] << "ms"
      << " final=" << result.stage_elapsed_ms[3] << "ms";
  result.message = out.str();
  result.success = true;
  reset();
  return result;
}

}  // namespace alfa_robot::motion
