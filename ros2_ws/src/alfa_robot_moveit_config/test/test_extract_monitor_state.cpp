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

  return 0;
}
