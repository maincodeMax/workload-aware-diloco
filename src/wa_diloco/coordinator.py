from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from wa_diloco.config import ExperimentConfig, ensure_workspace, load_config
from wa_diloco.policies import ClusterPressure, LearnerReport, build_policy
from wa_diloco.state import freshness_weight, load_state, merge_deltas, save_state
from wa_diloco.telemetry import append_jsonl, read_json, write_json


def _event_path(config: ExperimentConfig) -> Path:
    return config.runtime.workspace / "events" / "coordinator.jsonl"


def _checkpoint_path(config: ExperimentConfig, global_round: int) -> Path:
    return config.runtime.workspace / "checkpoints" / f"global_round_{global_round:04d}.pt"


def _assignment_path(config: ExperimentConfig, control_cycle: int, learner_id: int) -> Path:
    return (
        config.runtime.workspace
        / "assignments"
        / f"cycle_{control_cycle:05d}"
        / f"learner_{learner_id:02d}.json"
    )


def _report_path(config: ExperimentConfig, control_cycle: int, learner_id: int) -> Path:
    return (
        config.runtime.workspace
        / "reports"
        / f"cycle_{control_cycle:05d}"
        / f"learner_{learner_id:02d}.json"
    )


def _cycle_updates_path(config: ExperimentConfig, control_cycle: int) -> Path:
    return config.runtime.workspace / "updates" / f"cycle_{control_cycle:05d}"


def _cleanup_cycle_updates(config: ExperimentConfig, control_cycle: int) -> None:
    shutil.rmtree(_cycle_updates_path(config, control_cycle), ignore_errors=True)


def _cleanup_sync_state(config: ExperimentConfig, previous_checkpoint: Path) -> None:
    try:
        previous_checkpoint.unlink(missing_ok=True)
    except OSError:
        pass
    shutil.rmtree(config.runtime.workspace / "learners", ignore_errors=True)
    (config.runtime.workspace / "learners").mkdir(parents=True, exist_ok=True)


def _load_pressure(config: ExperimentConfig) -> ClusterPressure:
    candidates = []
    if config.runtime.pressure_file:
        candidates.append(config.runtime.pressure_file)
    candidates.append(config.runtime.workspace / "pressure" / "latest.json")
    for path in candidates:
        if path.exists():
            return ClusterPressure.from_mapping(read_json(path))
    return ClusterPressure()


def _write_assignment(
    config: ExperimentConfig,
    control_cycle: int,
    learner_id: int,
    *,
    global_round: int,
    local_steps: int,
    base_state_path: Path,
    continue_local: bool,
    stop: bool = False,
) -> None:
    write_json(
        _assignment_path(config, control_cycle, learner_id),
        {
            "type": "stop" if stop else "train",
            "experiment": config.name,
            "learner_id": learner_id,
            "control_cycle": control_cycle,
            "global_round": global_round,
            "local_steps": local_steps,
            "base_state_path": str(base_state_path),
            "continue_local": continue_local,
        },
    )


def _wait_reports(config: ExperimentConfig, control_cycle: int) -> list[LearnerReport]:
    deadline = time.time() + config.runtime.report_timeout_sec
    expected = list(range(config.learners.count))
    reports: dict[int, LearnerReport] = {}
    while time.time() < deadline:
        for learner_id in expected:
            if learner_id in reports:
                continue
            path = _report_path(config, control_cycle, learner_id)
            if not path.exists():
                continue
            reports[learner_id] = LearnerReport(**read_json(path))
        if len(reports) >= config.learners.min_ready:
            return [reports[i] for i in sorted(reports)]
        time.sleep(config.runtime.poll_interval_sec)
    raise TimeoutError(
        f"only received {len(reports)}/{config.learners.min_ready} reports for cycle {control_cycle}"
    )


def _merge_weight(config: ExperimentConfig, report: LearnerReport) -> float:
    weight = 1.0
    if config.merge.token_weighted:
        weight *= max(1.0, float(report.tokens_since_sync))
    weight *= freshness_weight(report.staleness_sec, config.merge.staleness_half_life_sec)
    return weight


def run(config_path: str | Path) -> None:
    config = load_config(config_path)
    ensure_workspace(config)
    policy = build_policy(config.policy)
    events = _event_path(config)

    initial_checkpoint = _checkpoint_path(config, 0)
    if not initial_checkpoint.exists():
        raise FileNotFoundError(
            f"missing initial checkpoint {initial_checkpoint}; run wa-diloco-init-checkpoint first"
        )

    global_round = 0
    stop_cycle = 0
    continue_local = False
    base_state_path = initial_checkpoint

    append_jsonl(
        events,
        {
            "event": "coordinator_start",
            "experiment": config.name,
            "workspace": str(config.runtime.workspace),
            "policy": config.policy.name,
        },
    )

    for control_cycle in range(config.runtime.max_control_cycles):
        if global_round >= config.runtime.max_global_rounds:
            stop_cycle = control_cycle
            break

        local_steps = config.policy.min_h if continue_local else config.policy.fixed_h
        for learner_id in range(config.learners.count):
            _write_assignment(
                config,
                control_cycle,
                learner_id,
                global_round=global_round,
                local_steps=local_steps,
                base_state_path=base_state_path,
                continue_local=continue_local,
            )

        reports = _wait_reports(config, control_cycle)
        pressure = _load_pressure(config)
        decisions = [(report, policy.decide(report, pressure)) for report in reports]
        sync_ready = [(report, decision) for report, decision in decisions if decision.sync_now]

        append_jsonl(
            events,
            {
                "event": "control_cycle",
                "control_cycle": control_cycle,
                "global_round": global_round,
                "pressure": pressure.__dict__,
                "ready_reports": len(reports),
                "sync_ready": len(sync_ready),
                "decisions": [
                    {
                        "learner_id": report.learner_id,
                        "sync_now": decision.sync_now,
                        "reason": decision.reason,
                        "score": decision.score,
                        "local_steps_since_sync": report.local_steps_since_sync,
                        "tokens_since_sync": report.tokens_since_sync,
                    }
                    for report, decision in decisions
                ],
            },
        )

        if len(sync_ready) < config.learners.min_ready:
            continue_local = True
            stop_cycle = control_cycle + 1
            _cleanup_cycle_updates(config, control_cycle)
            continue

        previous_base_state_path = base_state_path
        base_state = load_state(base_state_path, map_location="cpu")
        weighted_paths = [
            (report.delta_path, _merge_weight(config, report)) for report, _decision in sync_ready
        ]
        merged = merge_deltas(base_state, weighted_paths, max_update_norm=config.merge.max_update_norm)
        global_round += 1
        base_state_path = _checkpoint_path(config, global_round)
        save_state(base_state_path, merged)
        continue_local = False

        append_jsonl(
            events,
            {
                "event": "merged",
                "global_round": global_round,
                "checkpoint": str(base_state_path),
                "contributors": [report.learner_id for report, _decision in sync_ready],
            },
        )
        _cleanup_cycle_updates(config, control_cycle)
        _cleanup_sync_state(config, previous_base_state_path)
        stop_cycle = control_cycle + 1

    for learner_id in range(config.learners.count):
        _write_assignment(
            config,
            stop_cycle,
            learner_id,
            global_round=global_round,
            local_steps=0,
            base_state_path=base_state_path,
            continue_local=False,
            stop=True,
        )

    append_jsonl(events, {"event": "coordinator_stop", "global_round": global_round})


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="Path to experiment YAML")
    args = parser.parse_args(argv)
    run(args.config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
