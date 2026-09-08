"""Declared trajectories and the report that separates verifier errors from reward.

A reward says how well a trajectory did. It does not say whether the verifier
would have caught a bad one, so this module runs a fixed set of trajectories,
each declaring the invariant it is built to break, and reports two counts that
have nothing to do with reward: negatives the verifier let through, and valid
work it rejected.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from datalox_dirty_integration.contract import (
    provider_admission_path,
    provider_config_path,
    provider_release_path,
    provider_runtime_bundle_path,
)
from datalox_dirty_integration.episode import CommerceEpisode
from datalox_dirty_integration.policy import (
    INTERVENTION_KINDS,
    SeededCommercePolicy,
    load_profile,
)
from datalox_dirty_integration.reference import (
    ReferenceResult,
    run_careful_reference,
    run_credulous_reference,
    run_duplicating_reference,
    run_malformed_write_reference,
    run_miscounting_reference,
    run_persistent_reference,
    run_rate_limited_partial_reference,
    run_truncating_reference,
    run_type_credulous_reference,
    run_wasteful_reference,
)
from datalox_dirty_integration.scoring import EvaluationOracle
from datalox_dirty_integration.verifier import evaluate_invariants

SCHEMA_VERSION = "datalox_negative_trajectory_report_v1"


@dataclass(frozen=True)
class DeclaredTrajectory:
    """One model-free client plus the invariant it is expected to break."""

    name: str
    run: Callable[[CommerceEpisode, EvaluationOracle], ReferenceResult]
    profile: str
    seed: str
    expected_failure: str | None


#: `clean` isolates a task mistake from the intervention layer. The negatives
#: that depend on intervention-visible behavior use the profile and seed that
#: expose their target fault deterministically.
DECLARED: tuple[DeclaredTrajectory, ...] = (
    DeclaredTrajectory("careful", run_careful_reference, "clean", "1", None),
    DeclaredTrajectory("persistent", run_persistent_reference, "clean", "1", None),
    DeclaredTrajectory("wasteful", run_wasteful_reference, "clean", "1", None),
    DeclaredTrajectory("truncating", run_truncating_reference, "clean", "1", "pagination_complete"),
    DeclaredTrajectory("miscounting", run_miscounting_reference, "clean", "1", "count_reconciled"),
    DeclaredTrajectory(
        "malformed_write", run_malformed_write_reference, "clean", "1", "submission_well_formed"
    ),
    DeclaredTrajectory("credulous", run_credulous_reference, "hostile", "7", "repeat_reconciled"),
    DeclaredTrajectory("duplicating", run_duplicating_reference, "clean", "1", "no_duplicate_rows"),
    DeclaredTrajectory(
        "rate_limited_partial",
        run_rate_limited_partial_reference,
        "realistic",
        "3",
        "pagination_complete",
    ),
    DeclaredTrajectory(
        "type_credulous",
        run_type_credulous_reference,
        "hostile",
        "7",
        "count_type_normalized",
    ),
)


def action_counts(trace: dict[str, Any]) -> dict[str, dict[str, int]]:
    """Per action kind: decided, applied, observed, and observational no-ops.

    A no-op is an applied action whose delivered response carries the same
    canonical digest as the base response, so the agent saw nothing new. The
    digests are already in the trace, which is why this does not need the
    event-level `observation_changed` field.
    """

    counts: dict[str, dict[str, int]] = {
        kind: {"decided": 0, "applied": 0, "observation_changed": 0, "observational_noop": 0}
        for kind in INTERVENTION_KINDS
    }
    for event in trace["events"]:
        kind = event["decision"]["kind"]
        if kind not in counts:
            continue
        row = counts[kind]
        row["decided"] += 1
        if not event["applied"]:
            continue
        row["applied"] += 1
        delivered = event["delivered"]
        base_digest = event["base"]["response_sha256"]
        delivered_digest = None if delivered is None else delivered["response_sha256"]
        if delivered_digest != base_digest:
            row["observation_changed"] += 1
        else:
            row["observational_noop"] += 1
    return {kind: row for kind, row in counts.items() if row["decided"]}


def _episode(profile: str, seed: str) -> CommerceEpisode:
    return CommerceEpisode(
        provider_config=provider_config_path(),
        policy=SeededCommercePolicy(load_profile(profile)),
        intervention_seed=seed,
        intervention_enabled=True,
        provider_admission=provider_admission_path(),
        provider_runtime_bundle=provider_runtime_bundle_path(),
        provider_release=provider_release_path(),
    )


def run_declared(trajectory: DeclaredTrajectory) -> dict[str, Any]:
    with _episode(trajectory.profile, trajectory.seed) as episode:
        oracle = EvaluationOracle.from_provider_config(provider_config_path())
        result = trajectory.run(episode, oracle)
        invariants = evaluate_invariants(episode, oracle)
        trace = episode.export()["intervention"]
        return {
            "intervention_kinds": sorted(
                {event["decision"]["kind"] for event in trace["events"]} - {"none"}
            ),
            "actions": action_counts(trace),
            "name": trajectory.name,
            "strategy": result.strategy,
            "profile": trajectory.profile,
            "seed": trajectory.seed,
            "expected_failure": trajectory.expected_failure,
            "invariants": invariants,
            "failed_invariants": sorted(name for name, ok in invariants.items() if not ok),
            "delivered_calls": len(episode.delivered_calls),
            "task_correctness": result.task_correctness,
            "request_discipline": result.request_discipline,
            "outcome": result.outcome,
        }


def run_trace(trajectory: DeclaredTrajectory) -> dict[str, Any]:
    """The intervention trace a declared trajectory produces, for rerun comparison."""

    with _episode(trajectory.profile, trajectory.seed) as episode:
        oracle = EvaluationOracle.from_provider_config(provider_config_path())
        trajectory.run(episode, oracle)
        return episode.export()["intervention"]


def run_suite() -> dict[str, Any]:
    """Run every declared trajectory and report verifier errors apart from reward."""

    rows = [run_declared(trajectory) for trajectory in DECLARED]
    by_name = {row["name"]: row for row in rows}
    reached = sorted({kind for row in rows for kind in row["intervention_kinds"]})
    actions_by_profile: dict[str, dict[str, dict[str, int]]] = {}
    for row in rows:
        profile = actions_by_profile.setdefault(row["profile"], {})
        for kind, counts in row["actions"].items():
            totals = profile.setdefault(
                kind,
                {"decided": 0, "applied": 0, "observation_changed": 0, "observational_noop": 0},
            )
            for key, value in counts.items():
                totals[key] += value

    false_acceptance = [
        row["name"]
        for row in rows
        if row["expected_failure"] is not None
        and row["invariants"][row["expected_failure"]] is True
    ]
    false_rejection = [
        row["name"] for row in rows if row["expected_failure"] is None and row["failed_invariants"]
    ]

    ladder = [by_name[name] for name in ("careful", "persistent", "wasteful")]
    careful, persistent = ladder[0], ladder[1]
    return {
        "schema_version": SCHEMA_VERSION,
        "trajectories": rows,
        "intervention_branches_reached": reached,
        "actions_by_profile": actions_by_profile,
        "false_acceptance": false_acceptance,
        "false_rejection": false_rejection,
        "discipline_pair": {
            "both_task_correct": careful["task_correctness"]
            == persistent["task_correctness"]
            == 1.0,
            "separated": careful["request_discipline"] != persistent["request_discipline"],
            "careful": {
                "request_discipline": careful["request_discipline"],
                "delivered_calls": careful["delivered_calls"],
            },
            "persistent": {
                "request_discipline": persistent["request_discipline"],
                "delivered_calls": persistent["delivered_calls"],
            },
            "reason": "delivered provider call count",
        },
        "discipline_ladder": {
            "component": "request_discipline",
            "reason": "delivered provider call count",
            "ordered_by_calls": [
                {
                    "name": row["name"],
                    "delivered_calls": row["delivered_calls"],
                    "request_discipline": row["request_discipline"],
                    "task_correctness": row["task_correctness"],
                }
                for row in ladder
            ],
            "monotonic": all(
                first["delivered_calls"] < second["delivered_calls"]
                and first["request_discipline"] > second["request_discipline"]
                for first, second in zip(ladder, ladder[1:])
            ),
            "all_task_correct": all(row["task_correctness"] == 1.0 for row in ladder),
        },
    }
