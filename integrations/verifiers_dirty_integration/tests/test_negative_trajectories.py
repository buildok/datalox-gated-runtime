"""Negative trajectories and the verifier errors they are meant to expose.

A reward tells you how well a trajectory did. It does not tell you whether the
verifier would have caught a bad one, so each declared negative names the
invariant it breaks and the suite reports acceptance errors separately from
reward.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from datalox_dirty_integration.calibration import run_calibration
from datalox_dirty_integration.negative_suite import (
    DECLARED,
    run_declared,
    run_suite,
    run_trace,
)
from datalox_dirty_integration.policy import INTERVENTION_KINDS, PROFILES
from datalox_dirty_integration.rerun import compare_runs
from datalox_dirty_integration.verifier import INVARIANT_NAMES

NEGATIVES = [item for item in DECLARED if item.expected_failure is not None]
VALID = [item for item in DECLARED if item.expected_failure is None]


@pytest.mark.parametrize("trajectory", NEGATIVES, ids=lambda item: item.name)
def test_declared_negative_fails_its_own_invariant(trajectory) -> None:
    row = run_declared(trajectory)
    assert row["invariants"][trajectory.expected_failure] is False
    assert trajectory.expected_failure in row["failed_invariants"]


@pytest.mark.parametrize("trajectory", VALID, ids=lambda item: item.name)
def test_valid_trajectory_passes_every_invariant(trajectory) -> None:
    row = run_declared(trajectory)
    assert row["failed_invariants"] == []
    assert row["task_correctness"] == 1.0


def test_every_invariant_is_covered_by_a_declared_negative() -> None:
    """A named invariant nothing can break is not evidence of anything."""

    covered = {item.expected_failure for item in NEGATIVES}
    assert set(INVARIANT_NAMES) == covered


def test_incomplete_collections_are_distinguished_by_cause() -> None:
    """Stopping early and being cut off both leave gaps, for different reasons."""

    truncating = run_declared(next(item for item in NEGATIVES if item.name == "truncating"))
    cut_off = run_declared(next(item for item in NEGATIVES if item.name == "rate_limited_partial"))
    assert (
        truncating["failed_invariants"] == cut_off["failed_invariants"] == ["pagination_complete"]
    )
    assert truncating["delivered_calls"] < cut_off["delivered_calls"]
    assert cut_off["request_discipline"] < truncating["request_discipline"]


def test_suite_reports_acceptance_errors_apart_from_reward() -> None:
    report = run_suite()
    assert report["false_acceptance"] == []
    assert report["false_rejection"] == []
    assert {row["name"] for row in report["trajectories"]} == {item.name for item in DECLARED}


def test_request_discipline_orders_three_correct_trajectories() -> None:
    """Rejecting invalid work is not the same as ranking valid work."""

    report = run_suite()
    ladder = report["discipline_ladder"]
    assert ladder["all_task_correct"] is True
    assert ladder["monotonic"] is True
    assert [row["name"] for row in ladder["ordered_by_calls"]] == [
        "careful",
        "persistent",
        "wasteful",
    ]

    pair = report["discipline_pair"]
    assert pair["both_task_correct"] is True
    assert pair["separated"] is True
    assert pair["careful"]["request_discipline"] > pair["persistent"]["request_discipline"]
    assert pair["careful"]["delivered_calls"] < pair["persistent"]["delivered_calls"]


def test_every_intervention_branch_is_reached_by_a_declared_trajectory() -> None:
    """A configured branch nothing reaches is not calibrated by this suite."""

    reached = set(run_suite()["intervention_branches_reached"])
    assert reached == {"quota_response", "repeat_page", "json_type_drift"}


def test_documented_profiles_match_the_policy() -> None:
    """The profile table is evidence only while it matches the code."""

    table = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")
    rows = re.findall(
        r"^\| `(clean|realistic|hostile)` \| (.+?) \| (.+?) \| (.+?) \|$", table, re.M
    )
    assert {row[0] for row in rows} == set(PROFILES)
    for name, repeat, drift, quota in rows:
        profile = PROFILES[name]
        assert profile.repeat_page_rate == float(repeat)
        assert profile.count_type_drift_rate == float(drift)
        assert profile.request_quota == (None if quota == "none" else int(quota))


def test_profile_names_do_not_imply_a_single_ordering() -> None:
    """`hostile` is harsher on faults and more permissive on the request budget."""

    realistic, hostile = PROFILES["realistic"], PROFILES["hostile"]
    assert hostile.repeat_page_rate > realistic.repeat_page_rate
    assert hostile.count_type_drift_rate > realistic.count_type_drift_rate
    assert hostile.request_quota > realistic.request_quota


def test_reruns_of_a_seed_differ_only_by_incidental_identifiers() -> None:
    """Reproducing a reported seed should not require reading past random ids."""

    trajectory = next(item for item in DECLARED if item.name == "credulous")
    first, second = run_trace(trajectory), run_trace(trajectory)

    assert first != second
    comparison = compare_runs(first, second)
    assert comparison["identical"] is True
    assert comparison["canonical_differences"] == []
    assert comparison["incidental_differences"]
    assert all(path.endswith("event_id") for path in comparison["incidental_differences"])


def test_canonical_comparison_still_separates_different_seeds() -> None:
    """Dropping identifiers must not drop the schedule."""

    trajectory = next(item for item in DECLARED if item.name == "credulous")
    other = replace(trajectory, seed="8")
    comparison = compare_runs(run_trace(trajectory), run_trace(other))
    assert comparison["identical"] is False
    assert comparison["canonical_differences"]


def test_one_command_reproduces_the_calibration_and_stays_short() -> None:
    """A summary nobody can paste into an issue is not a summary."""

    summary = run_calibration()
    assert summary["passed"] is True
    assert {pair["profile"] for pair in summary["pairs"]} == set(PROFILES)
    assert all(pair["bindings_hold"] for pair in summary["pairs"])
    # Short enough to paste into an issue without folding. The bound moves only
    # when the summary genuinely gains evidence, never to accommodate noise.
    assert len(json.dumps(summary, indent=2).splitlines()) < 275


def test_canonical_comparison_keeps_an_event_id_inside_a_provider_payload() -> None:
    """Only the envelope identifiers are noise; a provider field is an observation."""

    trace = {
        "events": [
            {
                "event_id": "intv_1",
                "base": {"event_id": "evt_1", "response_sha256": None},
                "delivered": {"response": {"body": {"event_id": "evt_a", "status": "paid"}}},
            }
        ]
    }
    other = json.loads(json.dumps(trace))
    other["events"][0]["event_id"] = "intv_2"
    other["events"][0]["base"]["event_id"] = "evt_2"
    assert compare_runs(trace, other)["identical"] is True

    changed = json.loads(json.dumps(trace))
    changed["events"][0]["delivered"]["response"]["body"]["event_id"] = "evt_b"
    comparison = compare_runs(trace, changed)
    assert comparison["identical"] is False
    assert comparison["canonical_differences"] == [".events[0].delivered.response.body.event_id"]


def test_calibration_gate_requires_every_intervention_branch() -> None:
    """The command README calls the whole calibration has to check H4 too."""

    summary = run_calibration()
    assert summary["intervention_branches_complete"] is True
    assert set(summary["intervention_branches_reached"]) == set(INTERVENTION_KINDS)
    assert summary["intervention_branches_complete"] is True
    assert summary["passed"] is True


def test_action_counts_separate_applied_from_observed() -> None:
    """An applied repeat can leave the delivered observation unchanged."""

    counts = run_suite()["actions_by_profile"]
    realistic = counts["realistic"]["repeat_page"]
    assert realistic["applied"] == realistic["decided"]
    assert realistic["observational_noop"] > 0
    assert (
        realistic["observation_changed"] + realistic["observational_noop"] == realistic["applied"]
    )


def test_summary_names_the_verdict_the_component_and_the_reason() -> None:
    """Reading the protocol item by item should not require reconstruction."""

    summary = run_calibration()
    row = next(item for item in summary["trajectories"] if item["name"] == "truncating")
    assert row["verifier_passed"] is False
    assert row["execution_outcome"] == "submitted"

    ladder = summary["discipline_ladder"]
    assert ladder["component"] == "request_discipline"
    assert ladder["reason"] == "delivered provider call count"


def test_calibration_gate_includes_the_rerun_comparison() -> None:
    """A summary that claims rerun determinism has to check it."""

    summary = run_calibration()
    assert summary["rerun"]["identical"] is True
    assert summary["rerun"]["canonical_differences"] == []
    assert summary["rerun"]["incidental_differences"] > 0
    assert summary["passed"] is True
