"""One command that reproduces the public calibration and summarises it.

The pair runner writes full evidence directories, which is what you want when
reporting a seed. This runs the whole deterministic calibration instead - the
off/on pair for every profile plus the declared trajectories - and prints a
summary short enough to paste into an issue.
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from datalox_dirty_integration.contract import (
    provider_admission_path,
    provider_config_path,
    provider_release_path,
    provider_runtime_bundle_path,
)
from datalox_dirty_integration.negative_suite import DECLARED, run_suite, run_trace
from datalox_dirty_integration.paired import run_pair
from datalox_dirty_integration.policy import INTERVENTION_KINDS, PROFILES
from datalox_dirty_integration.rerun import compare_runs

SCHEMA_VERSION = "datalox_public_calibration_summary_v1"


def run_calibration(seed: str = "7") -> dict[str, Any]:
    pairs = []
    with tempfile.TemporaryDirectory() as scratch:
        for profile in PROFILES:
            comparison = run_pair(
                output=Path(scratch) / profile,
                provider_config=provider_config_path(),
                provider_admission=provider_admission_path(),
                provider_runtime_bundle=provider_runtime_bundle_path(),
                provider_release=provider_release_path(),
                profile=profile,
                intervention_seed=seed,
            )
            pairs.append(
                {
                    "profile": profile,
                    "bindings_hold": all(comparison["binding_checks"].values()),
                    "off": comparison["off"],
                    "on": comparison["on"],
                    # The full record belongs in the evidence directory the pair
                    # runner writes; a summary only needs where it happened.
                    "first_divergence_index": (
                        None
                        if comparison["first_delivered_observation_divergence"] is None
                        else comparison["first_delivered_observation_divergence"]["index"]
                    ),
                }
            )

    # The same seed run twice has to agree on everything except the identifiers
    # minted per run, otherwise a reported seed cannot be reproduced.
    rerun_of = next(item for item in DECLARED if item.name == "credulous")
    rerun = compare_runs(run_trace(rerun_of), run_trace(rerun_of))

    suite = run_suite()
    branches_complete = set(suite["intervention_branches_reached"]) == set(INTERVENTION_KINDS)
    return {
        "schema_version": SCHEMA_VERSION,
        "seed": seed,
        "pairs": pairs,
        "intervention_branches_reached": suite["intervention_branches_reached"],
        "intervention_branches_complete": branches_complete,
        "actions_by_profile": suite["actions_by_profile"],
        "rerun": {
            "trajectory": rerun_of.name,
            "profile": rerun_of.profile,
            "seed": rerun_of.seed,
            "identical": rerun["identical"],
            "canonical_differences": rerun["canonical_differences"],
            "incidental_differences": len(rerun["incidental_differences"]),
        },
        "trajectories": [
            {
                "name": row["name"],
                "expected_failure": row["expected_failure"],
                # The verifier verdict, kept apart from `outcome`, which says how
                # the run ended rather than whether the work was accepted.
                "verifier_passed": not row["failed_invariants"],
                "failed_invariants": row["failed_invariants"],
                "execution_outcome": row["outcome"],
                "delivered_calls": row["delivered_calls"],
                "task_correctness": row["task_correctness"],
                "request_discipline": row["request_discipline"],
            }
            for row in suite["trajectories"]
        ],
        "false_acceptance": suite["false_acceptance"],
        "false_rejection": suite["false_rejection"],
        "discipline_ladder": suite["discipline_ladder"],
        "passed": (
            all(pair["bindings_hold"] for pair in pairs)
            and branches_complete
            and not suite["false_acceptance"]
            and not suite["false_rejection"]
            and suite["discipline_ladder"]["monotonic"]
            and suite["discipline_ladder"]["all_task_correct"]
            and rerun["identical"]
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--intervention-seed", default="7")
    args = parser.parse_args(argv)
    summary = run_calibration(args.intervention_seed)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
