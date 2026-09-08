"""Named task invariants for the commerce collection task.

The two rewards are scalars: they say how well a trajectory did, not which
invariant it broke. A negative trajectory has to fail for a stated reason, so
the invariants below are checked individually and reported by name. They are
consumer-owned evaluation code and never reach the agent.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from datalox_dirty_integration.episode import CommerceEpisode
from datalox_dirty_integration.scoring import EvaluationOracle

#: Every invariant, in the order a reviewer would read them.
INVARIANT_NAMES = (
    "submission_well_formed",
    "pagination_complete",
    "no_duplicate_rows",
    "count_type_normalized",
    "count_reconciled",
    "repeat_reconciled",
)


@dataclass(frozen=True)
class _Submission:
    rows: tuple[Mapping[str, Any], ...]
    reported_count: Any
    well_formed: bool


def _submission(episode: CommerceEpisode) -> _Submission:
    submitted = episode.submitted
    if not isinstance(submitted, dict):
        return _Submission((), None, False)
    raw = submitted.get("products")
    if not isinstance(raw, list) or not raw:
        return _Submission((), submitted.get("reported_count"), False)
    rows = tuple(row for row in raw if isinstance(row, Mapping))
    well_formed = len(rows) == len(raw) and all(
        isinstance(row.get("id"), str) and isinstance(row.get("title"), str) for row in rows
    )
    return _Submission(rows, submitted.get("reported_count"), well_formed)


def submission_well_formed(episode: CommerceEpisode, oracle: EvaluationOracle) -> bool:
    """The final write is a non-empty array of objects with string id and title."""

    del oracle
    return _submission(episode).well_formed


def pagination_complete(episode: CommerceEpisode, oracle: EvaluationOracle) -> bool:
    """Every product the provider holds appears in the submission, matched by id."""

    submitted = {row.get("id") for row in _submission(episode).rows}
    return {row["id"] for row in oracle.products}.issubset(submitted)


def no_duplicate_rows(episode: CommerceEpisode, oracle: EvaluationOracle) -> bool:
    """No product id is submitted twice, which a repeated page invites."""

    del oracle
    ids = [row.get("id") for row in _submission(episode).rows]
    return len(ids) == len(set(ids))


def count_type_normalized(episode: CommerceEpisode, oracle: EvaluationOracle) -> bool:
    """The reported count is an integer.

    The provider sometimes answers with a decimal string. Passing that straight
    into the final write is a type mistake even when the number is right, so it
    is a separate invariant from whether the value reconciles.
    """

    del oracle
    return type(_submission(episode).reported_count) is int


def count_reconciled(episode: CommerceEpisode, oracle: EvaluationOracle) -> bool:
    """The reported count matches the collection, whatever type it arrived as."""

    reported = _submission(episode).reported_count
    if isinstance(reported, str) and reported.isdecimal():
        reported = int(reported)
    return type(reported) is int and reported == oracle.reported_count


def repeat_reconciled(episode: CommerceEpisode, oracle: EvaluationOracle) -> bool:
    """A page that came back at the wrong offset was requested again.

    A repeated page is visible to the agent: the provider answers with an offset
    it did not ask for. Advancing anyway silently drops a page, so the invariant
    is that every such offset appears in a later request.
    """

    del oracle
    pending: list[int] = []
    for call in episode.delivered_calls:
        observation = call["observation"]
        body = observation.get("body")
        requested = int(call["request"]["query"]["offset"])
        if requested in pending:
            pending.remove(requested)
        if observation.get("status_code") != 200 or not isinstance(body, dict):
            continue
        if body.get("offset") != requested:
            pending.append(requested)
    return not pending


CHECKS = {
    "submission_well_formed": submission_well_formed,
    "pagination_complete": pagination_complete,
    "no_duplicate_rows": no_duplicate_rows,
    "count_type_normalized": count_type_normalized,
    "count_reconciled": count_reconciled,
    "repeat_reconciled": repeat_reconciled,
}


def evaluate_invariants(episode: CommerceEpisode, oracle: EvaluationOracle) -> dict[str, bool]:
    """Every invariant by name, independent of the two reward scalars."""

    return {name: CHECKS[name](episode, oracle) for name in INVARIANT_NAMES}
