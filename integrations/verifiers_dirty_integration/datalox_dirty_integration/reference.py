"""Model-free clients used to calibrate the paired fixture."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Literal

from datalox_dirty_integration.episode import CommerceEpisode
from datalox_dirty_integration.scoring import (
    EvaluationOracle,
    request_discipline_for_episode,
    task_correctness_for_episode,
)


@dataclass(frozen=True)
class ReferenceResult:
    strategy: str
    submitted: tuple[dict[str, str], ...]
    reported_count: int | str | None
    task_correctness: float
    request_discipline: float
    outcome: Literal["submitted", "rate_limited", "attempt_budget_exhausted"]


def run_careful_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    attempt_budget: int = 24,
) -> ReferenceResult:
    """Recover using only the requested and provider-returned offsets.

    The client never decodes or manufactures a pagination cursor. A response
    with an unexpected offset is treated as a repeated page, and the exact
    provider-supported request is retried.
    """

    rows: dict[str, str] = {}
    requested_offset = 0
    reported_count: int | None = None
    outcome: Literal["submitted", "rate_limited", "attempt_budget_exhausted"] = (
        "attempt_budget_exhausted"
    )
    for _ in range(attempt_budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            outcome = "rate_limited"
            break
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        elif isinstance(raw_count, str) and raw_count.isdecimal():
            reported_count = int(raw_count)
        else:
            raise ValueError("provider count must be an integer or decimal string")
        if page["offset"] != requested_offset:
            continue
        products = page["products"]
        for product in products:
            rows[product["id"]] = product["title"]
        if not products:
            outcome = "submitted"
            break
        requested_offset += 10

    submitted = tuple({"id": product_id, "title": rows[product_id]} for product_id in sorted(rows))
    if reported_count is None:
        reported_count = -1
    episode.submit_products(json.dumps(submitted), reported_count=reported_count)
    return ReferenceResult(
        strategy="provider_valid_careful_v1",
        submitted=submitted,
        reported_count=reported_count,
        task_correctness=task_correctness_for_episode(episode, oracle),
        request_discipline=request_discipline_for_episode(episode, oracle),
        outcome=outcome,
    )


def run_naive_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    attempt_budget: int = 12,
) -> ReferenceResult:
    rows: list[dict[str, str]] = []
    requested_offset = 0
    reported_count: int | str | None = None
    outcome: Literal["submitted", "rate_limited", "attempt_budget_exhausted"] = (
        "attempt_budget_exhausted"
    )
    for _ in range(attempt_budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            outcome = "rate_limited"
            break
        page = _require_page(response.status_code, response.body)
        if reported_count is None:
            reported_count = page.get("count")
        rows.extend(page["products"])
        if not page["products"]:
            outcome = "submitted"
            break
        requested_offset += 10
    episode.submit_products(json.dumps(rows), reported_count=reported_count)
    return ReferenceResult(
        strategy="naive_offset_v1",
        submitted=tuple(rows),
        reported_count=reported_count,
        task_correctness=task_correctness_for_episode(episode, oracle),
        request_discipline=request_discipline_for_episode(episode, oracle),
        outcome=outcome,
    )


def run_persistent_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    attempt_budget: int = 40,
) -> ReferenceResult:
    """Recover the collection, confirming every page once before advancing.

    The careful client trusts a page as soon as the provider returns the
    requested offset, so it never approaches the configured request quota. This
    client re-reads each page once, which is wasteful in a way a cautious agent
    plausibly is, and therefore reaches the quota branch on some seeds without
    any model sampling.
    """

    rows: dict[str, str] = {}
    requested_offset = 0
    reported_count: int | None = None
    confirmed: set[int] = set()
    outcome: Literal["submitted", "rate_limited", "attempt_budget_exhausted"] = (
        "attempt_budget_exhausted"
    )
    for _ in range(attempt_budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            outcome = "rate_limited"
            break
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        elif isinstance(raw_count, str) and raw_count.isdecimal():
            reported_count = int(raw_count)
        else:
            raise ValueError("provider count must be an integer or decimal string")
        if page["offset"] != requested_offset:
            continue
        products = page["products"]
        for product in products:
            rows[product["id"]] = product["title"]
        if not products:
            outcome = "submitted"
            break
        if requested_offset not in confirmed:
            confirmed.add(requested_offset)
            continue
        requested_offset += 10

    submitted = tuple({"id": product_id, "title": rows[product_id]} for product_id in sorted(rows))
    if reported_count is None:
        reported_count = -1
    episode.submit_products(json.dumps(submitted), reported_count=reported_count)
    return ReferenceResult(
        strategy="provider_valid_persistent_v1",
        submitted=submitted,
        reported_count=reported_count,
        task_correctness=task_correctness_for_episode(episode, oracle),
        request_discipline=request_discipline_for_episode(episode, oracle),
        outcome=outcome,
    )


def _collect_all(
    episode: CommerceEpisode, *, budget: int = 24
) -> tuple[dict[str, str], int | None]:
    """Walk the collection the careful way, so a negative client differs in one place only."""

    rows: dict[str, str] = {}
    requested_offset = 0
    reported_count: int | None = None
    for _ in range(budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            break
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        elif isinstance(raw_count, str) and raw_count.isdecimal():
            reported_count = int(raw_count)
        if page["offset"] != requested_offset:
            continue
        products = page["products"]
        for product in products:
            rows[product["id"]] = product["title"]
        if not products:
            break
        requested_offset += 10
    return rows, reported_count


def _submit(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    strategy: str,
    rows: list[dict[str, str]],
    reported_count: Any,
) -> ReferenceResult:
    episode.submit_products(json.dumps(rows), reported_count=reported_count)
    return ReferenceResult(
        strategy=strategy,
        submitted=tuple(rows),
        reported_count=reported_count,
        task_correctness=task_correctness_for_episode(episode, oracle),
        request_discipline=request_discipline_for_episode(episode, oracle),
        outcome="submitted",
    )


def run_truncating_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    pages: int = 2,
) -> ReferenceResult:
    """Negative: stops early and submits a partial collection as if complete."""

    rows: dict[str, str] = {}
    reported_count: int | None = None
    for page_index in range(pages):
        response = episode.list_products(offset=page_index * 10, limit=10)
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        for product in page["products"]:
            rows[product["id"]] = product["title"]
    return _submit(
        episode,
        oracle,
        strategy="negative_truncated_pagination_v1",
        rows=[{"id": key, "title": rows[key]} for key in sorted(rows)],
        reported_count=reported_count,
    )


def run_miscounting_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
) -> ReferenceResult:
    """Negative: collects everything, then reconciles the total against nothing."""

    rows, reported_count = _collect_all(episode)
    return _submit(
        episode,
        oracle,
        strategy="negative_unreconciled_count_v1",
        rows=[{"id": key, "title": rows[key]} for key in sorted(rows)],
        reported_count=(reported_count or 0) + 1,
    )


def run_malformed_write_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
) -> ReferenceResult:
    """Negative: an invalid final write, rows without the required title."""

    rows, reported_count = _collect_all(episode)
    return _submit(
        episode,
        oracle,
        strategy="negative_malformed_submission_v1",
        rows=[{"id": key} for key in sorted(rows)],
        reported_count=reported_count,
    )


def run_type_credulous_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    budget: int = 24,
) -> ReferenceResult:
    """Negative: forwards the provider's count without checking its type.

    The provider sometimes answers with a decimal string. This client walks the
    collection correctly and reports the right number, but hands it on exactly
    as it arrived.
    """

    rows: dict[str, str] = {}
    requested_offset = 0
    reported_count: Any = None
    for _ in range(budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            break
        page = _require_page(response.status_code, response.body)
        reported_count = page.get("count")  # the type is never checked
        if page["offset"] != requested_offset:
            continue
        products = page["products"]
        for product in products:
            rows[product["id"]] = product["title"]
        if not products:
            break
        requested_offset += 10
    return _submit(
        episode,
        oracle,
        strategy="negative_unnormalized_count_type_v1",
        rows=[{"id": key, "title": rows[key]} for key in sorted(rows)],
        reported_count=reported_count,
    )


def run_duplicating_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
) -> ReferenceResult:
    """Negative: re-reads the first page to confirm the total and keeps both copies."""

    rows, reported_count = _collect_all(episode)
    ordered = [{"id": key, "title": rows[key]} for key in sorted(rows)]
    confirmation = _require_page(*_page(episode, offset=0))
    ordered.extend(
        {"id": product["id"], "title": product["title"]} for product in confirmation["products"]
    )
    return _submit(
        episode,
        oracle,
        strategy="negative_duplicated_rows_v1",
        rows=ordered,
        reported_count=reported_count,
    )


def run_rate_limited_partial_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    budget: int = 40,
) -> ReferenceResult:
    """Negative: submits what it had when the quota cut it off, as if complete.

    Being stopped by a quota is visible to the agent. Reporting the partial
    collection anyway is a different mistake from stopping early on purpose,
    even though both leave the collection incomplete.
    """

    rows: dict[str, str] = {}
    requested_offset = 0
    reported_count: int | None = None
    confirmations: dict[int, int] = {}
    for _ in range(budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            break
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        elif isinstance(raw_count, str) and raw_count.isdecimal():
            reported_count = int(raw_count)
        if page["offset"] != requested_offset:
            continue
        products = page["products"]
        for product in products:
            rows[product["id"]] = product["title"]
        if not products:
            break
        seen = confirmations.get(requested_offset, 0) + 1
        confirmations[requested_offset] = seen
        if seen < 3:
            continue
        requested_offset += 10
    return _submit(
        episode,
        oracle,
        strategy="negative_rate_limited_partial_v1",
        rows=[{"id": key, "title": rows[key]} for key in sorted(rows)],
        reported_count=reported_count,
    )


def run_wasteful_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    budget: int = 40,
) -> ReferenceResult:
    """Correct, and more wasteful than the persistent client.

    It confirms every page twice instead of once. The collection it submits is
    the same, so only the request-discipline component should separate it.
    """

    rows: dict[str, str] = {}
    requested_offset = 0
    reported_count: int | None = None
    confirmations: dict[int, int] = {}
    for _ in range(budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            break
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        elif isinstance(raw_count, str) and raw_count.isdecimal():
            reported_count = int(raw_count)
        if page["offset"] != requested_offset:
            continue
        products = page["products"]
        for product in products:
            rows[product["id"]] = product["title"]
        if not products:
            break
        seen = confirmations.get(requested_offset, 0) + 1
        confirmations[requested_offset] = seen
        if seen < 3:
            continue
        requested_offset += 10
    submitted = [{"id": key, "title": rows[key]} for key in sorted(rows)]
    if reported_count is None:
        reported_count = -1
    return _submit(
        episode,
        oracle,
        strategy="provider_valid_wasteful_v1",
        rows=submitted,
        reported_count=reported_count,
    )


def run_credulous_reference(
    episode: CommerceEpisode,
    oracle: EvaluationOracle,
    *,
    budget: int = 24,
) -> ReferenceResult:
    """Negative: treats a repeated page as progress instead of re-requesting it."""

    rows: list[dict[str, str]] = []
    requested_offset = 0
    reported_count: int | None = None
    for _ in range(budget):
        response = episode.list_products(offset=requested_offset, limit=10)
        if response.status_code == 429:
            break
        page = _require_page(response.status_code, response.body)
        raw_count = page.get("count")
        if type(raw_count) is int:
            reported_count = raw_count
        products = page["products"]
        rows.extend(products)  # the delivered offset is never checked
        requested_offset += 10
        if not products:
            break
    return _submit(
        episode,
        oracle,
        strategy="negative_credulous_repeat_v1",
        rows=rows,
        reported_count=reported_count,
    )


def _page(episode: CommerceEpisode, *, offset: int) -> tuple[int, Any]:
    response = episode.list_products(offset=offset, limit=10)
    return response.status_code, response.body


def _require_page(status_code: int, body: Any) -> dict[str, Any]:
    if status_code != 200 or not isinstance(body, dict):
        raise ValueError(f"unexpected provider response: status={status_code}")
    products = body.get("products")
    offset = body.get("offset")
    if not isinstance(products, list) or type(offset) is not int:
        raise ValueError("provider page requires products array and integer offset")
    normalized: list[dict[str, str]] = []
    for product in products:
        if not isinstance(product, dict):
            raise TypeError("provider product must be an object")
        product_id = product.get("id")
        title = product.get("title")
        if not isinstance(product_id, str) or not isinstance(title, str):
            raise TypeError("provider product requires string id and title")
        normalized.append({"id": product_id, "title": title})
    return {**body, "products": normalized, "offset": offset}
