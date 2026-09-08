"""Comparing two runs of the same seed without the noise.

Every trace event carries an identifier minted per run, so two runs of the same
seed differ on every event while meaning the same thing. A raw diff of two
evidence directories is therefore unreadable, which matters when a failing seed
is reported and someone else has to reproduce it.

Only the envelope identifiers are dropped, and only where the trace envelope
puts them. A provider payload that happens to carry a field called `event_id`
is part of the observation and is compared like any other field.
"""

from __future__ import annotations

from typing import Any

#: Exact locations of the identifiers the runtime mints per run. `*` matches
#: one list index.
INCIDENTAL_PATHS: frozenset[tuple[str, ...]] = frozenset(
    {
        ("events", "*", "event_id"),
        ("events", "*", "base", "event_id"),
        ("events", "*", "decision", "source_base", "event_id"),
    }
)

_MISSING = object()


def _is_incidental(path: tuple[str, ...]) -> bool:
    for pattern in INCIDENTAL_PATHS:
        if len(pattern) != len(path):
            continue
        if all(step == "*" or step == actual for step, actual in zip(pattern, path)):
            return True
    return False


def canonical_trace(value: Any, path: tuple[str, ...] = ()) -> Any:
    """The same structure with the envelope identifiers removed."""

    if isinstance(value, dict):
        return {
            key: canonical_trace(item, (*path, key))
            for key, item in value.items()
            if not _is_incidental((*path, key))
        }
    if isinstance(value, list):
        return [canonical_trace(item, (*path, "*")) for item in value]
    return value


def _differences(left: Any, right: Any, path: str = "") -> list[str]:
    if isinstance(left, dict) and isinstance(right, dict):
        out: list[str] = []
        for key in sorted(set(left) | set(right)):
            first, second = left.get(key, _MISSING), right.get(key, _MISSING)
            if first is _MISSING or second is _MISSING:
                out.append(f"{path}.{key}")
                continue
            out.extend(_differences(first, second, f"{path}.{key}"))
        return out
    if isinstance(left, list) and isinstance(right, list) and len(left) == len(right):
        out = []
        for index, (first, second) in enumerate(zip(left, right)):
            out.extend(_differences(first, second, f"{path}[{index}]"))
        return out
    return [] if left == right else [path or "."]


def compare_runs(left: Any, right: Any) -> dict[str, Any]:
    """Whether two traces agree once the envelope identifiers are dropped."""

    raw = _differences(left, right)
    canonical = _differences(canonical_trace(left), canonical_trace(right))
    return {
        "identical": not canonical,
        "canonical_differences": canonical,
        "incidental_differences": [path for path in raw if path not in canonical],
    }
