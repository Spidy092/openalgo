"""Explicit identity resolution for Upstox suspended-instrument evidence.

The suspended file contains multiple historical/variant rows for one ISIN and
instrument key.  A key-only anti-join therefore cannot establish that the
currently quoted security is suspended.  This module deliberately keeps the
same-segment/key evidence separate from an exact current-token match.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType


SUSPENDED_EXACT = "SUSPENDED_EXACT"
SUSPENSION_CONFLICT = "SUSPENSION_CONFLICT"
NO_SUSPENSION_RECORD = "NO_SUSPENSION_RECORD"
AMBIGUOUS_EXACT = "AMBIGUOUS_EXACT"

SUSPENSION_IDENTITY_POLICY = (
    "segment+instrument_key+exchange_token;"
    "exchange+instrument_type consistency guards"
)


def field_text(row: Mapping[str, object], field: str) -> str:
    """Return a stable textual identity value for string and numeric JSON fields."""

    value = row.get(field)
    if value is None:
        return ""
    return str(value).strip()


def row_evidence(row: Mapping[str, object]) -> dict[str, object]:
    """Keep all audit fields needed to explain a suspension decision."""

    return {
        field: row.get(field)
        for field in (
            "segment",
            "exchange",
            "isin",
            "instrument_key",
            "instrument_type",
            "security_type",
            "trading_symbol",
            "series",
            "lot_size",
            "exchange_token",
        )
    }


def row_sort_key(row: Mapping[str, object]) -> tuple[str, ...]:
    return tuple(str(row.get(field, "")) for field in row_evidence(row))


@dataclass(frozen=True)
class SuspensionIndex:
    """Immutable, deterministically ordered indexes over suspended-file rows."""

    rows_by_segment_key: Mapping[
        tuple[str, str], tuple[Mapping[str, object], ...]
    ]
    rows_by_segment_key_token: Mapping[
        tuple[str, str, str], tuple[Mapping[str, object], ...]
    ]

    def rows_for(
        self, segment: str, instrument_key: str
    ) -> tuple[Mapping[str, object], ...]:
        return self.rows_by_segment_key.get((segment, instrument_key), ())

    def exact_token_rows_for(
        self, segment: str, instrument_key: str, exchange_token: str
    ) -> tuple[Mapping[str, object], ...]:
        return self.rows_by_segment_key_token.get(
            (segment, instrument_key, exchange_token), ()
        )


def build_suspension_index(
    suspended_rows: Iterable[Mapping[str, object]],
) -> SuspensionIndex:
    """Build the canonical suspension index with one deterministic sort pass.

    Rows without both identity scope fields cannot match a current row under
    the approved policy and are intentionally omitted from the lookup maps.
    The original payload remains available to callers for source counts and
    hashing; this index only owns references to relevant row mappings.
    """

    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = {}
    for row in suspended_rows:
        scope = (field_text(row, "segment"), field_text(row, "instrument_key"))
        if not scope[0] or not scope[1]:
            continue
        grouped.setdefault(scope, []).append(row)

    rows_by_segment_key: dict[
        tuple[str, str], tuple[Mapping[str, object], ...]
    ] = {}
    token_groups: dict[
        tuple[str, str, str], list[Mapping[str, object]]
    ] = {}
    for scope in sorted(grouped):
        ordered_rows = tuple(sorted(grouped[scope], key=row_sort_key))
        rows_by_segment_key[scope] = ordered_rows
        segment, instrument_key = scope
        for row in ordered_rows:
            exchange_token = field_text(row, "exchange_token")
            if exchange_token:
                token_groups.setdefault(
                    (segment, instrument_key, exchange_token), []
                ).append(row)

    rows_by_segment_key_token = {
        scope: tuple(rows) for scope, rows in sorted(token_groups.items())
    }
    return SuspensionIndex(
        rows_by_segment_key=MappingProxyType(rows_by_segment_key),
        rows_by_segment_key_token=MappingProxyType(rows_by_segment_key_token),
    )


@dataclass(frozen=True)
class SuspensionResolution:
    """Resolution of one current BOD row against suspended-file evidence."""

    same_segment_key_rows: tuple[Mapping[str, object], ...]
    exact_token_rows: tuple[Mapping[str, object], ...]
    exact_identity_rows: tuple[Mapping[str, object], ...]
    guard_conflict_rows: tuple[Mapping[str, object], ...]
    status: str

    @property
    def same_key_rows(self) -> tuple[Mapping[str, object], ...]:
        """Backward-compatible name for the same segment/key evidence."""

        return self.same_segment_key_rows

    @property
    def matching_rows(self) -> tuple[Mapping[str, object], ...]:
        """Backward-compatible exact current identity rows."""

        return self.exact_identity_rows

    @property
    def ambiguous(self) -> bool:
        return self.status == AMBIGUOUS_EXACT


def resolve_suspension(
    current_row: Mapping[str, object],
    suspension_index: SuspensionIndex,
) -> SuspensionResolution:
    """Resolve suspension evidence without treating a key-only match as exact.

    The segment and instrument key scope the evidence.  The exchange token is
    the current row identity.  Exchange and instrument type are consistency
    guards: a mismatch at the current token is a conflict and never an active
    exact suspension. Lookups never scan the full suspended payload.
    """

    segment = field_text(current_row, "segment")
    instrument_key = field_text(current_row, "instrument_key")
    exchange_token = field_text(current_row, "exchange_token")
    instrument_type = field_text(current_row, "instrument_type")
    exchange = field_text(current_row, "exchange")

    same_segment_key_rows = suspension_index.rows_for(segment, instrument_key)
    exact_token_rows = (
        suspension_index.exact_token_rows_for(segment, instrument_key, exchange_token)
        if exchange_token
        else ()
    )

    def guard_matches(row: Mapping[str, object]) -> bool:
        return (
            field_text(row, "instrument_type") == instrument_type
            and field_text(row, "exchange") == exchange
        )

    exact_identity_rows = tuple(row for row in exact_token_rows if guard_matches(row))
    guard_conflict_rows = tuple(row for row in exact_token_rows if not guard_matches(row))

    if not same_segment_key_rows:
        status = NO_SUSPENSION_RECORD
    elif guard_conflict_rows:
        # A current-token row with inconsistent identity is contradictory
        # evidence; it must not be interpreted as proven active or suspended.
        status = SUSPENSION_CONFLICT
    elif len(exact_identity_rows) > 1:
        status = AMBIGUOUS_EXACT
    elif exact_identity_rows:
        status = SUSPENDED_EXACT
    else:
        status = SUSPENSION_CONFLICT

    return SuspensionResolution(
        same_segment_key_rows=same_segment_key_rows,
        exact_token_rows=exact_token_rows,
        exact_identity_rows=exact_identity_rows,
        guard_conflict_rows=guard_conflict_rows,
        status=status,
    )
