"""Explicit identity resolution for Upstox suspended-instrument evidence.

The suspended file contains multiple historical/variant rows for one ISIN and
instrument key.  A key-only anti-join therefore cannot establish that the
currently quoted security is suspended.  This module deliberately keeps the
same-segment/key evidence separate from an exact current-token match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass


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
    suspended_rows: Sequence[Mapping[str, object]],
) -> SuspensionResolution:
    """Resolve suspension evidence without treating a key-only match as exact.

    The segment and instrument key scope the evidence.  The exchange token is
    the current row identity.  Exchange and instrument type are consistency
    guards: a mismatch at the current token is a conflict and never an active
    exact suspension.
    """

    segment = field_text(current_row, "segment")
    instrument_key = field_text(current_row, "instrument_key")
    exchange_token = field_text(current_row, "exchange_token")
    instrument_type = field_text(current_row, "instrument_type")
    exchange = field_text(current_row, "exchange")

    same_segment_key_rows = tuple(
        sorted(
            (
                suspended_row
                for suspended_row in suspended_rows
                if segment
                and instrument_key
                and field_text(suspended_row, "segment") == segment
                and field_text(suspended_row, "instrument_key") == instrument_key
            ),
            key=row_sort_key,
        )
    )
    exact_token_rows = tuple(
        suspended_row
        for suspended_row in same_segment_key_rows
        if exchange_token
        and field_text(suspended_row, "exchange_token") == exchange_token
    )
    exact_identity_rows = tuple(
        suspended_row
        for suspended_row in exact_token_rows
        if field_text(suspended_row, "instrument_type") == instrument_type
        and field_text(suspended_row, "exchange") == exchange
    )
    guard_conflict_rows = tuple(
        suspended_row
        for suspended_row in exact_token_rows
        if suspended_row not in exact_identity_rows
    )

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
