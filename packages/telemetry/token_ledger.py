"""In-memory token ledger. Persistent storage will be added later."""

from dataclasses import asdict
from typing import Optional

from packages.contracts.usage import TokenUsage


class TokenLedger:
    def __init__(self):
        self._records = {}

    def record(self, usage: TokenUsage) -> None:
        """Accept identical replays; reject conflicting duplicate IDs."""
        existing = self._records.get(usage.call_id)

        if existing is not None:
            if existing != usage:
                raise ValueError(
                    f"Conflicting usage for call_id={usage.call_id}"
                )
            return

        self._records[usage.call_id] = usage

    def summary(
        self,
        experiment_id: str,
        approach: Optional[str] = None,
        phase: Optional[str] = None,
        usage_kind: Optional[str] = None,
    ) -> dict:
        selected = [
            record
            for record in self._records.values()
            if record.experiment_id == experiment_id
            and (approach is None or record.approach == approach)
            and (phase is None or record.phase == phase)
            and (usage_kind is None or record.usage_kind == usage_kind)
        ]

        # Keep accounting categories separate.
        groups = {}

        for record in selected:
            key = (
                record.approach,
                record.phase,
                record.stage,
                record.usage_kind,
                record.provider,
                record.model,
            )
            groups.setdefault(key, []).append(record)

        rows = []

        for key, records in sorted(groups.items()):
            complete = all(
                record.total_tokens is not None for record in records
            )
            cache_complete = all(
                record.cached_input_tokens is not None
                for record in records
            )

            rows.append({
                "approach": key[0],
                "phase": key[1],
                "stage": key[2],
                "usage_kind": key[3],
                "provider": key[4],
                "model": key[5],
                "calls": len(records),
                "retry_calls": sum(r.attempt > 1 for r in records),
                "missing_usage_calls": sum(
                    r.total_tokens is None for r in records
                ),
                "reported_input_tokens": sum(
                    r.input_tokens
                    for r in records
                    if r.input_tokens is not None
                ),
                "reported_output_tokens": sum(
                    r.output_tokens
                    for r in records
                    if r.output_tokens is not None
                ),
                "total_tokens": (
                    sum(r.total_tokens for r in records)
                    if complete else None
                ),
                "cached_input_tokens": (
                    sum(r.cached_input_tokens for r in records)
                    if cache_complete else None
                ),
                "usage_complete": complete,
            })

        return {
            "experiment_id": experiment_id,
            "recorded_calls": len(selected),
            "groups": rows,
        }

    def export_records(self) -> list:
        """Return raw records for inspection or later persistence."""
        return [asdict(record) for record in self._records.values()]
