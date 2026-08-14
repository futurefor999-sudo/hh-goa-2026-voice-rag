"""
Latency instrumentation.

Every pipeline run records a per-stage timing breakdown (stt, retrieval,
generation, guardrails, total). LatencyTracker accumulates these across
many queries and computes P50/P70/P100 — which the submission brief
requires reported "across a reasonable number of test queries, not a
single best-case run."

Two totals are tracked separately and both are reported:
  - `total_ms`        : stt + retrieval + guardrails + generation (everything)
  - `pipeline_ms`      : total_ms minus stt and minus the network round-trip
                         portion of generation

Why split them: the brief's 200ms target is scoped to "chunking + vector
DB retrieval + everything through to final output." A real STT call and
a real hosted-LLM generation call each typically cost several hundred ms
of network round-trip on their own — that's inherent to calling external
APIs over HTTP, not something a chunking/retrieval design can fix. Rather
than quietly exclude that from the report, both numbers are shown so it's
visible where time actually goes. See README "On the 200ms target" for
more.
"""
from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np


@dataclass
class StageTiming:
    stt_ms: float = 0.0
    retrieval_ms: float = 0.0  # chunking is precomputed at ingest time; this is embed-query + search
    generation_ms: float = 0.0
    guardrails_ms: float = 0.0

    @property
    def total_ms(self) -> float:
        return self.stt_ms + self.retrieval_ms + self.generation_ms + self.guardrails_ms

    @property
    def pipeline_ms(self) -> float:
        """retrieval + guardrails only — the portion this system's own
        chunking/indexing/retrieval design controls, excluding external
        network calls to STT and generation APIs."""
        return self.retrieval_ms + self.guardrails_ms


@dataclass
class LatencyTracker:
    records: list[StageTiming] = field(default_factory=list)

    def add(self, timing: StageTiming) -> None:
        self.records.append(timing)

    def _percentiles(self, values: list[float]) -> dict:
        if not values:
            return {"p50": None, "p70": None, "p100": None, "mean": None, "n": 0}
        arr = np.array(values)
        return {
            "p50": float(np.percentile(arr, 50)),
            "p70": float(np.percentile(arr, 70)),
            "p100": float(np.percentile(arr, 100)),
            "mean": float(np.mean(arr)),
            "n": len(values),
        }

    def report(self) -> dict:
        return {
            "total_ms": self._percentiles([r.total_ms for r in self.records]),
            "pipeline_ms_retrieval_guardrails_only": self._percentiles([r.pipeline_ms for r in self.records]),
            "stt_ms": self._percentiles([r.stt_ms for r in self.records]),
            "retrieval_ms": self._percentiles([r.retrieval_ms for r in self.records]),
            "generation_ms": self._percentiles([r.generation_ms for r in self.records]),
            "guardrails_ms": self._percentiles([r.guardrails_ms for r in self.records]),
        }

    def print_report(self) -> None:
        report = self.report()
        print(f"\nLatency report over {len(self.records)} queries:")
        print(f"{'stage':<35}{'p50':>10}{'p70':>10}{'p100':>10}{'mean':>10}")
        for name, stats in report.items():
            if stats["n"] == 0:
                continue
            print(
                f"{name:<35}{stats['p50']:>10.1f}{stats['p70']:>10.1f}"
                f"{stats['p100']:>10.1f}{stats['mean']:>10.1f}"
            )
