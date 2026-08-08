"""Common old FJSPLIB/Brandimarte parsing for the FJSP sidecar experiment.

The common text format is integer based.  Its first non-empty line contains
``number_of_jobs number_of_machines`` and may contain a third descriptive
field.  Every subsequent job record starts with its number of operations; each
operation is encoded as ``k machine_1 time_1 ... machine_k time_k``.

Machine identifiers in public collections are usually one based.  A file that
contains machine identifier zero is treated as zero based; otherwise one-based
indexing is used and normalized to zero-based identifiers internally.
"""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Iterable


class FJSPParseError(ValueError):
    """Raised when an instance violates the FJSPLIB/Brandimarte grammar."""


@dataclass(frozen=True, order=True)
class MachineOption:
    machine_id: int
    processing_time: int

    def snapshot(self) -> dict[str, int]:
        return {
            "machine_id": int(self.machine_id),
            "processing_time": int(self.processing_time),
        }


@dataclass(frozen=True)
class Operation:
    job_id: int
    operation_id: int
    alternatives: tuple[MachineOption, ...]

    @property
    def minimum_processing_time(self) -> int:
        return min(option.processing_time for option in self.alternatives)

    @property
    def preferred_machine_id(self) -> int:
        """Fastest eligible machine, with machine id as the deterministic tie break."""

        return min(
            self.alternatives,
            key=lambda option: (option.processing_time, option.machine_id),
        ).machine_id

    def option_for(self, machine_id: int) -> MachineOption | None:
        for option in self.alternatives:
            if option.machine_id == int(machine_id):
                return option
        return None

    def snapshot(self) -> dict[str, object]:
        return {
            "job_id": int(self.job_id),
            "operation_id": int(self.operation_id),
            "alternatives": [option.snapshot() for option in self.alternatives],
        }


@dataclass(frozen=True)
class Job:
    job_id: int
    operations: tuple[Operation, ...]

    @property
    def minimum_work(self) -> int:
        return sum(operation.minimum_processing_time for operation in self.operations)

    def snapshot(self) -> dict[str, object]:
        return {
            "job_id": int(self.job_id),
            "operations": [operation.snapshot() for operation in self.operations],
        }


@dataclass(frozen=True)
class FJSPInstance:
    name: str
    machine_count: int
    jobs: tuple[Job, ...]
    source_sha256: str
    source_machine_index_base: int
    header_metadata: tuple[str, ...] = ()

    @property
    def job_count(self) -> int:
        return len(self.jobs)

    @property
    def operation_count(self) -> int:
        return sum(len(job.operations) for job in self.jobs)

    @property
    def lower_bound_makespan(self) -> int:
        """A deterministic workload/precedence lower bound, not an optimum claim."""

        job_bound = max((job.minimum_work for job in self.jobs), default=0)
        total_minimum_work = sum(job.minimum_work for job in self.jobs)
        machine_bound = (
            (total_minimum_work + self.machine_count - 1) // self.machine_count
            if self.machine_count > 0
            else 0
        )
        return max(job_bound, machine_bound)

    def operation(self, job_id: int, operation_id: int) -> Operation:
        try:
            return self.jobs[int(job_id)].operations[int(operation_id)]
        except (IndexError, TypeError) as exc:
            raise KeyError((job_id, operation_id)) from exc

    def snapshot(self) -> dict[str, object]:
        return {
            "name": self.name,
            "source_sha256": self.source_sha256,
            "source_format": "fjsplib_brandimarte_old_job_line_integer_text",
            "source_machine_index_base": int(self.source_machine_index_base),
            "header_metadata": list(self.header_metadata),
            "job_count": int(self.job_count),
            "machine_count": int(self.machine_count),
            "operation_count": int(self.operation_count),
            "lower_bound_makespan": int(self.lower_bound_makespan),
            "jobs": [job.snapshot() for job in self.jobs],
        }


def load_fjsp_instance(path: str | Path) -> FJSPInstance:
    source = Path(path)
    return parse_fjsp_text(source.read_text(encoding="utf-8"), name=source.stem)


def parse_fjsp_text(text: str, *, name: str = "instance") -> FJSPInstance:
    clean_lines = tuple(_meaningful_lines(text))
    if not clean_lines:
        raise FJSPParseError("instance is empty")

    header = clean_lines[0].split()
    if len(header) < 2:
        raise FJSPParseError("header must contain job and machine counts")
    try:
        job_count = int(header[0])
        machine_count = int(header[1])
    except ValueError as exc:
        raise FJSPParseError("job and machine counts must be integers") from exc
    if job_count <= 0 or machine_count <= 0:
        raise FJSPParseError("job and machine counts must be positive")

    body_tokens = " ".join(clean_lines[1:]).split()
    try:
        values = [int(token) for token in body_tokens]
    except ValueError as exc:
        raise FJSPParseError("job records must contain integers only") from exc

    cursor = 0
    raw_jobs: list[list[list[tuple[int, int]]]] = []
    raw_machine_ids: list[int] = []
    for job_id in range(job_count):
        operation_count, cursor = _take(values, cursor, f"job {job_id} operation count")
        if operation_count <= 0:
            raise FJSPParseError(f"job {job_id} must have at least one operation")
        raw_operations: list[list[tuple[int, int]]] = []
        for operation_id in range(operation_count):
            alternative_count, cursor = _take(
                values,
                cursor,
                f"job {job_id} operation {operation_id} alternative count",
            )
            if alternative_count <= 0:
                raise FJSPParseError(
                    f"job {job_id} operation {operation_id} has no eligible machine"
                )
            alternatives: list[tuple[int, int]] = []
            for alternative_id in range(alternative_count):
                machine_id, cursor = _take(
                    values,
                    cursor,
                    f"job {job_id} operation {operation_id} alternative {alternative_id} machine",
                )
                processing_time, cursor = _take(
                    values,
                    cursor,
                    f"job {job_id} operation {operation_id} alternative {alternative_id} time",
                )
                if processing_time <= 0:
                    raise FJSPParseError(
                        f"job {job_id} operation {operation_id} has non-positive processing time"
                    )
                alternatives.append((machine_id, processing_time))
                raw_machine_ids.append(machine_id)
            raw_operations.append(alternatives)
        raw_jobs.append(raw_operations)

    if cursor != len(values):
        raise FJSPParseError(
            f"unexpected trailing integer data: consumed {cursor} of {len(values)} tokens"
        )

    index_base = 0 if any(machine_id == 0 for machine_id in raw_machine_ids) else 1
    jobs: list[Job] = []
    for job_id, raw_operations in enumerate(raw_jobs):
        operations: list[Operation] = []
        for operation_id, raw_alternatives in enumerate(raw_operations):
            normalized: list[MachineOption] = []
            seen: set[int] = set()
            for raw_machine_id, processing_time in raw_alternatives:
                machine_id = raw_machine_id - index_base
                if not 0 <= machine_id < machine_count:
                    expected = (
                        f"0..{machine_count - 1}" if index_base == 0 else f"1..{machine_count}"
                    )
                    raise FJSPParseError(
                        f"machine id {raw_machine_id} is outside detected {expected} range"
                    )
                if machine_id in seen:
                    raise FJSPParseError(
                        f"job {job_id} operation {operation_id} repeats machine {raw_machine_id}"
                    )
                seen.add(machine_id)
                normalized.append(MachineOption(machine_id, processing_time))
            operations.append(
                Operation(
                    job_id=job_id,
                    operation_id=operation_id,
                    alternatives=tuple(sorted(normalized)),
                )
            )
        jobs.append(Job(job_id=job_id, operations=tuple(operations)))

    canonical_source = "\n".join(clean_lines) + "\n"
    return FJSPInstance(
        name=str(name),
        machine_count=machine_count,
        jobs=tuple(jobs),
        source_sha256=sha256(canonical_source.encode("utf-8")).hexdigest(),
        source_machine_index_base=index_base,
        header_metadata=tuple(header[2:]),
    )


def _meaningful_lines(text: str) -> Iterable[str]:
    for raw_line in text.splitlines():
        line = raw_line
        for marker in ("//", "#", "%"):
            line = line.split(marker, 1)[0]
        line = line.strip()
        if line:
            yield line


def _take(values: list[int], cursor: int, label: str) -> tuple[int, int]:
    if cursor >= len(values):
        raise FJSPParseError(f"truncated instance while reading {label}")
    return values[cursor], cursor + 1
