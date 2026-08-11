"""Shared memory-aware execution for tabular inference estimators."""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass

import torch
from torch import nn

from .config import InferenceManifest
from .model.memory import (
    PeakMemoryEstimate,
    PeakMemoryMode,
    estimated_peak,
    feedforward_token_counts,
)
from .runtime import FLASH_ATTENTION_BATCH_LIMIT, FULL_FEATURE_ATTENTION_ROWS


_logger = logging.getLogger(__name__)

_COMPUTE_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_CPU_FEEDFORWARD_TOKEN_CHUNK = 524_288


@dataclass(frozen=True, slots=True)
class _CudaExecutionPlan:
    ensemble_step: int
    query_step: int
    feedforward_token_chunk: int
    support_only_cache: bool = False
    query_cache: bool = False
    cache_to_cpu: bool = False
    compact_cache: bool = False
    reload_cache: bool = False
    feature_attention_row_chunk: int = FULL_FEATURE_ATTENTION_ROWS


class _InferenceExecutor:
    """Plan and execute one task-neutral batch of ensemble members."""

    def __init__(
        self,
        manifest: InferenceManifest,
        *,
        device: torch.device | str,
        max_vram_bytes: int | None,
    ) -> None:
        if not isinstance(manifest, InferenceManifest):
            raise TypeError("manifest must be an InferenceManifest")
        self.manifest = manifest
        self.device = torch.device(device)
        self.max_vram_bytes = max_vram_bytes

    def empty_cuda_cache(self) -> None:
        if self.device.type != "cuda":
            return
        with torch.cuda.device(self.device):
            torch.cuda.empty_cache()

    def cuda_budget(self) -> tuple[int, int, int, int]:
        """Return target, allocated, reserved, and driver-free bytes."""
        if self.device.type != "cuda":
            raise RuntimeError("CUDA memory budget requested for a non-CUDA device")
        free_bytes, total_bytes = torch.cuda.mem_get_info(self.device)
        allocated = int(torch.cuda.memory_allocated(self.device))
        reserved = int(torch.cuda.memory_reserved(self.device))
        target = (
            min(int(total_bytes), int(free_bytes) + reserved)
            if self.max_vram_bytes is None
            else self.max_vram_bytes
        )
        return int(target), allocated, reserved, int(free_bytes)

    @staticmethod
    def safe_query_chunk(
        fitting_chunk: int,
        query_count: int,
        dtype: torch.dtype,
    ) -> int:
        """Select 85-90% of a memory limit, then align if calls are unchanged."""
        lower = (85 * fitting_chunk + 99) // 100
        upper = (90 * fitting_chunk + 99) // 100 - 1
        if upper < lower:
            return max(1, min(query_count, fitting_chunk))
        selected = min(upper, max(lower, (7 * fitting_chunk + 4) // 8))
        if query_count <= selected:
            return query_count
        element_size = torch.empty((), dtype=dtype).element_size()
        alignment = max(1, 16 // element_size)
        aligned = selected - selected % alignment
        if (
            aligned >= lower
            and (query_count + aligned - 1) // aligned
            == (query_count + selected - 1) // selected
        ):
            selected = aligned
        return max(1, selected)

    def plan_cuda_execution(
        self,
        model: nn.Module,
        support_batch: torch.Tensor,
        query_batch: torch.Tensor,
        compute_dtype: torch.dtype,
        estimate: Callable[[PeakMemoryMode, int, int, int], PeakMemoryEstimate],
        hard_estimate: (
            Callable[[PeakMemoryMode, int, int, int], PeakMemoryEstimate] | None
        ) = None,
        *,
        chunked_estimate: (
            Callable[
                [int], Callable[[PeakMemoryMode, int, int, int], PeakMemoryEstimate]
            ]
            | None
        ) = None,
        chunked_hard_estimate: (
            Callable[
                [int], Callable[[PeakMemoryMode, int, int, int], PeakMemoryEstimate]
            ]
            | None
        ) = None,
    ) -> _CudaExecutionPlan:
        """Choose one immutable CUDA execution plan before model execution.

        ``estimate`` is the *preference* budget: shrinking it biases selection
        toward smaller chunks. ``hard_estimate`` is the physical ceiling (100% of
        detected VRAM) and defaults to ``estimate``. The support peak cannot be
        chunked smaller, so a plan that fits the ceiling is never rejected just
        because a retry shrank the preference budget.

        ``chunked_estimate``/``chunked_hard_estimate`` re-bind those estimators
        to a feature-attention row chunk, letting each candidate derive its own
        chunk from its own build phase. Omit them to plan at full rows.
        """
        if hard_estimate is None:
            hard_estimate = estimate
        ensemble_count, support_count, feature_count = support_batch.shape
        query_count = query_batch.shape[1]
        groups = (
            feature_count + self.manifest.model.columns_per_group - 1
        ) // self.manifest.model.columns_per_group
        Phase = tuple[PeakMemoryMode, bool]

        def safely_fits(
            result: PeakMemoryEstimate,
            query_dependent: bool,
        ) -> bool:
            if not result.fits:
                return False
            if query_dependent:
                return True
            return (
                10 * result.projected_reserved_bytes
                < 9 * result.target_vram_bytes
            )

        def select_chunks(
            phases: tuple[Phase, ...],
            *,
            estimate_fn: Callable[
                [PeakMemoryMode, int, int, int], PeakMemoryEstimate
            ],
        ) -> tuple[int, int, int] | None:
            def phase_estimates(
                query_rows: int,
                members: int,
                feedforward_token_chunk: int,
            ):
                return tuple(
                    (
                        estimate_fn(
                            mode,
                            query_rows if query_dependent else 0,
                            members,
                            feedforward_token_chunk,
                        ),
                        query_dependent,
                    )
                    for mode, query_dependent in phases
                )

            def query_capacity(
                members: int,
                feedforward_token_chunk: int,
            ) -> int:
                minimum = phase_estimates(1, members, feedforward_token_chunk)
                if not all(safely_fits(*phase) for phase in minimum):
                    return 0
                maximum = phase_estimates(
                    FLASH_ATTENTION_BATCH_LIMIT,
                    members,
                    feedforward_token_chunk,
                )
                if all(safely_fits(*phase) for phase in maximum):
                    return FLASH_ATTENTION_BATCH_LIMIT
                low, high = 1, FLASH_ATTENTION_BATCH_LIMIT
                while low < high:
                    middle = (low + high + 1) // 2
                    if all(
                        safely_fits(*phase)
                        for phase in phase_estimates(
                            middle, members, feedforward_token_chunk
                        )
                    ):
                        low = middle
                    else:
                        high = middle - 1
                return low

            def safe_query(
                members: int,
                feedforward_token_chunk: int,
            ) -> int:
                capacity = query_capacity(members, feedforward_token_chunk)
                if capacity == 0:
                    return 0
                return self.safe_query_chunk(capacity, query_count, compute_dtype)

            one_member_query = safe_query(1, 1)
            if one_member_query == 0:
                # Not even the maximally chunked forward fits this budget. The
                # caller retries against the hard ceiling before giving up.
                return None

            low, high = 1, FLASH_ATTENTION_BATCH_LIMIT
            while low < high:
                middle = (low + high) // 2
                if (
                    self.safe_query_chunk(middle, query_count, compute_dtype)
                    >= one_member_query
                ):
                    high = middle
                else:
                    low = middle + 1
            required_capacity = low

            def preserves_query(
                members: int,
                feedforward_token_chunk: int,
            ) -> bool:
                return all(
                    safely_fits(*phase)
                    for phase in phase_estimates(
                        required_capacity,
                        members,
                        feedforward_token_chunk,
                    )
                )

            def maximum_token_chunk(members: int) -> int:
                maximum = 1
                for mode, query_dependent in phases:
                    counts = feedforward_token_counts(
                        self.manifest.model,
                        support_rows=support_count,
                        query_rows=(required_capacity if query_dependent else 0),
                        feature_count=feature_count,
                        members=members,
                        mode=mode,
                    )
                    if counts:
                        maximum = max(maximum, max(counts))
                return maximum

            def weighted_token_counts(
                member_step: int,
            ) -> tuple[tuple[int, int], ...]:
                def counts(
                    mode: PeakMemoryMode,
                    rows: int,
                    members: int,
                    multiplier: int = 1,
                ) -> list[tuple[int, int]]:
                    if rows < 0 or members <= 0 or multiplier <= 0:
                        return []
                    return [
                        (tokens, multiplier)
                        for tokens in feedforward_token_counts(
                            self.manifest.model,
                            support_rows=support_count,
                            query_rows=rows,
                            feature_count=feature_count,
                            members=members,
                            mode=mode,
                        )
                    ]

                full_members, final_members = divmod(
                    ensemble_count, member_step
                )
                member_batches = [(member_step, full_members)]
                if final_members:
                    member_batches.append((final_members, 1))
                result: list[tuple[int, int]] = []
                for members, member_multiplier in member_batches:
                    if len(phases) == 1:
                        mode, _query_dependent = phases[0]
                        full, remainder = divmod(query_count, one_member_query)
                        result.extend(
                            counts(
                                mode,
                                one_member_query,
                                members,
                                member_multiplier * full,
                            )
                        )
                        if remainder:
                            result.extend(
                                counts(
                                    mode,
                                    remainder,
                                    members,
                                    member_multiplier,
                                )
                            )
                        continue

                    first_mode = phases[0][0]
                    second_mode = phases[1][0]
                    if first_mode is PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU:
                        result.extend(
                            counts(first_mode, 0, members, member_multiplier)
                        )
                        result.extend(
                            counts(
                                second_mode,
                                query_count,
                                members,
                                member_multiplier,
                            )
                        )
                        continue

                    first_rows = min(query_count, one_member_query)
                    result.extend(
                        counts(
                            first_mode,
                            first_rows,
                            members,
                            member_multiplier,
                        )
                    )
                    remaining = query_count - first_rows
                    full, remainder = divmod(remaining, one_member_query)
                    result.extend(
                        counts(
                            second_mode,
                            one_member_query,
                            members,
                            member_multiplier * full,
                        )
                    )
                    if remainder:
                        result.extend(
                            counts(
                                second_mode,
                                remainder,
                                members,
                                member_multiplier,
                            )
                        )
                return tuple(result)

            def feedforward_calls(
                token_counts: tuple[tuple[int, int], ...],
                token_chunk: int,
            ) -> int:
                return sum(
                    multiplier * ((tokens + token_chunk - 1) // token_chunk)
                    for tokens, multiplier in token_counts
                )

            candidates: list[tuple[int, int, int]] = []
            for member_step in range(1, ensemble_count + 1):
                if not preserves_query(member_step, 1):
                    continue
                low, high = 1, maximum_token_chunk(member_step)
                while low < high:
                    middle = (low + high + 1) // 2
                    if preserves_query(member_step, middle):
                        low = middle
                    else:
                        high = middle - 1
                maximum_safe_chunk = low
                token_counts = weighted_token_counts(member_step)
                minimum_calls = feedforward_calls(
                    token_counts, maximum_safe_chunk
                )
                low, high = 1, maximum_safe_chunk
                while low < high:
                    middle = (low + high) // 2
                    if feedforward_calls(token_counts, middle) <= minimum_calls:
                        high = middle
                    else:
                        low = middle + 1
                candidates.append((minimum_calls, low, member_step))

            _calls, selected_token_chunk, selected_members = min(
                candidates,
                key=lambda candidate: (
                    candidate[0],
                    candidate[1],
                    -candidate[2],
                ),
            )
            return selected_members, one_member_query, selected_token_chunk

        def plan_chunks(
            phases: tuple[Phase, ...],
            *,
            required: bool,
        ) -> tuple[int, int, int, int] | None:
            """Select against the preference budget; validate against the ceiling.

            The support peak is not chunkable, so when the preference budget is
            too small to admit any plan we retry against the physical ceiling
            rather than refusing — a lowered preference budget must not reject a
            forward real VRAM can run. Only a minimum that overflows the ceiling
            fails.

            The feature-attention row chunk is derived first, from *this*
            candidate's own build phase, and then held fixed while members,
            query rows, and the FFN chunk are searched against it. Candidates
            differ by more than a GB in build peak, so one chunk shared across
            them would leave the heavier ones overflowing.
            """
            build_mode, build_query_dependent = phases[0]
            build_rows = 1 if build_query_dependent else 0

            def derive(factory) -> int:
                """Largest chunk under which this build clears its own bar.

                The bar is the one ``select_chunks`` will apply to that phase,
                so a build already admitted at full rows is never chunked, and a
                build that needs chunking is not held to a stricter test than
                the one deciding its fate.
                """
                return self._plan_feature_attention_chunk(
                    support_count,
                    lambda chunk: factory(chunk)(build_mode, build_rows, 1, 1),
                    accepts=lambda result: safely_fits(
                        result, build_query_dependent
                    ),
                )

            row_chunk = FULL_FEATURE_ATTENTION_ROWS
            budget_estimate, ceiling_estimate = estimate, hard_estimate
            if chunked_estimate is not None:
                row_chunk = derive(chunked_estimate)
                budget_estimate = chunked_estimate(row_chunk)
            selected = select_chunks(phases, estimate_fn=budget_estimate)
            if selected is None:
                # Re-derive against the ceiling so the chunk and the budget that
                # admits the plan are the same one.
                if chunked_hard_estimate is not None:
                    row_chunk = derive(chunked_hard_estimate)
                    ceiling_estimate = chunked_hard_estimate(row_chunk)
                selected = select_chunks(phases, estimate_fn=ceiling_estimate)
            if selected is not None:
                return (*selected, row_chunk)
            if not required:
                return None
            limiting = max(
                (
                    ceiling_estimate(mode, 1 if query_dependent else 0, 1, 1)
                    for mode, query_dependent in phases
                ),
                key=lambda result: result.projected_reserved_bytes,
            )
            raise RuntimeError(
                "detected VRAM cannot fit the minimum model forward; "
                f"mode={limiting.limiting_phase}, "
                f"support_rows={support_count}, query_rows=1, "
                f"feature_count={feature_count}, members=1, "
                f"feature_attention_row_chunk={row_chunk}, "
                "estimated_reserved_bytes="
                f"{limiting.projected_reserved_bytes}, "
                f"target_vram_bytes={limiting.target_vram_bytes}"
            )

        supports_cache = bool(
            getattr(model, "supports_query_chunk_cache", False)
        )
        if (
            supports_cache
            and query_count <= FLASH_ATTENTION_BATCH_LIMIT
            and bool(model.can_cache_support(support_batch))
        ):
            support_only = plan_chunks(
                (
                    (PeakMemoryMode.SUPPORT_ONLY_BUILD_GPU, False),
                    (PeakMemoryMode.CACHED_COMPACT_GPU, True),
                ),
                required=False,
            )
            if support_only is not None and support_only[1] == query_count:
                return _CudaExecutionPlan(
                    ensemble_step=support_only[0],
                    query_step=query_count,
                    feedforward_token_chunk=support_only[2],
                    support_only_cache=True,
                    compact_cache=True,
                    feature_attention_row_chunk=support_only[3],
                )

        joined = plan_chunks(((PeakMemoryMode.JOINED, True),), required=True)
        assert joined is not None
        if joined[1] == query_count or not supports_cache:
            return _CudaExecutionPlan(
                ensemble_step=joined[0],
                query_step=joined[1],
                feedforward_token_chunk=joined[2],
                feature_attention_row_chunk=joined[3],
            )

        gpu_compact = plan_chunks(
            (
                (PeakMemoryMode.CACHE_BUILD_GPU, True),
                (PeakMemoryMode.CACHED_COMPACT_GPU, True),
            ),
            required=False,
        )
        if (
            gpu_compact is not None
            and gpu_compact[1] < query_count
            and (groups + 1) * (support_count + gpu_compact[1]) >= 500_000
            and bool(
                model.can_cache_query_chunks(
                    support_batch, query_batch, gpu_compact[1]
                )
            )
        ):
            return _CudaExecutionPlan(
                ensemble_step=gpu_compact[0],
                query_step=gpu_compact[1],
                feedforward_token_chunk=gpu_compact[2],
                query_cache=True,
                compact_cache=True,
                feature_attention_row_chunk=gpu_compact[3],
            )

        gpu_padded = plan_chunks(
            (
                (PeakMemoryMode.CACHE_BUILD_GPU, True),
                (PeakMemoryMode.CACHED_PADDED_GPU, True),
            ),
            required=False,
        )
        if (
            gpu_padded is not None
            and gpu_padded[1] < query_count
            and bool(
                model.can_cache_query_chunks(
                    support_batch, query_batch, gpu_padded[1]
                )
            )
        ):
            return _CudaExecutionPlan(
                ensemble_step=gpu_padded[0],
                query_step=gpu_padded[1],
                feedforward_token_chunk=gpu_padded[2],
                query_cache=True,
                compact_cache=False,
                feature_attention_row_chunk=gpu_padded[3],
            )

        # GPU-resident caching did not fit even maximally chunked -- its build
        # peak overflows once the K/V-pin fragmentation multiplier is applied.
        # Offload the pins to host during the build (no fragmentation tail),
        # then bulk-reload them before the query phase so cross-attention stays
        # resident and fast. Preferred over the streaming fallback below; gated
        # by the runtime knob and the same size heuristic as the GPU path.
        if self.manifest.runtime.support_cache_offload:
            offload_reload = plan_chunks(
                (
                    (PeakMemoryMode.CACHE_BUILD_CPU, True),
                    (PeakMemoryMode.CACHED_COMPACT_GPU, True),
                ),
                required=False,
            )
            if (
                offload_reload is not None
                and offload_reload[1] < query_count
                and (groups + 1) * (support_count + offload_reload[1]) >= 500_000
                and bool(
                    model.can_cache_query_chunks(
                        support_batch, query_batch, offload_reload[1]
                    )
                )
            ):
                return _CudaExecutionPlan(
                    ensemble_step=offload_reload[0],
                    query_step=offload_reload[1],
                    feedforward_token_chunk=offload_reload[2],
                    query_cache=True,
                    cache_to_cpu=True,
                    compact_cache=True,
                    reload_cache=True,
                    feature_attention_row_chunk=offload_reload[3],
                )

        cpu_cached = plan_chunks(
            (
                (PeakMemoryMode.CACHE_BUILD_CPU, True),
                (PeakMemoryMode.CACHED_COMPACT_CPU, True),
            ),
            required=False,
        )
        if (
            cpu_cached is not None
            and cpu_cached[1] < query_count
            and bool(
                model.can_cache_query_chunks(
                    support_batch, query_batch, cpu_cached[1]
                )
            )
        ):
            return _CudaExecutionPlan(
                ensemble_step=cpu_cached[0],
                query_step=cpu_cached[1],
                feedforward_token_chunk=cpu_cached[2],
                query_cache=True,
                cache_to_cpu=True,
                compact_cache=True,
                feature_attention_row_chunk=cpu_cached[3],
            )
        return _CudaExecutionPlan(
            ensemble_step=joined[0],
            query_step=joined[1],
            feedforward_token_chunk=joined[2],
            feature_attention_row_chunk=joined[3],
        )

    def output_memory_options(
        self,
        *,
        ensemble_count: int,
        query_count: int,
        compute_dtype: torch.dtype,
    ) -> dict[str, int]:
        """Return the common output-head coefficients for peak estimation."""
        output_width = self.manifest.output_width
        element_size = torch.empty((), dtype=compute_dtype).element_size()
        return {
            "output_width": output_width,
            "decoder_hidden_width": self.manifest.decoder_hidden_width,
            # During model work one completed-output equivalent can remain
            # live. The estimator separately models the final two-copy cat.
            "outer_output_bytes": (
                ensemble_count
                * query_count
                * output_width
                * element_size
            ),
        }

    def _plan_with_budget(
        self,
        model: nn.Module,
        support_batch: torch.Tensor,
        query_batch: torch.Tensor,
        compute_dtype: torch.dtype,
        memory_options: dict[str, int],
        *,
        target: int,
        allocated: int,
        reserved: int,
        free: int,
        ceiling_target: int | None = None,
        ceiling_free: int | None = None,
    ) -> _CudaExecutionPlan:
        """Build one immutable CUDA plan against the supplied memory budget.

        ``target``/``free`` are the preference budget used to pick chunk sizes;
        ``ceiling_target``/``ceiling_free`` are the physical limit a plan must
        respect and default to the preference budget.
        """
        support_count, feature_count = support_batch.shape[1:]
        if ceiling_target is None:
            ceiling_target = target
        if ceiling_free is None:
            ceiling_free = free

        def estimator(
            budget: int,
            free_budget: int,
            row_chunk: int = FULL_FEATURE_ATTENTION_ROWS,
        ):
            def estimate(
                mode: PeakMemoryMode,
                query_rows: int,
                members: int,
                feedforward_token_chunk: int,
            ) -> PeakMemoryEstimate:
                return estimated_peak(
                    self.manifest.model,
                    support_rows=support_count,
                    query_rows=query_rows,
                    feature_count=feature_count,
                    members=members,
                    feedforward_token_chunk=feedforward_token_chunk,
                    feature_attention_row_chunk=row_chunk,
                    dtype=compute_dtype,
                    mode=mode,
                    max_vram_bytes=budget,
                    baseline_allocated_bytes=allocated,
                    baseline_reserved_bytes=reserved,
                    free_bytes=free_budget,
                    **memory_options,
                )

            return estimate

        # Rows are a batch axis in feature attention, the limiting phase for a
        # large support build, so each candidate plan proactively derives its own
        # row chunk from its own build phase and then selects members, query, and
        # FFN chunks against that reduced peak.
        plan = self.plan_cuda_execution(
            model,
            support_batch,
            query_batch,
            compute_dtype,
            estimator(target, free),
            estimator(ceiling_target, ceiling_free),
            chunked_estimate=lambda chunk: estimator(target, free, chunk),
            chunked_hard_estimate=(
                lambda chunk: estimator(ceiling_target, ceiling_free, chunk)
            ),
        )
        row_chunk = plan.feature_attention_row_chunk
        if row_chunk < support_count:
            _logger.info(
                "feature-attention row-chunk engaged: chunk=%d over %d support "
                "rows (~%d chunks) to fit the preference budget",
                row_chunk,
                support_count,
                (support_count + row_chunk - 1) // row_chunk,
            )
        return plan

    @staticmethod
    def _plan_feature_attention_chunk(
        support_count: int,
        build_peak: Callable[[int], PeakMemoryEstimate],
        *,
        accepts: Callable[[PeakMemoryEstimate], bool],
    ) -> int:
        """Largest feature-attention row chunk this build is accepted at.

        Rows are a batch axis, so the feature-attention transient -- the phase
        that limits a large support build -- scales with the chunk while its
        resident floor does not. ``build_peak`` must estimate the build phase of
        the candidate this chunk is for, and ``accepts`` the test that candidate
        will be judged by: builds differ by more than a gigabyte, and holding one
        to another's peak or bar picks a chunk that is too large to fit or too
        small to be worth its kernel launches. Selection is proactive -- the
        support-only bar carries the ~90% headroom margin, so a chunk engages to
        create headroom before anything overflows, not after. Chunking is the
        only recovery there is: the forward is planned once and an OOM is raised
        to the caller. ``FULL_FEATURE_ATTENTION_ROWS`` means "no chunk"; a chunk
        of 1 that is still rejected is returned so the caller surfaces the honest
        ceiling failure through the ordinary plan path rather than here.
        """
        rows = support_count + 1  # support-dominated joined build

        if accepts(build_peak(rows)):
            return FULL_FEATURE_ATTENTION_ROWS
        low, high = 1, rows
        while low < high:
            middle = (low + high + 1) // 2
            if accepts(build_peak(middle)):
                low = middle
            else:
                high = middle - 1
        return low

    def forward(
        self,
        model: nn.Module,
        support_batch: torch.Tensor,
        label_batch: torch.Tensor,
        query_batch: torch.Tensor,
    ) -> torch.Tensor:
        ensemble_count = support_batch.shape[0]
        query_count = query_batch.shape[1]
        compute_dtype = _COMPUTE_DTYPES[self.manifest.runtime.compute_dtype]
        memory_options = self.output_memory_options(
            ensemble_count=ensemble_count,
            query_count=query_count,
            compute_dtype=compute_dtype,
        )

        if self.device.type != "cuda":
            execution_plan = _CudaExecutionPlan(
                ensemble_step=ensemble_count,
                query_step=min(
                    query_count, self.manifest.runtime.query_batch_limit
                ),
                feedforward_token_chunk=_CPU_FEEDFORWARD_TOKEN_CHUNK,
            )
            return self._run_execution_plan(
                model, support_batch, label_batch, query_batch, execution_plan
            )

        # Reclaim reserved segments before planning so a user-set budget is read
        # against a defragmented allocator.
        if self.max_vram_bytes is not None:
            self.empty_cuda_cache()
        target, allocated, reserved, free = self.cuda_budget()
        if allocated > target:
            raise RuntimeError(
                "max_vram_bytes is smaller than the live CUDA baseline "
                f"({target} < {allocated})"
            )
        execution_plan = self._plan_with_budget(
            model,
            support_batch,
            query_batch,
            compute_dtype,
            memory_options,
            target=target,
            allocated=allocated,
            reserved=reserved,
            free=free,
        )
        # Plans are open-loop -- the estimate ignores allocator fragmentation, so
        # a forward can still OOM. The error propagates untouched: choosing how to
        # recover (a tighter budget, fewer members, a smaller support set) belongs
        # to the caller, not to a library that would burn their VRAM retrying.
        return self._run_execution_plan(
            model, support_batch, label_batch, query_batch, execution_plan
        )

    def _run_execution_plan(
        self,
        model: nn.Module,
        support_batch: torch.Tensor,
        label_batch: torch.Tensor,
        query_batch: torch.Tensor,
        execution_plan: _CudaExecutionPlan,
    ) -> torch.Tensor:
        """Execute one immutable plan across ensemble-member and query chunks."""
        ensemble_count = support_batch.shape[0]
        query_count = query_batch.shape[1]
        ensemble_outputs: list[torch.Tensor] = []
        for ensemble_start in range(
            0, ensemble_count, execution_plan.ensemble_step
        ):
            ensemble_stop = min(
                ensemble_start + execution_plan.ensemble_step,
                ensemble_count,
            )
            query_outputs: list[torch.Tensor] = []
            member_support = support_batch[ensemble_start:ensemble_stop]
            member_labels = label_batch[ensemble_start:ensemble_stop]
            member_queries = query_batch[ensemble_start:ensemble_stop]
            try:
                if execution_plan.support_only_cache:
                    model(
                        member_support,
                        member_labels,
                        member_queries[:, :0],
                        feedforward_token_chunk=(
                            execution_plan.feedforward_token_chunk
                        ),
                        feature_attention_row_chunk=(
                            execution_plan.feature_attention_row_chunk
                        ),
                        query_chunk_size=query_count,
                        cache_support=True,
                        trusted_internal_inputs=True,
                    )
                    if self.device.type == "cuda":
                        self.empty_cuda_cache()
                    query_outputs.append(
                        model(
                            member_support,
                            member_labels,
                            member_queries,
                            feedforward_token_chunk=(
                                execution_plan.feedforward_token_chunk
                            ),
                            feature_attention_row_chunk=(
                                execution_plan.feature_attention_row_chunk
                            ),
                            query_chunk_size=query_count,
                            use_cached_support=True,
                            compact_cached_support=True,
                            trusted_internal_inputs=True,
                        )
                    )
                elif execution_plan.query_cache:
                    query_outputs.append(
                        model(
                            member_support,
                            member_labels,
                            member_queries[:, : execution_plan.query_step],
                            feedforward_token_chunk=(
                                execution_plan.feedforward_token_chunk
                            ),
                            feature_attention_row_chunk=(
                                execution_plan.feature_attention_row_chunk
                            ),
                            query_chunk_size=execution_plan.query_step,
                            cache_support=True,
                            cache_support_to_cpu=execution_plan.cache_to_cpu,
                            trusted_internal_inputs=True,
                        )
                    )
                    if self.device.type == "cuda":
                        self.empty_cuda_cache()
                    if execution_plan.reload_cache and self.device.type == "cuda":
                        # Build offloaded the pins to host; restore them to the
                        # accelerator once, now that the build's transients are
                        # freed, so the query chunks read a resident cache.
                        model.load_support_cache(self.device)
                    for query_start in range(
                        execution_plan.query_step,
                        query_count,
                        execution_plan.query_step,
                    ):
                        query_outputs.append(
                            model(
                                member_support,
                                member_labels,
                                member_queries[
                                    :,
                                    query_start : min(
                                        query_start + execution_plan.query_step,
                                        query_count,
                                    ),
                                ],
                                feedforward_token_chunk=(
                                    execution_plan.feedforward_token_chunk
                                ),
                                feature_attention_row_chunk=(
                                    execution_plan.feature_attention_row_chunk
                                ),
                                query_chunk_size=execution_plan.query_step,
                                use_cached_support=True,
                                compact_cached_support=(
                                    execution_plan.compact_cache
                                ),
                                trusted_internal_inputs=True,
                            )
                        )
                else:
                    for query_start in range(
                        0, query_count, execution_plan.query_step
                    ):
                        query_stop = min(
                            query_start + execution_plan.query_step,
                            query_count,
                        )
                        query_outputs.append(
                            model(
                                member_support,
                                member_labels,
                                member_queries[:, query_start:query_stop],
                                feedforward_token_chunk=(
                                    execution_plan.feedforward_token_chunk
                                ),
                                feature_attention_row_chunk=(
                                    execution_plan.feature_attention_row_chunk
                                ),
                                query_chunk_size=execution_plan.query_step,
                                trusted_internal_inputs=True,
                            )
                        )
            finally:
                if (
                    execution_plan.support_only_cache
                    or execution_plan.query_cache
                ):
                    model.clear_support_cache()
                if (
                    (
                        execution_plan.support_only_cache
                        or execution_plan.query_cache
                    )
                    and self.device.type == "cuda"
                ):
                    self.empty_cuda_cache()
            ensemble_outputs.append(torch.cat(query_outputs, dim=1))
        return torch.cat(ensemble_outputs, dim=0)


__all__ = ["_CudaExecutionPlan", "_InferenceExecutor"]
