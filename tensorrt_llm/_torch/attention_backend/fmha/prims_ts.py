# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""TRT-LLM FMHA adapter for the vendored PrimTS Blackwell kernels."""

from __future__ import annotations

import math
import weakref
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING, Optional, cast

import torch
from packaging.version import InvalidVersion, Version

from tensorrt_llm._torch.attention_backend.interface import AttentionForwardArgs, AttentionInputType
from tensorrt_llm._torch.flashinfer_utils import get_env_enable_pdl
from tensorrt_llm._utils import binding_to_torch_dtype, get_sm_version
from tensorrt_llm.bindings.internal import thop
from tensorrt_llm.functional import AttentionMaskType
from tensorrt_llm.logger import logger
from tensorrt_llm.math_utils import ceil_div, pad_up
from tensorrt_llm.quantization.mode import QuantMode

from .interface import FmhaPhase
from .phased import FmhaParams, PhasedFmha

if TYPE_CHECKING:
    from tensorrt_llm._torch.attention_backend.prims_ts.context import BatchPrefillPagedTSWrapper
    from tensorrt_llm._torch.attention_backend.prims_ts.decode import BatchDecodePagedTSWrapper
    from tensorrt_llm._torch.attention_backend.prims_ts.mla_decode import (
        BatchMLADecodePagedTSWrapper,
    )
    from tensorrt_llm._torch.attention_backend.trtllm import (
        TrtllmAttention,
        TrtllmAttentionMetadata,
    )


_MIN_CUTLASS_DSL_VERSION = Version("4.7.0")
_MIN_CUTLASS_COMPILER_VERSION = "13.3"
_WORKSPACE_ALIGNMENT = 32
_KV_CACHE_QUANT_MODE_MASK = int(
    QuantMode.INT8_KV_CACHE | QuantMode.FP8_KV_CACHE | QuantMode.NVFP4_KV_CACHE
)
_SUPPORTED_MASK_TYPE_VALUES = (
    int(AttentionMaskType.causal),
    int(AttentionMaskType.padding),
)


class _WeakIdentity:
    """Weak identity value that cannot match a recycled Python object ID."""

    __slots__ = ("_object_id", "_ref")

    def __init__(self, value: object) -> None:
        self._object_id = id(value)
        self._ref = weakref.ref(value)

    def get(self) -> Optional[object]:
        return self._ref()

    def __hash__(self) -> int:
        return self._object_id

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, _WeakIdentity):
            return NotImplemented
        value = self._ref()
        return value is not None and value is other._ref()


@dataclass(slots=True, eq=False)
class _V2KvPageOffsetBinding:
    manager_identity: _WeakIdentity
    manager_impl_id: int
    pool_mapping: torch.Tensor
    pool_mapping_shape: tuple[int, ...]
    pool_mapping_version: Optional[int]
    kv_offsets: torch.Tensor
    kv_offsets_shape: tuple[int, ...]
    kv_offsets_version: Optional[int]
    local_layer_idx: int
    pool_index: int
    kv_page_offset: int


@dataclass(slots=True, eq=False)
class _DenseContextPageAlias:
    source: torch.Tensor
    source_key: tuple[object, ...]
    dense_page_idx_kv: torch.Tensor


class PrimsTSFmha(PhasedFmha):
    """Blackwell task-scheduled paged context and decode FMHA library."""

    SUPPORTED_PAGE_SIZES = {16, 32, 64, 128}
    SUPPORTED_CONTEXT_HEAD_DIMS = {128, 256}
    SUPPORTED_DECODE_HEAD_DIMS = {64, 128, 256}
    SUPPORTED_DTYPES = {torch.float16, torch.bfloat16}
    MAX_DECODE_GQA_RATIO = 32

    def __init__(self, attn: "TrtllmAttention") -> None:
        super().__init__(attn)
        # Read once so cached decode plans are not sensitive to later environment changes.
        self._enable_pdl = get_env_enable_pdl()
        self._page_indices_buffer: Optional[torch.Tensor] = None
        self._fixed_indptr_buffer: Optional[torch.Tensor] = None
        self._interleaved_indptr_buffer: Optional[torch.Tensor] = None
        self._sequence_lengths_buffer: Optional[torch.Tensor] = None
        self._context_page_indices_buffer: Optional[torch.Tensor] = None
        self._context_page_gather_indices_buffer: Optional[torch.Tensor] = None
        self._context_page_columns_buffer: Optional[torch.Tensor] = None
        self._context_last_page_indices_buffer: Optional[torch.Tensor] = None
        self._metadata_row_capacity = 0
        self._metadata_column_capacity = 0
        self._context_metadata_row_capacity = 0
        self._context_page_column_capacity = 0
        self._dense_context_page_alias: Optional[_DenseContextPageAlias] = None
        self._kv_page_offset_cache: dict[tuple[int, int], int] = {}
        self._v2_kv_page_offset_binding: Optional[_V2KvPageOffsetBinding] = None
        self._multi_processor_count: Optional[int] = None
        self._retained_metadata_buffers: list[tuple[torch.Tensor, ...]] = []
        # Every other plan attribute is fixed by this layer/model instance.
        # Batch size is the only execution profile that needs its own wrapper.
        self._context_wrappers: dict[int, "BatchPrefillPagedTSWrapper"] = {}
        self._decode_wrappers: dict[int, "BatchDecodePagedTSWrapper"] = {}
        self._mla_decode_wrappers: dict[int, "BatchMLADecodePagedTSWrapper"] = {}
        self._decode_workspace_sizes: dict[int, int] = {}
        self._mla_workspace_sizes: dict[int, int] = {}
        self._workspace_allocation: Optional[tuple[object, ...]] = None
        self._decode_workspace_offset_bytes: Optional[int] = None
        self._decode_workspace_required_bytes = 0

    @staticmethod
    def _get_static_max_kv_len(
        meta: "TrtllmAttentionMetadata",
        *,
        page_capacity: int,
        page_size: int,
    ) -> int:
        """Return the configured semantic bound, not allocator padding."""

        capacity = page_capacity * page_size
        configured_max = meta.max_seq_len
        max_kv_len = capacity if configured_max is None else int(configured_max)
        if max_kv_len <= 0 or max_kv_len > capacity:
            raise RuntimeError(
                f"Invalid PrimTS maximum KV length {max_kv_len} for a "
                f"{capacity}-token page-table capacity."
            )
        return max_kv_len

    @classmethod
    def is_available(cls, attn: "TrtllmAttention") -> bool:
        sm = get_sm_version()
        if sm not in (100, 103):
            logger.debug(f"PrimTS FMHA is unavailable: requires SM100 or SM103, got SM{sm}.")
            return False
        try:
            installed_version = Version(version("nvidia-cutlass-dsl"))
        except (PackageNotFoundError, InvalidVersion):
            logger.debug("PrimTS FMHA is unavailable: nvidia-cutlass-dsl>=4.7 is required.")
            return False
        if installed_version < _MIN_CUTLASS_DSL_VERSION:
            logger.debug(
                "PrimTS FMHA is unavailable: "
                f"nvidia-cutlass-dsl>={_MIN_CUTLASS_DSL_VERSION} is required, "
                f"got {installed_version}."
            )
            return False
        try:
            cutlass = import_module("cutlass")
            compiler_version_supported = cutlass.target_version(
                min_version=_MIN_CUTLASS_COMPILER_VERSION
            )
        except Exception as error:  # noqa: BLE001 - availability probes must fail closed
            logger.debug(
                "PrimTS FMHA is unavailable: could not query the active CUTLASS compiler "
                f"version: {error}"
            )
            return False
        if not compiler_version_supported:
            logger.debug(
                "PrimTS FMHA is unavailable: the active CUTLASS compiler must target "
                f"CUDA>={_MIN_CUTLASS_COMPILER_VERSION}."
            )
            return False
        try:
            import_module("cutlass.experimental.task_scheduling")
        except ImportError:
            logger.debug("PrimTS FMHA is unavailable: CUTLASS task scheduling is missing.")
            return False
        missing_ops = cls._missing_fused_nanobind_ops()
        if missing_ops:
            logger.debug(f"PrimTS FMHA is unavailable: missing nanobind ops {missing_ops}.")
            return False
        return True

    @staticmethod
    def _missing_fused_nanobind_ops() -> list[str]:
        required_ops = (
            "get_trtllm_gen_context_workspace_layout",
            "get_trtllm_gen_generation_workspace_layout",
            "trtllm_gen_context_preprocess",
            "trtllm_gen_context_postprocess",
            "trtllm_gen_generation_preprocess",
            "build_trtllm_gen_kv_cache_metadata",
        )
        return [name for name in required_ops if not hasattr(thop, name)]

    def is_supported(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> bool:
        support_key = self._get_b1_context_support_key(q, k, v, self.attn, metadata, forward_args)
        cached_support_key = metadata._prims_ts_b1_context_support_key
        if support_key is not None and support_key == cached_support_key:
            # Capture can begin between layers in piecewise graph warmup. A
            # cached key must therefore recheck the live stream.
            if not torch.cuda.is_current_stream_capturing():
                return True

        supported, reason = self._is_supported_with_reason(
            q,
            k,
            v,
            self.attn,
            metadata,
            forward_args,
            phase=phase,
        )
        if supported and support_key is not None and cached_support_key is None:
            metadata._prims_ts_b1_context_support_key = support_key
        if not supported:
            logger.debug(f"PrimTS FMHA does not support request: {reason}")
        return supported

    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
    ) -> None:
        if metadata.num_contexts != 1 or metadata.num_generations != 0:
            return super().forward(q, k, v, metadata, forward_args)

        output = forward_args.output
        if output is None:
            raise RuntimeError(f"{type(self).__name__} requires output.")
        if self.REQUIRES_PAGED_KV and metadata.kv_cache_block_offsets is None:
            raise RuntimeError(f"{type(self).__name__} requires paged KV cache.")

        input_type = forward_args.attention_input_type
        num_ctx_tokens = metadata.num_ctx_tokens
        if (
            input_type not in (AttentionInputType.context_only, AttentionInputType.mixed)
            or num_ctx_tokens <= 0
            or q.size(0) != num_ctx_tokens
        ):
            return super().forward(q, k, v, metadata, forward_args)

        attn = self.attn
        if attn.is_mla_enable:
            return super().forward(q, k, v, metadata, forward_args)
        runtime_lengths = (
            metadata.kv_lens_cuda_runtime,
            metadata.kv_lens_runtime,
            metadata.prompt_lens_cuda_runtime,
            metadata.prompt_lens_cpu_runtime,
        )
        if any(
            not isinstance(lengths, torch.Tensor) or lengths.ndim != 1 or lengths.numel() != 1
            for lengths in runtime_lengths
        ):
            return super().forward(q, k, v, metadata, forward_args)

        workspace = cast(torch.Tensor, metadata.effective_workspace)
        self.prepare_workspace(
            q,
            k,
            v,
            metadata,
            forward_args,
            workspace,
        )

        out_tensor = output.view(
            num_ctx_tokens,
            attn.num_heads,
            self.context_out_head_size,
        )
        attention_window_size = forward_args.attention_window_size
        cache_indirection = metadata.cache_indirection
        max_attention_window_size = (
            attention_window_size
            if metadata.beam_width == 1
            else (
                cache_indirection.size(2)
                if cache_indirection is not None
                else attention_window_size
            )
        )
        tokens_per_block = (
            metadata.tokens_per_block if metadata.tokens_per_block is not None else 64
        )
        params = FmhaParams(
            attn=attn,
            meta=metadata,
            fwd=forward_args,
            workspace=workspace,
            max_attention_window_size=max_attention_window_size,
            cyclic_attention_window_size=attention_window_size,
            tokens_per_block=tokens_per_block,
            kv_factor=self.kv_factor,
            total_num_blocks=self._get_total_num_blocks(metadata),
            is_cross=metadata.is_cross,
        )
        kv_lens_cuda = metadata.kv_lens_cuda_runtime
        kv_lens_cpu = metadata.kv_lens_runtime
        prompt_lens_cuda = metadata.prompt_lens_cuda_runtime
        prompt_lens_cpu = metadata.prompt_lens_cpu_runtime
        input_seq_length = int(prompt_lens_cpu[0])
        max_past_kv_length = int(kv_lens_cpu[0])
        params.attention_input = q
        params.qkv_input = q
        params.context_buf = out_tensor
        params.sequence_lengths = kv_lens_cuda
        params.context_lengths = prompt_lens_cuda
        params.max_past_kv_length = max_past_kv_length
        params.num_tokens = num_ctx_tokens
        params.seq_offset = 0
        params.input_seq_length = input_seq_length
        params.batch_size = 1
        params.num_requests = 1
        self.run_context(params)

    def _get_b1_context_support_key(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
    ) -> Optional[tuple[object, ...]]:
        """Build a cheap positive-certificate key for dense B1 context."""
        input_type = fwd.attention_input_type
        if (
            meta.is_cuda_graph
            or meta.is_cross
            or meta.num_contexts != 1
            or meta.num_generations != 0
            or input_type not in (AttentionInputType.context_only, AttentionInputType.mixed)
            or k is not None
            or v is not None
            or not fwd.is_fused_qkv
            or attn.is_mla_enable
        ):
            return None

        manager = meta.kv_cache_manager
        if manager is None:
            return None
        manager_impl = manager.impl
        if (
            not callable(getattr(manager_impl, "get_page_index_upper_bound", None))
            or manager.enable_swa_scratch_reuse
            or manager.num_pools != 1
            or meta.kv_cache_block_offsets is None
            or meta.host_kv_cache_pool_pointers is None
            or meta.host_kv_cache_pool_mapping is None
            or meta.kv_layout != "HND"
        ):
            return None

        if (
            meta.beam_width != 1
            or meta.is_spec_decoding_enabled
            or meta.use_spec_decoding
            or meta.is_spec_dec_tree
            or meta.is_spec_dec_dynamic_tree
        ):
            return None
        runtime_features = meta.runtime_features
        if runtime_features is not None and (
            runtime_features.chunked_prefill
            or runtime_features.cache_reuse
            or runtime_features.has_speculative_draft_tokens
        ):
            return None

        output = fwd.output
        if (
            q.device.type != "cuda"
            or not q.is_contiguous()
            or q.ndim != 2
            or output is None
            or output.device != q.device
            or not output.is_contiguous()
            or fwd.output_sf is not None
            or fwd.out_scale is not None
        ):
            return None

        if (
            attn.sparse_params is not None
            or meta.num_sparse_topk > 0
            or meta.helix_position_offsets is not None
            or fwd.relative_attention_bias is not None
            or fwd.attention_sinks is not None
            or fwd.attention_mask_data is not None
            or fwd.enable_dsv4_epilogue_fusion
            or fwd.sage_attn_num_elts_per_blk_q > 0
            or fwd.sage_attn_num_elts_per_blk_k > 0
            or fwd.sage_attn_num_elts_per_blk_v > 0
            or fwd.sparse_runtime_params.sparse_kv_indices is not None
            or fwd.sparse_runtime_params.sparse_attn_indices is not None
        ):
            return None

        try:
            mask_type = int(fwd.mask_type)
            position_embedding_type = int(attn.position_embedding_type)
            quant_mode = int(attn.quant_mode)
        except (AttributeError, TypeError, ValueError):
            return None
        if (
            mask_type not in _SUPPORTED_MASK_TYPE_VALUES
            or position_embedding_type in (4, 5, 6, 7, 10)
            or quant_mode & _KV_CACHE_QUANT_MODE_MASK
            or (attn.attention_chunk_size or 0) != 0
        ):
            return None

        num_heads = attn.num_heads
        num_kv_heads = attn.num_kv_heads
        head_dim = attn.head_dim
        if (
            num_heads <= 0
            or num_kv_heads <= 0
            or num_heads % num_kv_heads != 0
            or head_dim not in self.SUPPORTED_CONTEXT_HEAD_DIMS
        ):
            return None

        cache_dtype = binding_to_torch_dtype(meta.kv_cache_manager.dtype)
        expected_width = (num_heads + 2 * num_kv_heads) * head_dim
        num_ctx_tokens = meta.num_ctx_tokens
        if (
            q.dtype not in self.SUPPORTED_DTYPES
            or cache_dtype != q.dtype
            or output.dtype != q.dtype
            or q.shape[0] != num_ctx_tokens
            or q.shape[1] != expected_width
            or output.numel() != q.shape[0] * num_heads * head_dim
        ):
            return None

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block not in self.SUPPORTED_PAGE_SIZES:
            return None
        max_seq_len = meta.max_seq_len
        attention_window_size = fwd.attention_window_size
        if (
            max_seq_len <= 0
            or not isinstance(attention_window_size, int)
            or attention_window_size < max_seq_len
        ):
            return None

        binding = self._get_v2_kv_page_offset_binding(attn, meta)
        if (
            binding is None
            or binding.pool_mapping_version is None
            or binding.kv_offsets_version is None
        ):
            return None
        pool_index = binding.pool_index
        kv_page_offset = binding.kv_page_offset

        return (
            q.device,
            q.dtype,
            q.shape[0],
            q.shape[1],
            output.dtype,
            output.numel(),
            input_type,
            mask_type,
            num_heads,
            num_kv_heads,
            head_dim,
            position_embedding_type,
            int(quant_mode),
            int(tokens_per_block),
            max_seq_len,
            attention_window_size,
            meta.kv_layout,
            cache_dtype,
            binding.manager_identity,
            binding.manager_impl_id,
            id(binding.pool_mapping),
            binding.pool_mapping_shape,
            binding.pool_mapping_version,
            id(binding.kv_offsets),
            binding.kv_offsets_shape,
            binding.kv_offsets_version,
            pool_index,
            kv_page_offset,
        )

    def _is_supported_with_reason(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        fwd: AttentionForwardArgs,
        *,
        phase: Optional[FmhaPhase] = None,
    ) -> tuple[bool, str]:
        """Return a conservative, side-effect-free whole-request support decision."""
        # PrimTS prepares workspace for every active request phase before
        # dispatch. Accept the phased dispatcher keyword, but do not narrow
        # support until that preparation is phase-aware too.
        del phase
        if q.device.type != "cuda":
            return False, "CUDA tensors are required."
        if not q.is_contiguous():
            return False, "the fused attention input must be contiguous."
        if k is not None or v is not None:
            return False, "only fused QKV input is supported."
        if not fwd.is_fused_qkv:
            return False, "only fused QKV input is supported."
        if meta.is_cross:
            return False, "cross attention is not supported."
        if meta.kv_cache_manager is None:
            return False, "a KV cache manager is required."
        if meta.kv_cache_block_offsets is None:
            return False, "paged KV-cache block offsets are required."
        if meta.host_kv_cache_pool_pointers is None:
            return False, "KV-cache pool pointers are required."
        if meta.host_kv_cache_pool_mapping is None:
            return False, "KV-cache pool mapping is required."
        if meta.kv_layout != "HND":
            return False, "only HND KV-cache layout is supported."
        kv_cache_manager = meta.kv_cache_manager
        get_page_index_upper_bound = getattr(
            kv_cache_manager.impl, "get_page_index_upper_bound", None
        )
        if callable(get_page_index_upper_bound):
            if kv_cache_manager.enable_swa_scratch_reuse:
                return False, "KVCacheManagerV2 SWA scratch reuse is not supported."
        else:
            if kv_cache_manager.num_pools != 1:
                return False, "KVCacheManagerV1 with multiple memory pools is not supported."
            pool_mapping = meta.host_kv_cache_pool_mapping
            local_layer_idx = attn.local_layer_idx
            num_local_layers = kv_cache_manager.num_local_layers
            if (
                pool_mapping.ndim != 2
                or pool_mapping.shape[1] < 2
                or local_layer_idx is None
                or local_layer_idx < 0
                or local_layer_idx >= pool_mapping.shape[0]
            ):
                return False, "KVCacheManagerV1 has an invalid layer-to-pool mapping."
            pool_index = int(pool_mapping[local_layer_idx, 0])
            layer_idx_in_pool = int(pool_mapping[local_layer_idx, 1])
            if pool_index != 0 or not 0 <= layer_idx_in_pool < num_local_layers:
                return False, "KVCacheManagerV1 has an invalid layer-to-pool mapping."

        output = fwd.output
        if output is None:
            return False, "an output tensor is required."
        if output.device != q.device or not output.is_contiguous():
            return False, "output must be a contiguous tensor on the query device."
        if fwd.output_sf is not None or fwd.out_scale is not None:
            return False, "quantized attention output is not supported."

        if attn.sparse_params is not None:
            return False, "sparse attention is not supported."
        if (
            fwd.sparse_runtime_params.sparse_kv_indices is not None
            or fwd.sparse_runtime_params.sparse_attn_indices is not None
        ):
            return False, "sparse attention metadata is not supported."
        if meta.num_sparse_topk > 0:
            return False, "sparse attention metadata is not supported."
        if meta.helix_position_offsets is not None:
            return False, "Helix parallelism is not supported."
        if fwd.relative_attention_bias is not None:
            return False, "relative attention bias is not supported."
        if fwd.attention_sinks is not None:
            return False, "attention sinks are not supported."
        if fwd.attention_mask_data is not None:
            return False, "custom attention masks are not supported."
        if fwd.enable_dsv4_epilogue_fusion:
            return False, "DSv4 epilogue fusion is not supported."
        if (
            fwd.sage_attn_num_elts_per_blk_q > 0
            or fwd.sage_attn_num_elts_per_blk_k > 0
            or fwd.sage_attn_num_elts_per_blk_v > 0
        ):
            return False, "SageAttention is not supported."

        if meta.beam_width != 1:
            return False, "beam search is not supported."
        if (
            meta.is_spec_decoding_enabled
            or meta.use_spec_decoding
            or meta.is_spec_dec_tree
            or meta.is_spec_dec_dynamic_tree
        ):
            return False, "speculative decoding is not supported by the initial adapter."

        try:
            mask_type = AttentionMaskType(fwd.mask_type)
        except (AttributeError, TypeError, ValueError):
            return False, "the attention mask is not causal or dense."
        if mask_type not in (AttentionMaskType.causal, AttentionMaskType.padding):
            return False, f"attention mask type {mask_type} is not supported."

        position_embedding_type = int(attn.position_embedding_type)
        if position_embedding_type in (4, 5, 6, 7, 10):
            return False, f"position embedding type {position_embedding_type} is not supported."

        try:
            quant_mode = QuantMode(attn.quant_mode)
        except (TypeError, ValueError):
            return False, "invalid KV-cache quantization mode."
        if quant_mode.has_kv_cache_quant():
            return False, "quantized KV cache is not supported by the initial adapter."

        input_type = fwd.attention_input_type
        if input_type not in (
            AttentionInputType.context_only,
            AttentionInputType.generation_only,
            AttentionInputType.mixed,
        ):
            return False, f"invalid attention input type {input_type}."
        num_contexts = int(meta.num_contexts)
        num_generations = int(meta.num_generations)
        has_context = num_contexts > 0 and input_type != AttentionInputType.generation_only
        has_generation = num_generations > 0 and input_type != AttentionInputType.context_only
        if not has_context and not has_generation:
            return False, "the request contains no active attention phase."
        if has_context and torch.cuda.is_current_stream_capturing():
            return False, "context planning is not CUDA-graph capturable."
        if has_context and (attn.attention_chunk_size or 0) != 0:
            return False, "chunked context attention is not supported."

        tokens_per_block = meta.tokens_per_block
        if tokens_per_block not in self.SUPPORTED_PAGE_SIZES:
            return False, (
                f"page size {tokens_per_block} is unsupported; "
                f"supported sizes are {sorted(self.SUPPORTED_PAGE_SIZES)}."
            )
        if attn.num_heads <= 0 or attn.num_kv_heads <= 0:
            return False, "query and KV head counts must be positive."
        is_mla = attn.is_mla_enable
        if is_mla:
            if attn.num_kv_heads != 1:
                return False, "MLA decode requires one logical KV head."
            if attn.num_heads > 128:
                return False, "MLA decode supports at most 128 local query heads."
        else:
            if attn.num_heads % attn.num_kv_heads != 0:
                return False, "the query head count must be divisible by the KV head count."
            if has_generation and attn.num_heads // attn.num_kv_heads > self.MAX_DECODE_GQA_RATIO:
                return False, f"decode GQA ratio exceeds {self.MAX_DECODE_GQA_RATIO}."

        if q.dtype not in self.SUPPORTED_DTYPES:
            return False, f"query dtype {q.dtype} is unsupported."
        cache_dtype = binding_to_torch_dtype(meta.kv_cache_manager.dtype)
        if cache_dtype != q.dtype:
            return False, f"query and KV-cache dtypes must match, got {q.dtype} and {cache_dtype}."
        if output.dtype != q.dtype:
            return False, f"output dtype must match query dtype, got {output.dtype} and {q.dtype}."

        if q.ndim != 2:
            return False, f"fused attention input must be rank 2, got rank {q.ndim}."
        if is_mla:
            if has_context or input_type != AttentionInputType.generation_only:
                return False, "MLA is supported only for generation-only requests."
            if q.dtype != torch.bfloat16 or output.dtype != torch.bfloat16:
                return False, "MLA decode requires BF16 query, cache, and output."
            if attn.kv_lora_rank != 512 or attn.qk_rope_head_dim != 64:
                return False, (
                    "MLA decode requires kv_lora_rank=512 and qk_rope_head_dim=64, got "
                    f"{attn.kv_lora_rank} and {attn.qk_rope_head_dim}."
                )
            if attn.head_dim != attn.kv_lora_rank + attn.qk_rope_head_dim:
                return False, "MLA head dimension must equal the latent plus RoPE dimensions."
            if attn.qk_nope_head_dim is None or attn.qk_nope_head_dim <= 0:
                return False, "MLA decode requires a positive qk_nope_head_dim."
            expected_width = attn.num_heads * (attn.kv_lora_rank + attn.qk_rope_head_dim)
            if q.shape[1] != expected_width:
                return False, f"MLA query width must be {expected_width}, got {q.shape[1]}."
            if output.numel() != q.shape[0] * attn.num_heads * attn.kv_lora_rank:
                return False, "MLA output has an incompatible extent."
        else:
            expected_width = (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim
            if q.shape[1] != expected_width:
                return False, f"fused QKV width must be {expected_width}, got {q.shape[1]}."
            if output.numel() != q.shape[0] * attn.num_heads * attn.head_dim:
                return False, "attention output has an incompatible extent."
            if has_context and attn.head_dim not in self.SUPPORTED_CONTEXT_HEAD_DIMS:
                return False, f"context head dimension {attn.head_dim} is unsupported."
            if has_generation and attn.head_dim not in self.SUPPORTED_DECODE_HEAD_DIMS:
                return False, f"decode head dimension {attn.head_dim} is unsupported."

        num_ctx_tokens = int(meta.num_ctx_tokens)
        num_gen_tokens = (
            q.shape[0]
            if input_type == AttentionInputType.generation_only
            else q.shape[0] - num_ctx_tokens
        )
        if has_generation and (
            num_gen_tokens <= 0 or num_generations <= 0 or num_gen_tokens % num_generations != 0
        ):
            return False, "generation tokens must be uniformly divisible across requests."
        if has_generation and num_gen_tokens != num_generations:
            return False, "only single-token generation is supported by the initial adapter."

        host_kv_lens = meta.kv_lens_runtime
        if host_kv_lens is None or host_kv_lens.numel() < num_contexts + num_generations:
            return False, "host KV lengths are required for safe policy selection."
        active_kv_lens = host_kv_lens[: num_contexts + num_generations]
        if active_kv_lens.numel() == 0 or int(active_kv_lens.min()) <= 0:
            return False, "every active request must contain at least one KV token."
        max_kv_length = int(active_kv_lens.max())
        max_seq_len = int(meta.max_seq_len)
        if max_kv_length > max_seq_len:
            return False, "an active KV length exceeds the configured maximum sequence length."
        attention_window_size = fwd.attention_window_size
        if not isinstance(attention_window_size, int) or attention_window_size <= 0:
            return False, "attention_window_size must be a positive integer."
        if attention_window_size < max_seq_len:
            return False, (
                "sliding-window attention uses cyclic TRT-LLM page tables, which are not "
                "compatible with the PrimTS native CSR page-table ABI."
            )

        if not is_mla and self._get_kv_page_offset(attn, meta, 0) is None:
            return False, "the K-to-V page displacement could not be resolved."
        return True, ""

    def _get_v2_kv_page_offset_binding(
        self,
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
    ) -> Optional[_V2KvPageOffsetBinding]:
        """Resolve one validated single-pool V2 layer binding."""
        manager = meta.kv_cache_manager
        if manager is None:
            return None
        manager_impl = manager.impl
        if not callable(getattr(manager_impl, "get_page_index_upper_bound", None)):
            return None
        num_pools = manager.num_pools
        if num_pools != 1:
            return None

        local_layer_idx = attn.local_layer_idx
        if local_layer_idx is None:
            local_layer_idx = int(attn.get_local_layer_idx(meta))
        if not isinstance(local_layer_idx, int) or local_layer_idx < 0:
            return None

        pool_mapping = meta.host_kv_cache_pool_mapping
        kv_offsets = manager.kv_offset
        if not isinstance(pool_mapping, torch.Tensor) or not isinstance(kv_offsets, torch.Tensor):
            return None
        pool_mapping_shape = tuple(pool_mapping.shape)
        kv_offsets_shape = tuple(kv_offsets.shape)
        if (
            len(pool_mapping_shape) != 2
            or pool_mapping_shape[1] < 2
            or local_layer_idx >= pool_mapping_shape[0]
            or len(kv_offsets_shape) != 1
        ):
            return None
        try:
            pool_mapping_version = pool_mapping._version
            kv_offsets_version = kv_offsets._version
        except RuntimeError:
            # Inference tensors do not expose version counters. Resolve them
            # for this call, but never retain a binding or publish a B1 support
            # certificate that could outlive the validated scalar reads.
            pool_mapping_version = None
            kv_offsets_version = None
            self._v2_kv_page_offset_binding = None

        binding = self._v2_kv_page_offset_binding
        if binding is not None and pool_mapping_version is not None:
            bound_manager = binding.manager_identity.get()
            # Check the weak manager identity before consulting the impl ID;
            # this prevents a recycled Python ID from validating a new manager.
            if bound_manager is manager and (
                binding.manager_impl_id == id(manager_impl)
                and binding.pool_mapping is pool_mapping
                and binding.pool_mapping_shape == pool_mapping_shape
                and binding.pool_mapping_version == pool_mapping_version
                and binding.kv_offsets is kv_offsets
                and binding.kv_offsets_shape == kv_offsets_shape
                and binding.kv_offsets_version == kv_offsets_version
                and binding.local_layer_idx == local_layer_idx
                and binding.pool_index == 0
            ):
                return binding
            self._v2_kv_page_offset_binding = None

        try:
            manager_identity = _WeakIdentity(manager)
        except TypeError:
            # Never retain an estimation or shutdown manager just to cache a
            # host-side scalar lookup.
            return None

        try:
            pool_index = self._read_host_tensor_scalar(pool_mapping, (local_layer_idx, 0))
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
        if (
            pool_index < 0
            or pool_index >= num_pools
            or pool_index >= kv_offsets_shape[0]
            or pool_index != 0
        ):
            return None
        try:
            kv_page_offset = self._read_host_tensor_scalar(kv_offsets, pool_index)
        except (IndexError, RuntimeError, TypeError, ValueError):
            return None
        if kv_page_offset <= 0:
            return None
        if pool_mapping_version is not None:
            try:
                versions_changed = (
                    pool_mapping._version != pool_mapping_version
                    or kv_offsets._version != kv_offsets_version
                )
            except RuntimeError:
                return None
            if versions_changed:
                return None
        elif (
            meta.kv_cache_manager is not manager
            or manager.impl is not manager_impl
            or meta.host_kv_cache_pool_mapping is not pool_mapping
            or manager.kv_offset is not kv_offsets
            or tuple(pool_mapping.shape) != pool_mapping_shape
            or tuple(kv_offsets.shape) != kv_offsets_shape
        ):
            return None

        # Cached engine-owned tensors are mutated only through PyTorch, so
        # identity, version, and shape cover supported mutation. A versionless
        # binding is returned only to its caller and is never retained.
        binding = _V2KvPageOffsetBinding(
            manager_identity=manager_identity,
            manager_impl_id=id(manager_impl),
            pool_mapping=pool_mapping,
            pool_mapping_shape=pool_mapping_shape,
            pool_mapping_version=pool_mapping_version,
            kv_offsets=kv_offsets,
            kv_offsets_shape=kv_offsets_shape,
            kv_offsets_version=kv_offsets_version,
            local_layer_idx=local_layer_idx,
            pool_index=pool_index,
            kv_page_offset=kv_page_offset,
        )
        if pool_mapping_version is not None:
            self._v2_kv_page_offset_binding = binding
        return binding

    @staticmethod
    def _read_host_tensor_scalar(tensor: torch.Tensor, index: object) -> int:
        return int(tensor[index])

    def _get_kv_page_offset(
        self,
        attn: "TrtllmAttention",
        meta: "TrtllmAttentionMetadata",
        seq_offset: int,
    ) -> Optional[int]:
        """Return the V-page displacement relative to a K page ID."""
        manager = meta.kv_cache_manager
        manager_impl = manager.impl
        if callable(getattr(manager_impl, "get_page_index_upper_bound", None)):
            num_pools = manager.num_pools
            if num_pools == 1:
                binding = self._get_v2_kv_page_offset_binding(attn, meta)
                return None if binding is None else binding.kv_page_offset
            if not isinstance(num_pools, int) or num_pools <= 1:
                return None

        # V1 and V2 multi-pool requests retain the original per-pool fallback.
        local_layer_idx = attn.local_layer_idx
        if local_layer_idx is None:
            local_layer_idx = int(attn.get_local_layer_idx(meta))
        pool_mapping = meta.host_kv_cache_pool_mapping
        pool_index = int(pool_mapping[local_layer_idx, 0])
        cache_key = (id(manager), pool_index)
        cached = self._kv_page_offset_cache.get(cache_key)
        if cached is not None:
            return cached

        if callable(getattr(manager_impl, "get_page_index_upper_bound", None)):
            kv_offsets = manager.kv_offset
            kv_offset = int(kv_offsets[pool_index])
            if kv_offset > 0:
                self._kv_page_offset_cache[cache_key] = kv_offset
                return kv_offset

        host_block_offsets = manager.host_kv_cache_block_offsets
        if host_block_offsets is None or host_block_offsets.ndim != 4:
            return None
        if pool_index >= host_block_offsets.shape[0]:
            return None

        rows = host_block_offsets[pool_index]
        if 0 <= seq_offset < rows.shape[0]:
            row_deltas = rows[seq_offset, 1] - rows[seq_offset, 0]
            positive = row_deltas[row_deltas > 0]
            if positive.numel() > 0:
                kv_offset = int(positive[0])
                self._kv_page_offset_cache[cache_key] = kv_offset
                return kv_offset
        all_deltas = rows[:, 1] - rows[:, 0]
        positive = all_deltas[all_deltas > 0]
        if positive.numel() == 0:
            return None
        kv_offset = int(positive[0])
        self._kv_page_offset_cache[cache_key] = kv_offset
        return kv_offset

    def _ensure_metadata_buffers(
        self,
        device: torch.device,
        row_capacity: int,
        column_capacity: int,
        page_size: int,
        *,
        need_context: bool = False,
    ) -> None:
        pages_per_kv_tile = 128 // page_size
        context_column_capacity = (
            (column_capacity + pages_per_kv_tile - 1) // pages_per_kv_tile
        ) * pages_per_kv_tile
        base_needs_allocation = (
            self._page_indices_buffer is None
            or self._fixed_indptr_buffer is None
            or self._interleaved_indptr_buffer is None
            or self._sequence_lengths_buffer is None
            or self._page_indices_buffer.device != device
            or self._metadata_row_capacity < row_capacity
            or self._metadata_column_capacity != column_capacity
        )
        context_needs_allocation = need_context and (
            self._context_page_indices_buffer is None
            or self._context_page_gather_indices_buffer is None
            or self._context_page_columns_buffer is None
            or self._context_last_page_indices_buffer is None
            or self._context_page_indices_buffer.device != device
            or self._context_metadata_row_capacity < row_capacity
            or self._context_page_column_capacity != context_column_capacity
        )
        if not base_needs_allocation and not context_needs_allocation:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "PrimTS metadata buffers must be allocated before CUDA graph capture."
            )
        retained_buffers: list[torch.Tensor] = []
        if base_needs_allocation:
            if (
                self._page_indices_buffer is not None
                and self._fixed_indptr_buffer is not None
                and self._interleaved_indptr_buffer is not None
                and self._sequence_lengths_buffer is not None
            ):
                retained_buffers.extend(
                    (
                        self._page_indices_buffer,
                        self._fixed_indptr_buffer,
                        self._interleaved_indptr_buffer,
                        self._sequence_lengths_buffer,
                    )
                )
            self._page_indices_buffer = torch.empty(
                (row_capacity, column_capacity), dtype=torch.int32, device=device
            )
            self._fixed_indptr_buffer = torch.arange(
                row_capacity + 1, dtype=torch.int32, device=device
            ).mul_(column_capacity)
            self._interleaved_indptr_buffer = torch.arange(
                row_capacity + 1, dtype=torch.int32, device=device
            ).mul_(2 * column_capacity)
            self._sequence_lengths_buffer = torch.empty(
                row_capacity, dtype=torch.int32, device=device
            )
            self._metadata_row_capacity = row_capacity
            self._metadata_column_capacity = column_capacity

        if context_needs_allocation:
            if (
                self._context_page_indices_buffer is not None
                and self._context_page_gather_indices_buffer is not None
                and self._context_page_columns_buffer is not None
                and self._context_last_page_indices_buffer is not None
            ):
                retained_buffers.extend(
                    (
                        self._context_page_indices_buffer,
                        self._context_page_gather_indices_buffer,
                        self._context_page_columns_buffer,
                        self._context_last_page_indices_buffer,
                    )
                )
            self._context_page_indices_buffer = torch.zeros(
                (row_capacity, 2, context_column_capacity),
                dtype=torch.int32,
                device=device,
            )
            self._context_page_gather_indices_buffer = torch.empty(
                (row_capacity, context_column_capacity),
                dtype=torch.int64,
                device=device,
            )
            self._context_page_columns_buffer = torch.arange(
                context_column_capacity,
                dtype=torch.int64,
                device=device,
            )
            self._context_last_page_indices_buffer = torch.empty(
                row_capacity,
                dtype=torch.int64,
                device=device,
            )
            self._context_metadata_row_capacity = row_capacity
            self._context_page_column_capacity = context_column_capacity

        if retained_buffers:
            # Captured copies and kernel nodes retain these addresses. A replay
            # still updates the old destinations, so keeping the allocations
            # alive preserves both pointer validity and live metadata values.
            self._retained_metadata_buffers.append(tuple(retained_buffers))

    def _make_fixed_stride_csr(
        self,
        block_tables: torch.Tensor,
        batch_size: int,
        page_size: int,
        *,
        max_kv_len: Optional[int] = None,
        allow_interleaved_tables: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build fixed-stride CSR, reusing native page-table storage when safe."""
        if block_tables.ndim != 3 or block_tables.shape[1] != 2:
            raise RuntimeError(
                "PrimTS expects block tables with shape [batch, 2, max_blocks], got "
                f"{tuple(block_tables.shape)}."
            )
        if block_tables.dtype != torch.int32:
            raise RuntimeError(f"PrimTS expects int32 block tables, got {block_tables.dtype}.")
        if batch_size <= 0 or batch_size > block_tables.shape[0]:
            raise RuntimeError(
                f"Invalid PrimTS CSR batch size {batch_size} for {block_tables.shape[0]} rows."
            )
        columns = int(block_tables.shape[-1])
        self._ensure_metadata_buffers(block_tables.device, batch_size, columns, page_size)
        if (
            self._page_indices_buffer is None
            or self._fixed_indptr_buffer is None
            or self._interleaved_indptr_buffer is None
        ):
            raise RuntimeError("PrimTS metadata buffers were not allocated.")
        if max_kv_len is not None and (max_kv_len <= 0 or max_kv_len > columns * page_size):
            raise RuntimeError(
                f"Invalid PrimTS maximum KV length {max_kv_len} for "
                f"{columns} pages of size {page_size}."
            )

        source_page_table = block_tables[:batch_size, 0, :]
        if (
            batch_size == 1
            and source_page_table.is_contiguous()
            and source_page_table.data_ptr() % 16 == 0
        ):
            return self._fixed_indptr_buffer[:2], source_page_table.view(-1)

        if allow_interleaved_tables and max_kv_len is not None:
            # Some schedules coalesce a 32-entry page-ID window. Keep every
            # page ID that can be consumed inside the K plane; any unused lanes
            # remain in-bounds in the adjacent V plane of the source table.
            pages_per_kv_tile = 128 // page_size
            padded_pages = ((max_kv_len + 127) // 128) * pages_per_kv_tile
            coalesced_pages = ((padded_pages + 31) // 32) * 32
            interleaved_tables = block_tables[:batch_size]
            if (
                interleaved_tables.is_contiguous()
                and interleaved_tables.data_ptr() % 16 == 0
                and coalesced_pages <= columns
            ):
                return (
                    self._interleaved_indptr_buffer[: batch_size + 1],
                    interleaved_tables.view(-1),
                )

        page_table = self._page_indices_buffer[:batch_size]
        page_table.copy_(source_page_table)
        return self._fixed_indptr_buffer[: batch_size + 1], page_table.reshape(-1)

    def _stage_context_metadata(
        self,
        block_tables: torch.Tensor,
        cu_kv_seqlens: torch.Tensor,
        sequence_lengths: torch.Tensor,
        *,
        batch_size: int,
        page_size: int,
        max_kv_len: int,
        window_left: int,
        cache_dense_page_alias: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build the live native context metadata entirely on the current stream."""
        if not cache_dense_page_alias:
            self._dense_context_page_alias = None
        if block_tables.ndim != 3 or block_tables.shape[1] != 2:
            raise RuntimeError(
                "PrimTS expects block tables with shape [batch, 2, max_blocks], got "
                f"{tuple(block_tables.shape)}."
            )
        if (
            block_tables.dtype != torch.int32
            or cu_kv_seqlens.dtype != torch.int32
            or sequence_lengths.dtype != torch.int32
        ):
            raise RuntimeError("PrimTS context metadata must use int32 tensors.")
        if cu_kv_seqlens.numel() < batch_size + 1:
            raise RuntimeError("PrimTS context cumulative KV lengths are too short.")
        if sequence_lengths.numel() < batch_size:
            raise RuntimeError("PrimTS context sequence lengths are too short.")

        pages_per_kv_tile = 128 // page_size
        active_pages = (max_kv_len + page_size - 1) // page_size
        required_padded_pages = (
            (active_pages + pages_per_kv_tile - 1) // pages_per_kv_tile
        ) * pages_per_kv_tile
        padded_page_capacity = self._context_page_column_capacity
        if (
            self._context_page_indices_buffer is None
            or self._context_page_gather_indices_buffer is None
            or self._context_page_columns_buffer is None
            or self._context_last_page_indices_buffer is None
            or batch_size > self._context_metadata_row_capacity
            or required_padded_pages > padded_page_capacity
            or active_pages > block_tables.shape[-1]
        ):
            raise RuntimeError("PrimTS context metadata storage was not prepared.")

        logical_kv_indptr = (
            cu_kv_seqlens
            if cu_kv_seqlens.ndim == 1 and cu_kv_seqlens.shape[0] == batch_size + 1
            else cu_kv_seqlens[: batch_size + 1]
        )
        seq_lens_kv = (
            sequence_lengths
            if sequence_lengths.ndim == 1 and sequence_lengths.shape[0] == batch_size
            else sequence_lengths[:batch_size]
        )
        padded_pages = required_padded_pages
        if (
            batch_size == 1
            and self.attn.head_dim == 128
            and window_left < 0
            and active_pages == padded_pages
        ):
            source_storage_capacity = (
                block_tables.untyped_storage().nbytes() // block_tables.element_size()
            )
            source_key: Optional[tuple[object, ...]] = None
            if cache_dense_page_alias:
                source_key = (
                    block_tables.device,
                    block_tables.dtype,
                    block_tables.data_ptr(),
                    tuple(block_tables.shape),
                    tuple(block_tables.stride()),
                    block_tables.storage_offset(),
                    source_storage_capacity,
                    page_size,
                    padded_pages,
                    batch_size,
                    int(self.attn.head_dim),
                    window_left,
                    active_pages,
                    max_kv_len,
                )
                cached_alias = self._dense_context_page_alias
                if cached_alias is not None and cached_alias.source_key == source_key:
                    return (
                        logical_kv_indptr,
                        seq_lens_kv,
                        cached_alias.dense_page_idx_kv,
                    )
            source_pages = block_tables[:1, 0:1, :active_pages]
            source_capacity = source_storage_capacity - source_pages.storage_offset()
            if (
                source_pages.is_contiguous()
                and source_pages.data_ptr() % 16 == 0
                and source_capacity >= 2 * padded_pages
            ):
                # Paired D128 reads only plane 0 and uses separately shifted K/V
                # pools. Plane 1 of this compact alias is therefore intentionally
                # unspecified; D256 reads both planes and must keep using the copy.
                dense_page_idx_kv = source_pages.as_strided(
                    (1, 2, padded_pages),
                    (2 * padded_pages, padded_pages, 1),
                )
                if source_key is not None:
                    self._dense_context_page_alias = _DenseContextPageAlias(
                        source=block_tables,
                        source_key=source_key,
                        dense_page_idx_kv=dense_page_idx_kv,
                    )
                return (
                    logical_kv_indptr,
                    seq_lens_kv,
                    dense_page_idx_kv,
                )

        dense_page_idx_kv = self._context_page_indices_buffer.view(-1)[
            : batch_size * 2 * padded_pages
        ].view(batch_size, 2, padded_pages)

        if batch_size == 1:
            source_pages = block_tables[:1, 0:1, :active_pages]
            dense_page_idx_kv[:, :, :active_pages].copy_(source_pages.expand(-1, 2, -1))
            if active_pages < padded_pages:
                last_page = block_tables[:1, 0:1, active_pages - 1 : active_pages]
                dense_page_idx_kv[:, :, active_pages:].copy_(
                    last_page.expand(-1, 2, padded_pages - active_pages)
                )
            return logical_kv_indptr, seq_lens_kv, dense_page_idx_kv

        last_page_indices = self._context_last_page_indices_buffer[:batch_size]
        torch.sub(seq_lens_kv, 1, out=last_page_indices)
        last_page_indices.div_(page_size, rounding_mode="floor")
        gather_indices = self._context_page_gather_indices_buffer.view(-1)[
            : batch_size * padded_pages
        ].view(batch_size, padded_pages)
        torch.minimum(
            self._context_page_columns_buffer[:padded_pages].view(1, -1),
            last_page_indices.view(-1, 1),
            out=gather_indices,
        )
        torch.gather(
            block_tables[:batch_size, 0:1, :].expand(-1, 2, -1),
            2,
            gather_indices.unsqueeze(1).expand(-1, 2, -1),
            out=dense_page_idx_kv,
        )
        return logical_kv_indptr, seq_lens_kv, dense_page_idx_kv

    def _get_mla_sequence_lengths(
        self,
        sequence_lengths: torch.Tensor,
        batch_size: int,
        *,
        copy_to_stable_storage: bool = False,
    ) -> torch.Tensor:
        """Return live MLA lengths with the storage guarantees PrimTS requires."""
        if sequence_lengths.dtype != torch.int32:
            raise RuntimeError(
                f"PrimTS expects int32 sequence lengths, got {sequence_lengths.dtype}."
            )
        if batch_size <= 0 or batch_size > sequence_lengths.numel():
            raise RuntimeError(
                f"Invalid PrimTS sequence-length batch size {batch_size} for "
                f"{sequence_lengths.numel()} entries."
            )
        active_sequence_lengths = sequence_lengths[:batch_size]
        if not copy_to_stable_storage and active_sequence_lengths.data_ptr() % 16 == 0:
            return active_sequence_lengths

        buffer = self._sequence_lengths_buffer
        if buffer is None or buffer.device != sequence_lengths.device:
            raise RuntimeError("PrimTS sequence-length storage was not prepared.")
        if batch_size > buffer.numel():
            raise RuntimeError(
                "PrimTS sequence-length storage must be sized before kernel execution."
            )
        aligned_sequence_lengths = buffer[:batch_size]
        aligned_sequence_lengths.copy_(active_sequence_lengths)
        if aligned_sequence_lengths.data_ptr() % 16 != 0:
            raise RuntimeError("PrimTS sequence-length storage is not 16-byte aligned.")
        return aligned_sequence_lengths

    def _update_workspace_allocation(self, workspace: torch.Tensor) -> None:
        """Invalidate plans that retain views into a reallocated workspace."""

        storage = workspace.untyped_storage()
        allocation = (
            workspace.device,
            storage.data_ptr(),
            storage.nbytes(),
            workspace.storage_offset(),
            workspace.numel(),
            workspace.element_size(),
        )
        if allocation == self._workspace_allocation:
            return
        self._workspace_allocation = allocation
        self._decode_wrappers.clear()
        self._mla_decode_wrappers.clear()

    def _get_or_plan_context_wrapper(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        *,
        batch_size: int,
        max_seq_len_q: int,
        max_seq_len_k: int,
        max_num_pages_per_seq_kv: int,
        page_size: int,
        mask_type: str,
        window_left: int,
        sm_scale: float,
        output_dtype: torch.dtype,
    ) -> "BatchPrefillPagedTSWrapper":
        wrapper = self._context_wrappers.get(batch_size)
        if wrapper is not None:
            return wrapper
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS context must be planned before CUDA graph capture.")
        from tensorrt_llm._torch.attention_backend.prims_ts.context import (
            BatchPrefillPagedTSWrapper,
        )

        wrapper = BatchPrefillPagedTSWrapper(kv_layout="HND")
        wrapper.plan_live(
            q,
            k_cache,
            v_cache,
            batch_size=batch_size,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            max_num_pages_per_seq_kv=max_num_pages_per_seq_kv,
            page_size=page_size,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=sm_scale,
            output_scale=1.0,
            out_dtype=output_dtype,
        )
        self._context_wrappers[batch_size] = wrapper
        return wrapper

    def _get_or_plan_decode_wrapper(
        self,
        paged_kv_indptr: torch.Tensor,
        paged_kv_indices: torch.Tensor,
        workspace_buffer: torch.Tensor,
        *,
        batch_size: int,
        num_qo_heads: int,
        num_kv_heads: int,
        head_dim: int,
        page_size: int,
        seq_len_q: int,
        max_kv_len: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        mask_type: str,
        window_left: int,
    ) -> "BatchDecodePagedTSWrapper":
        if batch_size <= 0:
            raise RuntimeError("PrimTS decode requires a positive batch size.")
        if paged_kv_indptr.numel() != batch_size + 1:
            raise RuntimeError("PrimTS decode indptr extent does not match the phase batch size.")
        wrapper = self._decode_wrappers.get(batch_size)
        if wrapper is not None:
            return wrapper
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS decode must be planned before CUDA graph capture.")
        from tensorrt_llm._torch.attention_backend.prims_ts.decode import BatchDecodePagedTSWrapper

        wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")
        wrapper.plan(
            paged_kv_indptr,
            paged_kv_indices,
            None,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            page_size,
            seq_len_q=seq_len_q,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=output_dtype,
            mask_type=mask_type,
            window_left=window_left,
            max_kv_len=max_kv_len,
            live_metadata=True,
            enable_pdl=self._enable_pdl,
            workspace_buffer=workspace_buffer,
        )
        self._decode_wrappers[batch_size] = wrapper
        return wrapper

    def _get_or_plan_mla_decode_wrapper(
        self,
        block_tables: torch.Tensor,
        seq_lens: torch.Tensor,
        workspace_buffer: torch.Tensor,
        *,
        batch_size: int,
        num_heads: int,
        kv_lora_rank: int,
        qk_rope_head_dim: int,
        page_size: int,
        max_seq_len_q: int,
        max_kv_len: int,
        q_dtype: torch.dtype,
        kv_dtype: torch.dtype,
        output_dtype: torch.dtype,
        mask_type: str,
    ) -> "BatchMLADecodePagedTSWrapper":
        if batch_size <= 0:
            raise RuntimeError("PrimTS MLA decode requires a positive batch size.")
        if block_tables.shape[0] != batch_size or seq_lens.numel() != batch_size:
            raise RuntimeError("PrimTS MLA metadata extents do not match the phase batch size.")
        wrapper = self._mla_decode_wrappers.get(batch_size)
        if wrapper is not None:
            return wrapper
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("PrimTS MLA decode must be planned before CUDA graph capture.")
        from tensorrt_llm._torch.attention_backend.prims_ts.mla_decode import (
            BatchMLADecodePagedTSWrapper,
        )

        wrapper = BatchMLADecodePagedTSWrapper()
        wrapper.plan(
            block_tables,
            seq_lens,
            num_heads,
            kv_lora_rank,
            qk_rope_head_dim,
            page_size,
            max_seq_len_q=max_seq_len_q,
            q_data_type=q_dtype,
            kv_data_type=kv_dtype,
            o_data_type=output_dtype,
            mask_type=mask_type,
            max_kv_len=max_kv_len,
            live_metadata=True,
            workspace_buffer=workspace_buffer,
        )
        self._mla_decode_wrappers[batch_size] = wrapper
        return wrapper

    def prepare_workspace(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor],
        v: Optional[torch.Tensor],
        metadata: "TrtllmAttentionMetadata",
        forward_args: AttentionForwardArgs,
        workspace: torch.Tensor,
    ) -> None:
        del k, v
        block_offsets = metadata.kv_cache_block_offsets
        if block_offsets is None:
            raise RuntimeError("PrimTS requires paged KV-cache block offsets.")
        column_capacity = int(block_offsets.shape[-1])
        input_type = forward_args.attention_input_type
        has_context = metadata.num_contexts > 0 and input_type != AttentionInputType.generation_only
        has_generation = (
            metadata.num_generations > 0 and input_type != AttentionInputType.context_only
        )
        self._ensure_metadata_buffers(
            q.device,
            max(int(metadata.max_num_requests), 1),
            column_capacity,
            int(metadata.tokens_per_block),
            need_context=has_context and not self.attn.is_mla_enable,
        )

        if self._multi_processor_count is None:
            self._multi_processor_count = torch.cuda.get_device_properties(
                q.device
            ).multi_processor_count

        required_preprocess_bytes = 0
        if has_context and not self.attn.is_mla_enable:
            context_layout = thop.get_trtllm_gen_context_workspace_layout(
                q.dtype,
                int(metadata.num_contexts),
                int(metadata.num_ctx_tokens),
                self.attn.num_heads,
                self.attn.head_dim,
                self.attn.rope_dim,
                True,
                False,
                skip_workspace=True,
            )
            required_preprocess_bytes = max(
                required_preprocess_bytes, int(context_layout["total_size"])
            )
        if has_generation and not self.attn.is_mla_enable:
            num_gen_tokens_for_layout = (
                q.shape[0]
                if input_type == AttentionInputType.generation_only
                else q.shape[0] - int(metadata.num_ctx_tokens)
            )
            generation_layout = thop.get_trtllm_gen_generation_workspace_layout(
                q.dtype,
                int(metadata.num_generations),
                num_gen_tokens_for_layout,
                self.attn.num_heads,
                self.attn.head_dim,
                self.attn.rope_dim,
                self.attn.num_kv_heads,
                0,
                False,
                skip_workspace=True,
            )
            required_preprocess_bytes = max(
                required_preprocess_bytes, int(generation_layout["total_size"])
            )

        if not has_generation:
            current_workspace_bytes = workspace.numel() * workspace.element_size()
            if current_workspace_bytes < required_preprocess_bytes:
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError(
                        "TRT-LLM QKV preprocessing workspace must be sized before "
                        "CUDA graph capture."
                    )
                required_numel = ceil_div(required_preprocess_bytes, workspace.element_size())
                workspace.resize_((required_numel,))
            self._update_workspace_allocation(workspace)
            return

        batch_size = int(metadata.num_generations)
        num_gen_tokens = (
            q.shape[0]
            if input_type == AttentionInputType.generation_only
            else q.shape[0] - int(metadata.num_ctx_tokens)
        )
        seq_len_q = num_gen_tokens // batch_size
        max_seq_len = self._get_static_max_kv_len(
            metadata,
            page_capacity=column_capacity,
            page_size=int(metadata.tokens_per_block),
        )
        mask_type = self._get_prims_mask_type(forward_args)
        # is_supported() rejects cyclic sliding-window page tables. Keep the
        # sizing key aligned with the non-windowed plan selected at runtime.
        window_left = -1

        if self.attn.is_mla_enable:
            required_bytes = self._mla_workspace_sizes.get(batch_size)
            if required_bytes is None:
                from tensorrt_llm._torch.attention_backend.prims_ts import (
                    get_prims_ts_batch_decode_mla_workspace_size,
                )

                required_bytes = get_prims_ts_batch_decode_mla_workspace_size(
                    batch_size,
                    self.attn.num_heads,
                    int(self.attn.kv_lora_rank),
                    int(self.attn.qk_rope_head_dim),
                    int(metadata.tokens_per_block),
                    max_seq_len,
                    max_seq_len_q=seq_len_q,
                    q_dtype=q.dtype,
                    kv_dtype=q.dtype,
                    out_dtype=forward_args.output.dtype,
                    mask_type=mask_type,
                    device=q.device,
                )
                self._mla_workspace_sizes[batch_size] = required_bytes
        else:
            required_bytes = self._decode_workspace_sizes.get(batch_size)
            if required_bytes is None:
                from tensorrt_llm._torch.attention_backend.prims_ts import (
                    get_prims_ts_batch_decode_workspace_size,
                )

                required_bytes = get_prims_ts_batch_decode_workspace_size(
                    batch_size,
                    self.attn.num_heads,
                    self.attn.num_kv_heads,
                    self.attn.head_dim,
                    int(metadata.tokens_per_block),
                    max_seq_len,
                    seq_len_q=seq_len_q,
                    q_dtype=q.dtype,
                    kv_dtype=q.dtype,
                    out_dtype=forward_args.output.dtype,
                    mask_type=mask_type,
                    window_left=window_left,
                    device=q.device,
                )
                self._decode_workspace_sizes[batch_size] = required_bytes

        decode_workspace_min_offset_bytes: Optional[int] = None
        if self.attn.is_mla_enable:
            required_workspace_bytes = required_bytes
        else:
            self._decode_workspace_required_bytes = required_bytes
            # QKV preprocessing leaves its query and sequence metadata live
            # while PrimTS runs. Keep PrimTS scratch in a separate aligned tail,
            # anchored to the final root allocation so cached batch profiles
            # retain a stable address across mixed-context layout changes.
            # FlashInfer requires the caller workspace base to be 32-byte aligned.
            decode_workspace_min_offset_bytes = pad_up(
                required_preprocess_bytes, _WORKSPACE_ALIGNMENT
            )
            required_workspace_bytes = decode_workspace_min_offset_bytes + required_bytes
        current_workspace_bytes = workspace.numel() * workspace.element_size()
        if current_workspace_bytes < required_workspace_bytes:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError(
                    "PrimTS caller workspace must be sized before CUDA graph capture."
                )
            required_numel = ceil_div(required_workspace_bytes, workspace.element_size())
            workspace.resize_((required_numel,))
        if decode_workspace_min_offset_bytes is not None:
            current_workspace_bytes = workspace.numel() * workspace.element_size()
            available_tail_bytes = current_workspace_bytes - required_bytes
            decode_workspace_offset_bytes = (
                available_tail_bytes // _WORKSPACE_ALIGNMENT * _WORKSPACE_ALIGNMENT
            )
            if decode_workspace_offset_bytes < decode_workspace_min_offset_bytes:
                raise RuntimeError("PrimTS decode workspace tail does not fit its root allocation.")
            self._decode_workspace_offset_bytes = decode_workspace_offset_bytes
        self._update_workspace_allocation(workspace)

    @staticmethod
    def _get_prims_mask_type(forward_args: AttentionForwardArgs) -> str:
        mask_type = AttentionMaskType(forward_args.mask_type)
        return "causal" if mask_type == AttentionMaskType.causal else "dense"

    @staticmethod
    def _get_bmm1_scale(attn: "TrtllmAttention") -> float:
        return 1.0 / (math.sqrt(attn.head_dim) * attn.q_scaling)

    @staticmethod
    def _standard_kv_views(
        kv_pool: torch.Tensor,
        kv_page_offset: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if kv_pool.ndim != 4:
            raise RuntimeError(
                f"PrimTS expects a flat rank-4 KV page pool, got {tuple(kv_pool.shape)}."
            )
        usable_pages = kv_pool.shape[0] - kv_page_offset
        if kv_page_offset <= 0 or usable_pages <= 0:
            raise RuntimeError(
                f"Invalid PrimTS K-to-V page displacement {kv_page_offset} "
                f"for a {kv_pool.shape[0]}-page pool."
            )
        return (
            kv_pool.narrow(0, 0, usable_pages),
            kv_pool.narrow(0, kv_page_offset, usable_pages),
        )

    def run_context(self, params: FmhaParams) -> None:
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError("PrimTS context requires QKV input and an output buffer.")
        if params.sequence_lengths is None or params.context_lengths is None:
            raise RuntimeError("PrimTS context requires sequence and context lengths.")
        if self._multi_processor_count is None:
            raise RuntimeError("PrimTS context workspace was not prepared.")

        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        attention_chunk_size = attn.attention_chunk_size or 0
        (
            q_processed,
            kv_pool,
            block_tables,
            _kv_scale_pool,
            _bmm1_scale,
            _bmm2_scale,
            fmha_workspace,
            cu_q_seqlens,
            cu_kv_seqlens,
            _max_q_len,
            _max_kv_len,
            window_left,
        ) = thop.trtllm_gen_context_preprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            fwd.mask_type,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            self._get_bmm1_scale(attn),
            1.0,
            attention_chunk_size,
            False,
            True,
            False,
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,
            fwd.cross_kv,
            False,
            skip_workspace=True,
        )
        if fmha_workspace.numel() != 0:
            raise RuntimeError("PrimTS context preprocessing returned an FMHA workspace.")
        # The returned pool and block table share the THOP flat-page index ABI.
        if kv_pool is None or block_tables is None:
            raise RuntimeError("TRT-LLM preprocessing did not return PrimTS KV metadata.")
        kv_page_offset = self._get_kv_page_offset(attn, meta, params.seq_offset)
        if kv_page_offset is None:
            raise RuntimeError("PrimTS could not resolve the K-to-V page displacement.")
        k_cache, v_cache = self._standard_kv_views(kv_pool, kv_page_offset)
        max_seq_len_q = int(meta.max_context_length)
        max_seq_len_k = int(meta.max_seq_len)
        logical_kv_indptr, seq_lens_kv, dense_page_idx_kv = self._stage_context_metadata(
            block_tables,
            cu_kv_seqlens,
            params.sequence_lengths,
            batch_size=params.batch_size,
            page_size=params.tokens_per_block,
            max_kv_len=max_seq_len_k,
            window_left=int(window_left),
            cache_dense_page_alias=not meta.kv_cache_manager.is_estimating_kv_cache,
        )
        mask_type = self._get_prims_mask_type(fwd)
        wrapper = self._get_or_plan_context_wrapper(
            q_processed,
            k_cache,
            v_cache,
            batch_size=params.batch_size,
            max_seq_len_q=max_seq_len_q,
            max_seq_len_k=max_seq_len_k,
            max_num_pages_per_seq_kv=int(dense_page_idx_kv.shape[-1]),
            page_size=params.tokens_per_block,
            mask_type=mask_type,
            window_left=window_left,
            sm_scale=self._get_bmm1_scale(attn),
            output_dtype=params.context_buf.dtype,
        )
        # This adapter owns the cached plan and constructs every live tensor.
        wrapper._run_live_unchecked(
            q_processed,
            k_cache,
            v_cache,
            params.context_buf,
            cu_q_seqlens,
            logical_kv_indptr,
            dense_page_idx_kv,
            seq_lens_kv,
        )

        thop.trtllm_gen_context_postprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.context_lengths,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_cos_sin,
            fwd.mrope_rotary_cos_sin,
            attn.local_layer_idx,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            fwd.mask_type,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            params.batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            self._get_bmm1_scale(attn),
            False,
            True,
            False,
            attention_chunk_size,
            self._multi_processor_count,
            skip_workspace=True,
        )

    def run_generation(self, params: FmhaParams) -> None:
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError("PrimTS decode requires QKV input and an output buffer.")
        if params.sequence_lengths is None:
            raise RuntimeError("PrimTS decode requires sequence lengths.")
        if self._multi_processor_count is None:
            raise RuntimeError("PrimTS decode workspace was not prepared.")

        attn = params.attn
        meta = params.meta
        fwd = params.fwd
        rope_params = attn.rope_params
        batch_size = params.batch_size
        attention_chunk_size = attn.attention_chunk_size or 0
        (
            q_processed,
            kv_pool,
            block_tables,
            _kv_scale_pool,
            _bmm1_scale,
            _bmm2_scale,
            fmha_workspace,
            _cu_seqlens,
            _max_q_len,
            _max_kv_len,
            window_left,
            is_multi_token_gen,
        ) = thop.trtllm_gen_generation_preprocess(
            params.qkv_input,
            params.workspace,
            params.sequence_lengths,
            params.spec_decoding_generation_lengths,
            params.spec_decoding_position_offsets,
            meta.kv_cache_block_offsets,
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            fwd.kv_scale_orig_quant,
            fwd.kv_scale_quant_orig,
            fwd.out_scale,
            attn.rotary_inv_freq,
            attn.rotary_cos_sin,
            fwd.mrope_position_deltas,
            attn.local_layer_idx,
            params.seq_offset,
            attn.num_heads,
            attn.num_kv_heads,
            attn.head_dim,
            params.tokens_per_block,
            attn.quant_mode,
            params.max_attention_window_size,
            params.cyclic_attention_window_size,
            params.num_tokens,
            batch_size,
            params.input_seq_length,
            params.max_past_kv_length,
            rope_params.dim,
            rope_params.theta,
            int(rope_params.scale_type),
            rope_params.scale,
            rope_params.max_positions,
            attn.position_embedding_type,
            self._get_bmm1_scale(attn),
            1.0,
            False,
            attn.predicted_tokens_per_seq,
            attention_chunk_size,
            self._multi_processor_count,
            params.total_num_blocks,
            params.kv_factor,
            True,
            False,
            skip_workspace=True,
        )
        if fmha_workspace.numel() != 0:
            raise RuntimeError("PrimTS generation preprocessing returned an FMHA workspace.")
        if is_multi_token_gen:
            raise RuntimeError("PrimTS was selected for unsupported speculative decoding.")
        # The returned pool and block table share the THOP flat-page index ABI.
        if kv_pool is None or block_tables is None:
            raise RuntimeError("TRT-LLM preprocessing did not return PrimTS KV metadata.")
        kv_page_offset = self._get_kv_page_offset(attn, meta, params.seq_offset)
        if kv_page_offset is None:
            raise RuntimeError("PrimTS could not resolve the K-to-V page displacement.")
        k_cache, v_cache = self._standard_kv_views(kv_pool, kv_page_offset)
        max_seq_len = self._get_static_max_kv_len(
            meta,
            page_capacity=int(block_tables.shape[-1]),
            page_size=params.tokens_per_block,
        )
        paged_kv_indptr, paged_kv_indices = self._make_fixed_stride_csr(
            block_tables,
            batch_size,
            params.tokens_per_block,
            max_kv_len=max_seq_len,
            allow_interleaved_tables=True,
        )
        seq_lens = params.sequence_lengths[:batch_size]
        query = q_processed.view(
            batch_size,
            params.input_seq_length,
            attn.num_heads,
            attn.head_dim,
        )
        output = params.context_buf.view_as(query)
        if params.input_seq_length == 1:
            query = query[:, 0]
            output = output[:, 0]
        mask_type = self._get_prims_mask_type(fwd)
        decode_workspace = self._get_decode_workspace(params.workspace)
        wrapper = self._get_or_plan_decode_wrapper(
            paged_kv_indptr,
            paged_kv_indices,
            decode_workspace,
            batch_size=batch_size,
            num_qo_heads=attn.num_heads,
            num_kv_heads=attn.num_kv_heads,
            head_dim=attn.head_dim,
            page_size=params.tokens_per_block,
            seq_len_q=params.input_seq_length,
            max_kv_len=max_seq_len,
            q_dtype=query.dtype,
            kv_dtype=k_cache.dtype,
            output_dtype=output.dtype,
            mask_type=mask_type,
            window_left=window_left,
        )
        # Only the fused global-memory reducer consumes the counter/control
        # tail. Direct, cluster-reduced, and separately reduced plans either
        # overwrite their scratch or do not read this span.
        if wrapper._requires_control_reset:
            control_offset = wrapper._workspace_layout.split_kv_counter.byte_offset
            decode_workspace[control_offset : wrapper._workspace_layout.total_bytes].zero_()
        wrapper.run(
            query,
            (k_cache, v_cache),
            seq_lens,
            paged_kv_indptr=paged_kv_indptr,
            paged_kv_indices=paged_kv_indices,
            bmm1_scale=self._get_bmm1_scale(attn),
            bmm2_scale=1.0,
            out=output,
        )

    def _get_decode_workspace(
        self,
        root_workspace: torch.Tensor,
    ) -> torch.Tensor:
        """Return the caller-owned PrimTS tail after QKV preprocessing storage."""

        byte_offset = self._decode_workspace_offset_bytes
        if byte_offset is None:
            raise RuntimeError("PrimTS decode workspace was not prepared.")
        root_bytes = root_workspace.reshape(-1).view(torch.uint8)
        byte_end = byte_offset + self._decode_workspace_required_bytes
        if byte_end > root_bytes.numel():
            raise RuntimeError("PrimTS decode workspace was not sized before kernel execution.")
        return root_bytes[byte_offset:byte_end]

    def run_mla_generation(self, params: FmhaParams) -> None:
        if params.qkv_input is None or params.context_buf is None:
            raise RuntimeError("PrimTS MLA decode requires query input and an output buffer.")
        if params.sequence_lengths is None:
            raise RuntimeError("PrimTS MLA decode requires sequence lengths.")

        attn = params.attn
        meta = params.meta
        batch_size = params.batch_size
        kv_cache, block_tables, _kv_scale_pool = thop.build_trtllm_gen_kv_cache_metadata(
            meta.host_kv_cache_pool_pointers,
            meta.host_kv_cache_pool_mapping,
            meta.kv_cache_block_offsets,
            attn.local_layer_idx,
            attn.num_kv_heads,
            params.tokens_per_block,
            attn.head_dim,
            params.kv_factor,
            params.total_num_blocks,
            attn.quant_mode,
            params.seq_offset,
            batch_size,
            params.qkv_input.dtype,
        )
        # The returned pool and block table share the THOP flat-page index ABI.
        if kv_cache is None or block_tables is None:
            raise RuntimeError("TRT-LLM did not return PrimTS MLA KV metadata.")
        max_seq_len = self._get_static_max_kv_len(
            meta,
            page_capacity=int(block_tables.shape[-1]),
            page_size=params.tokens_per_block,
        )
        _, page_indices = self._make_fixed_stride_csr(
            block_tables,
            batch_size,
            params.tokens_per_block,
            max_kv_len=max_seq_len,
            allow_interleaved_tables=True,
        )
        dense_block_tables = page_indices.view(batch_size, -1)
        seq_len_q = params.input_seq_length
        query = params.qkv_input.view(
            batch_size,
            seq_len_q,
            attn.num_heads,
            int(attn.kv_lora_rank) + int(attn.qk_rope_head_dim),
        )
        output = params.context_buf.view(
            batch_size,
            seq_len_q,
            attn.num_heads,
            int(attn.kv_lora_rank),
        )
        bmm1_scale = 1.0 / (
            attn.q_scaling * math.sqrt(int(attn.qk_nope_head_dim) + int(attn.qk_rope_head_dim))
        )
        mask_type = self._get_prims_mask_type(params.fwd)
        seq_lens = self._get_mla_sequence_lengths(
            params.sequence_lengths,
            batch_size,
        )
        caller_workspace = params.workspace.reshape(-1).view(torch.uint8)
        wrapper = self._get_or_plan_mla_decode_wrapper(
            dense_block_tables,
            seq_lens,
            caller_workspace,
            batch_size=batch_size,
            num_heads=attn.num_heads,
            kv_lora_rank=int(attn.kv_lora_rank),
            qk_rope_head_dim=int(attn.qk_rope_head_dim),
            page_size=params.tokens_per_block,
            max_seq_len_q=seq_len_q,
            max_kv_len=max_seq_len,
            q_dtype=query.dtype,
            kv_dtype=kv_cache.dtype,
            output_dtype=output.dtype,
            mask_type=mask_type,
        )
        wrapper.run(
            query,
            kv_cache,
            block_tables=dense_block_tables,
            seq_lens=seq_lens,
            out=output,
            bmm1_scale=bmm1_scale,
            bmm2_scale=1.0,
        )


__all__ = ["PrimsTSFmha"]
