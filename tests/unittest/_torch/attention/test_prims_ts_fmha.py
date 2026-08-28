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

import ast
import gc
import inspect
import math
import weakref
from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from packaging.version import Version

import tensorrt_llm._torch.attention_backend.fmha.prims_ts as prims_ts_module
from tensorrt_llm._torch.attention_backend.fmha.fallback import FallbackFmha
from tensorrt_llm._torch.attention_backend.fmha.interface import FmhaPhase
from tensorrt_llm._torch.attention_backend.fmha.phased import FmhaParams, PhasedFmha
from tensorrt_llm._torch.attention_backend.fmha.prims_ts import PrimsTSFmha
from tensorrt_llm._torch.attention_backend.fmha.registry import get_enabled_fmha_lib_classes
from tensorrt_llm._torch.attention_backend.interface import (
    AttentionForwardArgs,
    AttentionInputType,
    AttentionMetadata,
    PredefinedAttentionMask,
)
from tensorrt_llm._torch.attention_backend.trtllm import TrtllmAttentionMetadata
from tensorrt_llm.bindings import DataType


class _WeakManager(SimpleNamespace):
    """Simple weak-referenceable stand-in for KVCacheManager."""


class _TensorSpec:
    """Minimal tensor-like object for the pure support predicate."""

    def __init__(
        self,
        shape: tuple[int, ...],
        dtype: torch.dtype,
        *,
        device: str = "cuda",
        contiguous: bool = True,
    ) -> None:
        self.shape = shape
        self.dtype = dtype
        self.device = torch.device(device)
        self.ndim = len(shape)
        self._contiguous = contiguous

    def is_contiguous(self) -> bool:
        return self._contiguous

    def numel(self) -> int:
        return math.prod(self.shape)

    def data_ptr(self) -> int:
        return id(self)


class _Attention:
    def __init__(
        self,
        *,
        head_dim: int = 128,
        is_mla: bool = False,
        num_heads: int = 8,
        num_kv_heads: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.num_kv_heads = (1 if is_mla else 2) if num_kv_heads is None else num_kv_heads
        self.head_dim = head_dim
        self.is_mla_enable = is_mla
        self.kv_lora_rank = 512 if is_mla else None
        self.qk_rope_head_dim = 64 if is_mla else None
        self.qk_nope_head_dim = 128 if is_mla else None
        self.v_head_dim = 128 if is_mla else None
        self.predicted_tokens_per_seq = 1
        self.sparse_params = None
        self.position_embedding_type = 0
        self.quant_mode = 0
        self.q_scaling = 1.0
        self.attention_chunk_size = 0
        self.rope_dim = head_dim
        self.local_layer_idx = 0
        self.rope_params = SimpleNamespace(
            dim=head_dim,
            theta=10000.0,
            scale_type=0,
            scale=1.0,
            max_positions=4096,
        )
        self.rotary_inv_freq = None
        self.rotary_cos_sin = None


def _function_source(path: Path, function_name: str) -> str:
    source = path.read_text()
    tree = ast.parse(source)
    function = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name
    )
    assert function.end_lineno is not None
    return "".join(source.splitlines(keepends=True)[function.lineno - 1 : function.end_lineno])


def _support_result(
    *,
    attention_input_type: AttentionInputType,
    head_dim: int = 128,
    num_heads: int = 8,
    num_kv_heads: int | None = None,
    dtype: torch.dtype = torch.bfloat16,
    output_dtype: torch.dtype | None = None,
    kv_dtype: DataType | None = None,
    tokens_per_block: int = 32,
    is_mla: bool = False,
    is_fused_qkv: bool = True,
    has_separate_kv: bool = False,
    has_paged_cache: bool = True,
    is_cross: bool = False,
    beam_width: int = 1,
    use_spec_decoding: bool = False,
    is_spec_dec_tree: bool = False,
    has_attention_sinks: bool = False,
    has_relative_attention_bias: bool = False,
    has_sparse_attention: bool = False,
    position_embedding_type: int = 0,
    kv_lora_rank: int | None = None,
    qk_rope_head_dim: int | None = None,
    has_output: bool = True,
    attention_window_size: int = 128,
    attention_chunk_size: int = 0,
    max_seq_len: int = 128,
    kv_layout: str = "HND",
    num_kv_cache_pools: int = 1,
    use_kv_cache_v2: bool = False,
    enable_swa_scratch_reuse: bool = False,
    phase: FmhaPhase | None = None,
) -> tuple[bool, str]:
    attn = _Attention(
        head_dim=head_dim,
        is_mla=is_mla,
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
    )
    attn.position_embedding_type = position_embedding_type
    attn.attention_chunk_size = attention_chunk_size
    if has_sparse_attention:
        attn.sparse_params = SimpleNamespace(algorithm="mqa_gqa")
    if kv_lora_rank is not None:
        attn.kv_lora_rank = kv_lora_rank
    if qk_rope_head_dim is not None:
        attn.qk_rope_head_dim = qk_rope_head_dim

    output_dtype = dtype if output_dtype is None else output_dtype
    if kv_dtype is None:
        kv_dtype = DataType.BF16 if dtype == torch.bfloat16 else DataType.HALF
    q_width = attn.num_heads * head_dim
    if is_fused_qkv and not is_mla:
        q_width += 2 * attn.num_kv_heads * head_dim
    q = _TensorSpec((4, q_width), dtype)
    output_width = attn.num_heads * (512 if is_mla else head_dim)
    forward_args = AttentionForwardArgs(
        output=_TensorSpec((4, output_width), output_dtype) if has_output else None,
        attention_input_type=attention_input_type,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=attention_window_size,
        attention_sinks=torch.empty(1) if has_attention_sinks else None,
        relative_attention_bias=torch.empty(1) if has_relative_attention_bias else None,
        is_fused_qkv=is_fused_qkv,
    )
    if attention_input_type == AttentionInputType.context_only:
        num_contexts, num_generations, num_ctx_tokens = 1, 0, 4
        kv_lens = [4]
    elif attention_input_type == AttentionInputType.generation_only:
        num_contexts, num_generations, num_ctx_tokens = 0, 4, 0
        kv_lens = [128, 96, 64, 32]
    else:
        num_contexts, num_generations, num_ctx_tokens = 1, 1, 3
        kv_lens = [3, 128]
    metadata = SimpleNamespace(
        helix_position_offsets=None,
        num_sparse_topk=0,
        use_spec_decoding=use_spec_decoding,
        is_spec_dec_tree=is_spec_dec_tree,
        is_spec_decoding_enabled=use_spec_decoding,
        kv_cache_block_offsets=torch.empty(1) if has_paged_cache else None,
        host_kv_cache_pool_pointers=torch.empty(1),
        host_kv_cache_pool_mapping=torch.zeros((1, 2), dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            dtype=kv_dtype,
            impl=(
                SimpleNamespace(get_page_index_upper_bound=lambda *args: 128)
                if use_kv_cache_v2
                else SimpleNamespace()
            ),
            enable_swa_scratch_reuse=enable_swa_scratch_reuse,
            num_local_layers=1,
            num_pools=num_kv_cache_pools,
        ),
        is_cross=is_cross,
        beam_width=beam_width,
        tokens_per_block=tokens_per_block,
        kv_layout=kv_layout,
        num_contexts=num_contexts,
        num_generations=num_generations,
        num_ctx_tokens=num_ctx_tokens,
        kv_lens_runtime=torch.tensor(kv_lens, dtype=torch.int32),
        max_seq_len=max_seq_len,
    )
    fmha = PrimsTSFmha(attn)
    fmha._get_kv_page_offset = Mock(return_value=1)
    k = _TensorSpec((4, attn.num_kv_heads * head_dim), dtype) if has_separate_kv else None
    v = _TensorSpec((4, attn.num_kv_heads * head_dim), dtype) if has_separate_kv else None
    return fmha._is_supported_with_reason(
        q,
        k,
        v,
        attn,
        metadata,
        forward_args,
        phase=phase,
    )


def _make_b1_context_cache_inputs(
    *,
    local_layer_idx: int = 0,
    head_dim: int = 128,
) -> tuple[
    PrimsTSFmha,
    _TensorSpec,
    SimpleNamespace,
    AttentionForwardArgs,
]:
    attn = _Attention(head_dim=head_dim)
    attn.local_layer_idx = local_layer_idx
    fmha = PrimsTSFmha(attn)
    num_tokens = 4
    q = _TensorSpec(
        (num_tokens, (attn.num_heads + 2 * attn.num_kv_heads) * head_dim),
        torch.bfloat16,
    )
    output = _TensorSpec(
        (num_tokens, attn.num_heads * head_dim),
        torch.bfloat16,
    )
    manager = _WeakManager(
        dtype=DataType.BF16,
        impl=SimpleNamespace(get_page_index_upper_bound=lambda *args: 128),
        enable_swa_scratch_reuse=False,
        num_pools=1,
        kv_offset=torch.tensor([64], dtype=torch.int32),
    )
    metadata = SimpleNamespace(
        _attention_owner=attn,
        _prims_ts_b1_context_support_key=None,
        is_cuda_graph=False,
        is_cross=False,
        kv_cache_manager=manager,
        kv_cache_block_offsets=torch.empty(1),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
        kv_layout="HND",
        beam_width=1,
        is_spec_decoding_enabled=False,
        use_spec_decoding=False,
        is_spec_dec_tree=False,
        is_spec_dec_dynamic_tree=False,
        runtime_features=SimpleNamespace(
            chunked_prefill=False,
            cache_reuse=False,
            has_speculative_draft_tokens=False,
        ),
        num_sparse_topk=0,
        helix_position_offsets=None,
        num_contexts=1,
        num_generations=0,
        num_ctx_tokens=num_tokens,
        tokens_per_block=32,
        max_seq_len=128,
        kv_lens_runtime=torch.tensor([num_tokens], dtype=torch.int32),
    )
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.mixed,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=128,
        is_fused_qkv=True,
    )
    return fmha, q, metadata, forward_args


def _make_b1_forward_inputs(
    *,
    attention_input_type: AttentionInputType = AttentionInputType.context_only,
    num_ctx_tokens: int = 4,
    prompt_length: int = 4,
    kv_length: int = 4,
) -> tuple[
    PrimsTSFmha,
    torch.Tensor,
    SimpleNamespace,
    AttentionForwardArgs,
]:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    q = torch.zeros(
        (num_ctx_tokens, (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim),
        dtype=torch.bfloat16,
    )
    output = torch.zeros(
        (num_ctx_tokens, attn.num_heads * attn.head_dim),
        dtype=torch.bfloat16,
    )
    metadata = SimpleNamespace(
        _attention_owner=attn,
        effective_workspace=torch.empty(16, dtype=torch.uint8),
        kv_cache_block_offsets=torch.empty((1, 1, 2, 4), dtype=torch.int32),
        cache_indirection=None,
        beam_width=1,
        num_contexts=1,
        num_generations=0,
        num_ctx_tokens=num_ctx_tokens,
        tokens_per_block=32,
        is_cross=False,
        kv_lens_cuda_runtime=torch.tensor([kv_length], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([kv_length], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([prompt_length], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([prompt_length], dtype=torch.int32),
    )
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=attention_input_type,
        attention_mask=PredefinedAttentionMask.CAUSAL,
        attention_window_size=128,
        is_fused_qkv=True,
    )
    return fmha, q, metadata, forward_args


def _capture_b1_forward_params(
    monkeypatch: pytest.MonkeyPatch,
    fmha: PrimsTSFmha,
    q: torch.Tensor,
    metadata: SimpleNamespace,
    forward_args: AttentionForwardArgs,
    *,
    generic: bool,
) -> tuple[FmhaParams, list[str]]:
    events: list[str] = []
    captured: list[FmhaParams] = []

    def get_fp8(*args: object, **kwargs: object) -> bool:
        events.append("fp8")
        return False

    def prepare(*args: object, **kwargs: object) -> None:
        events.append("prepare")

    def get_total_blocks(*args: object, **kwargs: object) -> int:
        events.append("blocks")
        return 96

    def run(params: FmhaParams) -> None:
        events.append("run")
        captured.append(params)

    monkeypatch.setattr(fmha, "get_fp8_context_fmha", get_fp8)
    monkeypatch.setattr(fmha, "prepare_workspace", prepare)
    monkeypatch.setattr(fmha, "_get_total_num_blocks", get_total_blocks)
    monkeypatch.setattr(fmha, "run_context", run)
    if generic:
        PhasedFmha.forward(fmha, q, None, None, metadata, forward_args)
    else:
        fmha.forward(q, None, None, metadata, forward_args)

    assert len(captured) == 1
    return captured[0], events


def test_b1_context_support_cache_reuses_first_layer_positive_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    first, first_q, metadata, first_args = _make_b1_context_cache_inputs()
    second, second_q, _, second_args = _make_b1_context_cache_inputs(local_layer_idx=1)
    second_args.output = _TensorSpec(first_args.output.shape, first_args.output.dtype)
    first_full_check = Mock(wraps=first._is_supported_with_reason)
    second_full_check = Mock(wraps=second._is_supported_with_reason)
    monkeypatch.setattr(first, "_is_supported_with_reason", first_full_check)
    monkeypatch.setattr(second, "_is_supported_with_reason", second_full_check)

    assert first.is_supported(first_q, None, None, metadata, first_args)
    support_key = metadata._prims_ts_b1_context_support_key
    assert support_key == first._get_b1_context_support_key(
        first_q, None, None, first.attn, metadata, first_args
    )
    assert first_full_check.call_count == 1

    assert second_q.data_ptr() != first_q.data_ptr()
    assert second_args.output.data_ptr() != first_args.output.data_ptr()
    assert second.is_supported(second_q, None, None, metadata, second_args)
    assert second_full_check.call_count == 0
    assert metadata._prims_ts_b1_context_support_key is support_key


def test_ungrouped_support_cache_live_capture_runs_full_check(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capturing = False
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: capturing)
    fmha, q, metadata, forward_args = _make_b1_context_cache_inputs()
    full_check = Mock(wraps=fmha._is_supported_with_reason)
    monkeypatch.setattr(fmha, "_is_supported_with_reason", full_check)

    assert fmha.is_supported(q, None, None, metadata, forward_args)
    assert metadata._prims_ts_b1_context_support_key is not None
    assert full_check.call_count == 1

    capturing = True
    assert not fmha.is_supported(q, None, None, metadata, forward_args)
    assert full_check.call_count == 2


def test_b1_context_support_cache_mismatch_runs_full_check_without_replacement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    first, first_q, metadata, first_args = _make_b1_context_cache_inputs()
    assert first.is_supported(first_q, None, None, metadata, first_args)
    support_key = metadata._prims_ts_b1_context_support_key

    different, different_q, _, different_args = _make_b1_context_cache_inputs(
        local_layer_idx=1, head_dim=256
    )
    different_full_check = Mock(wraps=different._is_supported_with_reason)
    monkeypatch.setattr(different, "_is_supported_with_reason", different_full_check)

    assert different.is_supported(different_q, None, None, metadata, different_args)
    assert different_full_check.call_count == 1
    assert metadata._prims_ts_b1_context_support_key is support_key


def test_b1_context_support_cache_does_not_store_unsupported_decision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fmha, q, metadata, forward_args = _make_b1_context_cache_inputs(head_dim=64)
    full_check = Mock(wraps=fmha._is_supported_with_reason)
    monkeypatch.setattr(fmha, "_is_supported_with_reason", full_check)

    assert not fmha.is_supported(q, None, None, metadata, forward_args)
    assert full_check.call_count == 1
    assert metadata._prims_ts_b1_context_support_key is None


@pytest.mark.parametrize(
    "case",
    ["cuda-graph", "speculative", "cross", "v1", "multi-pool", "sparse"],
)
def test_b1_context_support_cache_bypasses_non_exact_requests(case: str) -> None:
    fmha, q, metadata, forward_args = _make_b1_context_cache_inputs()
    if case == "cuda-graph":
        metadata.is_cuda_graph = True
    elif case == "speculative":
        metadata.use_spec_decoding = True
    elif case == "cross":
        metadata.is_cross = True
    elif case == "v1":
        metadata.kv_cache_manager.impl = SimpleNamespace()
    elif case == "multi-pool":
        metadata.kv_cache_manager.num_pools = 2
    else:
        fmha.attn.sparse_params = SimpleNamespace(algorithm="mqa_gqa")

    assert (
        fmha._get_b1_context_support_key(q, None, None, fmha.attn, metadata, forward_args) is None
    )


def test_b1_context_support_cache_is_request_scoped_and_non_comparing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = object.__new__(TrtllmAttentionMetadata)
    metadata._prims_ts_b1_context_support_key = ("positive",)
    prepare_error = RuntimeError("stop after request cache reset")
    monkeypatch.setattr(
        AttentionMetadata,
        "prepare",
        Mock(side_effect=prepare_error),
    )

    with pytest.raises(RuntimeError, match="stop after request cache reset"):
        metadata.prepare()
    assert metadata._prims_ts_b1_context_support_key is None
    cache_field = TrtllmAttentionMetadata.__dataclass_fields__["_prims_ts_b1_context_support_key"]
    assert not cache_field.init
    assert not cache_field.repr
    assert not cache_field.compare


@pytest.mark.parametrize(
    "attention_input_type",
    [AttentionInputType.context_only, AttentionInputType.mixed],
    ids=["context-only", "mixed-without-generation"],
)
def test_b1_forward_fast_path_matches_generic_params_and_order(
    monkeypatch: pytest.MonkeyPatch,
    attention_input_type: AttentionInputType,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs(
        attention_input_type=attention_input_type,
        num_ctx_tokens=4,
        prompt_length=3,
        kv_length=105,
    )
    generic_params, generic_events = _capture_b1_forward_params(
        monkeypatch,
        fmha,
        q,
        metadata,
        forward_args,
        generic=True,
    )
    fast_params, fast_events = _capture_b1_forward_params(
        monkeypatch,
        fmha,
        q,
        metadata,
        forward_args,
        generic=False,
    )

    assert generic_events == ["fp8", "prepare", "blocks", "run"]
    assert fast_events == generic_events
    identity_fields = {"attn", "meta", "fwd", "workspace"}
    for params_field in fields(FmhaParams):
        generic_value = getattr(generic_params, params_field.name)
        fast_value = getattr(fast_params, params_field.name)
        if params_field.name in identity_fields:
            assert fast_value is generic_value
        elif isinstance(generic_value, torch.Tensor):
            assert isinstance(fast_value, torch.Tensor)
            torch.testing.assert_close(fast_value, generic_value)
            assert fast_value.shape == generic_value.shape
            assert fast_value.stride() == generic_value.stride()
            assert fast_value.storage_offset() == generic_value.storage_offset()
        else:
            assert fast_value == generic_value

    assert fast_params.attention_input is q
    assert fast_params.qkv_input is q
    assert fast_params.sequence_lengths is metadata.kv_lens_cuda_runtime
    assert fast_params.context_lengths is metadata.prompt_lens_cuda_runtime
    assert fast_params.context_buf is not None
    assert fast_params.context_buf.data_ptr() == forward_args.output.data_ptr()
    assert fast_params.context_buf.shape == (4, fmha.attn.num_heads, fmha.attn.head_dim)
    assert generic_params.attention_input is not q
    assert generic_params.sequence_lengths is not metadata.kv_lens_cuda_runtime
    assert generic_params.context_lengths is not metadata.prompt_lens_cuda_runtime


def test_b1_forward_fast_path_reads_live_runtime_views_and_prefix_lengths(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs(
        num_ctx_tokens=4,
        prompt_length=3,
        kv_length=105,
    )
    captured: list[FmhaParams] = []
    monkeypatch.setattr(fmha, "get_fp8_context_fmha", Mock(return_value=False))
    monkeypatch.setattr(fmha, "prepare_workspace", Mock())
    monkeypatch.setattr(fmha, "_get_total_num_blocks", Mock(return_value=96))
    monkeypatch.setattr(fmha, "run_context", captured.append)

    fmha.forward(q, None, None, metadata, forward_args)
    first = captured[-1]
    assert first.num_tokens == 4
    assert first.input_seq_length == 3
    assert first.max_past_kv_length == 105

    q.fill_(7)
    metadata.kv_lens_cuda_runtime.fill_(106)
    metadata.prompt_lens_cuda_runtime.fill_(2)
    metadata.kv_lens_runtime.fill_(107)
    metadata.prompt_lens_cpu_runtime.fill_(2)
    assert first.attention_input is not None
    assert torch.count_nonzero(first.attention_input != 7) == 0
    torch.testing.assert_close(first.sequence_lengths, torch.tensor([106], dtype=torch.int32))
    torch.testing.assert_close(first.context_lengths, torch.tensor([2], dtype=torch.int32))
    assert first.input_seq_length == 3
    assert first.max_past_kv_length == 105

    fmha.forward(q, None, None, metadata, forward_args)
    second = captured[-1]
    assert second.input_seq_length == 2
    assert second.max_past_kv_length == 107


def test_b1_forward_fast_path_rereads_views_after_prepare(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs()
    replacement_kv_cuda = torch.tensor([109], dtype=torch.int32)
    replacement_kv_cpu = torch.tensor([109], dtype=torch.int32)
    replacement_prompt_cuda = torch.tensor([3], dtype=torch.int32)
    replacement_prompt_cpu = torch.tensor([3], dtype=torch.int32)
    scalar_reads: list[str] = []

    class _ScalarReadTensor(torch.Tensor):
        label: str

        @staticmethod
        def __new__(cls, tensor: torch.Tensor, label: str) -> "_ScalarReadTensor":
            result = torch.Tensor._make_subclass(cls, tensor, tensor.requires_grad)
            result.label = label
            return result

        def __getitem__(self, index: object) -> "_ScalarReadTensor":
            result = super().__getitem__(index)
            result.label = self.label
            return result

        def __int__(self) -> int:
            scalar_reads.append(self.label)
            return super().__int__()

    replacement_kv_cpu = _ScalarReadTensor(replacement_kv_cpu, "kv")
    replacement_prompt_cpu = _ScalarReadTensor(replacement_prompt_cpu, "prompt")

    def prepare(*args: object, **kwargs: object) -> None:
        metadata.kv_lens_cuda_runtime = replacement_kv_cuda
        metadata.kv_lens_runtime = replacement_kv_cpu
        metadata.prompt_lens_cuda_runtime = replacement_prompt_cuda
        metadata.prompt_lens_cpu_runtime = replacement_prompt_cpu

    run_context = Mock()
    monkeypatch.setattr(fmha, "get_fp8_context_fmha", Mock(return_value=False))
    monkeypatch.setattr(fmha, "prepare_workspace", prepare)
    monkeypatch.setattr(fmha, "_get_total_num_blocks", Mock(return_value=96))
    monkeypatch.setattr(fmha, "run_context", run_context)

    fmha.forward(q, None, None, metadata, forward_args)

    params = run_context.call_args.args[0]
    assert params.sequence_lengths is replacement_kv_cuda
    assert params.context_lengths is replacement_prompt_cuda
    assert params.input_seq_length == 3
    assert params.max_past_kv_length == 109
    assert scalar_reads == ["prompt", "kv"]


@pytest.mark.parametrize(
    "case",
    [
        "multiple-contexts",
        "has-generation",
        "generation-only",
        "mla",
        "zero-context-tokens",
        "q-row-mismatch",
        "missing-runtime-length",
    ],
)
def test_b1_forward_fast_path_falls_back_for_other_requests(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs()
    if case == "multiple-contexts":
        metadata.num_contexts = 2
    elif case == "has-generation":
        metadata.num_generations = 1
    elif case == "generation-only":
        forward_args.attention_input_type = AttentionInputType.generation_only
    elif case == "mla":
        fmha.attn.is_mla_enable = True
    elif case == "zero-context-tokens":
        metadata.num_ctx_tokens = 0
    elif case == "q-row-mismatch":
        metadata.num_ctx_tokens = q.shape[0] + 1
    else:
        del metadata.prompt_lens_cpu_runtime
    generic_forward = Mock()
    monkeypatch.setattr(PhasedFmha, "forward", generic_forward)

    fmha.forward(q, None, None, metadata, forward_args)

    generic_forward.assert_called_once_with(q, None, None, metadata, forward_args)


@pytest.mark.parametrize(
    "field_name",
    [
        "kv_lens_cuda_runtime",
        "kv_lens_runtime",
        "prompt_lens_cuda_runtime",
        "prompt_lens_cpu_runtime",
    ],
)
@pytest.mark.parametrize("invalid_shape", ["non-1d", "empty", "oversized"])
def test_b1_forward_fast_path_falls_back_for_invalid_runtime_length_views(
    monkeypatch: pytest.MonkeyPatch,
    field_name: str,
    invalid_shape: str,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs()
    if invalid_shape == "non-1d":
        replacement = torch.ones((1, 1), dtype=torch.int32)
    elif invalid_shape == "empty":
        replacement = torch.empty(0, dtype=torch.int32)
    else:
        replacement = torch.ones(2, dtype=torch.int32)
    setattr(metadata, field_name, replacement)
    generic_forward = Mock()
    monkeypatch.setattr(PhasedFmha, "forward", generic_forward)

    fmha.forward(q, None, None, metadata, forward_args)

    generic_forward.assert_called_once_with(q, None, None, metadata, forward_args)


def test_b1_forward_fast_path_propagates_prepare_failure_before_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs()
    events: list[str] = []

    def prepare(*args: object, **kwargs: object) -> None:
        events.append("prepare")
        raise RuntimeError("workspace capture guard")

    def run_context(params: FmhaParams) -> None:
        events.append("run")

    monkeypatch.setattr(fmha, "get_fp8_context_fmha", Mock(return_value=False))
    monkeypatch.setattr(fmha, "prepare_workspace", prepare)
    monkeypatch.setattr(fmha, "run_context", run_context)

    with pytest.raises(RuntimeError, match="workspace capture guard"):
        fmha.forward(q, None, None, metadata, forward_args)

    assert events == ["prepare"]


@pytest.mark.parametrize(
    ("case", "message"),
    [
        ("missing-output", "requires output"),
        ("missing-page-table", "requires paged KV cache"),
    ],
)
@pytest.mark.parametrize("phase", ["b1-context", "generation-fallback"])
def test_b1_forward_preserves_required_input_checks(
    monkeypatch: pytest.MonkeyPatch,
    case: str,
    message: str,
    phase: str,
) -> None:
    fmha, q, metadata, forward_args = _make_b1_forward_inputs()
    if phase == "generation-fallback":
        metadata.num_contexts = 0
        metadata.num_generations = 1
        forward_args.attention_input_type = AttentionInputType.generation_only
    if case == "missing-output":
        forward_args.output = None
    else:
        metadata.kv_cache_block_offsets = None
    generic_calls: list[tuple[object, ...]] = []
    generic_forward = PhasedFmha.forward

    def record_generic_forward(self: PhasedFmha, *args: object) -> None:
        generic_calls.append(args)
        generic_forward(self, *args)

    monkeypatch.setattr(PhasedFmha, "forward", record_generic_forward)

    with pytest.raises(RuntimeError, match=message):
        fmha.forward(q, None, None, metadata, forward_args)

    assert len(generic_calls) == (1 if phase == "generation-fallback" else 0)


@pytest.mark.parametrize(
    "case",
    [
        {
            "attention_input_type": AttentionInputType.context_only,
            "head_dim": 128,
        },
        {
            "attention_input_type": AttentionInputType.mixed,
            "head_dim": 256,
            "dtype": torch.float16,
        },
        {
            "attention_input_type": AttentionInputType.generation_only,
            "head_dim": 64,
            "dtype": torch.float16,
        },
        {
            "attention_input_type": AttentionInputType.generation_only,
            "head_dim": 576,
            "is_mla": True,
            "num_heads": 128,
        },
        {
            "attention_input_type": AttentionInputType.context_only,
            "num_kv_cache_pools": 2,
            "use_kv_cache_v2": True,
        },
        {
            "attention_input_type": AttentionInputType.context_only,
            "num_heads": 64,
            "num_kv_heads": 1,
        },
    ],
    ids=[
        "context",
        "mixed",
        "generation",
        "mla-generation",
        "v2-multi-pool",
        "context-gqa-ratio-over-32",
    ],
)
def test_supported_matrix(case: dict) -> None:
    supported, reason = _support_result(**case)

    assert supported, reason


@pytest.mark.parametrize("phase", [FmhaPhase.CONTEXT, FmhaPhase.GENERATION])
def test_phase_support_check_preserves_whole_request_semantics(phase: FmhaPhase) -> None:
    supported, reason = _support_result(
        attention_input_type=AttentionInputType.mixed,
        head_dim=64,
        phase=phase,
    )

    assert not supported
    assert "context head dimension" in reason


def test_is_supported_accepts_and_forwards_phase_keyword(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    support_check = Mock(return_value=(True, ""))
    monkeypatch.setattr(fmha, "_is_supported_with_reason", support_check)
    q = Mock(spec=torch.Tensor)
    metadata = SimpleNamespace()
    forward_args = AttentionForwardArgs()

    assert fmha.is_supported(
        q,
        None,
        None,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )
    support_check.assert_called_once_with(
        q,
        None,
        None,
        attn,
        metadata,
        forward_args,
        phase=FmhaPhase.GENERATION,
    )


@pytest.mark.parametrize(
    ("case", "expected_reason"),
    [
        (
            {"attention_input_type": AttentionInputType.context_only, "head_dim": 64},
            "context head dimension",
        ),
        (
            {"attention_input_type": AttentionInputType.generation_only, "head_dim": 96},
            "decode head dimension",
        ),
        (
            {"attention_input_type": AttentionInputType.context_only, "tokens_per_block": 8},
            "page size",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_paged_cache": False,
            },
            "paged KV-cache block offsets",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "is_fused_qkv": False,
                "has_separate_kv": True,
            },
            "only fused QKV",
        ),
        (
            {"attention_input_type": AttentionInputType.context_only, "is_cross": True},
            "cross attention",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_sparse_attention": True,
            },
            "sparse attention",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "use_spec_decoding": True,
            },
            "speculative decoding",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "is_spec_dec_tree": True,
                "use_spec_decoding": True,
            },
            "speculative decoding",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_attention_sinks": True,
            },
            "attention sinks",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "has_relative_attention_bias": True,
            },
            "relative attention bias",
        ),
        (
            {"attention_input_type": AttentionInputType.generation_only, "beam_width": 2},
            "beam search",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "position_embedding_type": 4,
            },
            "position embedding type",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "kv_dtype": DataType.HALF,
            },
            "query and KV-cache dtypes",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "output_dtype": torch.float16,
            },
            "output dtype must match",
        ),
        (
            {"attention_input_type": AttentionInputType.context_only, "has_output": False},
            "output tensor",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "num_heads": 7,
                "num_kv_heads": 2,
            },
            "divisible",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "num_heads": 64,
                "num_kv_heads": 1,
            },
            "GQA ratio",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "dtype": torch.float8_e4m3fn,
                "kv_dtype": DataType.FP8,
            },
            "query dtype",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "head_dim": 576,
                "is_mla": True,
            },
            "generation-only",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 576,
                "is_mla": True,
                "kv_lora_rank": 256,
            },
            "kv_lora_rank=512",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 576,
                "is_mla": True,
                "qk_rope_head_dim": 32,
            },
            "qk_rope_head_dim=64",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 576,
                "is_mla": True,
                "num_heads": 129,
            },
            "at most 128 local query heads",
        ),
        (
            {
                "attention_input_type": AttentionInputType.generation_only,
                "head_dim": 640,
                "is_mla": True,
            },
            "latent plus RoPE dimensions",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "attention_window_size": 64,
            },
            "cyclic TRT-LLM page tables",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "attention_window_size": 1,
            },
            "cyclic TRT-LLM page tables",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "attention_chunk_size": 64,
            },
            "chunked context attention",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "kv_layout": "NHD",
            },
            "HND KV-cache layout",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "num_kv_cache_pools": 2,
            },
            "V1 with multiple memory pools",
        ),
        (
            {
                "attention_input_type": AttentionInputType.context_only,
                "use_kv_cache_v2": True,
                "enable_swa_scratch_reuse": True,
            },
            "V2 SWA scratch reuse",
        ),
    ],
    ids=[
        "context-head-dim",
        "generation-head-dim",
        "page-size",
        "no-paged-cache",
        "separate-qkv",
        "cross",
        "sparse",
        "spec-decode",
        "tree-mask",
        "sinks",
        "relative-bias",
        "beam-search",
        "alibi",
        "kv-dtype-mismatch",
        "output-dtype-mismatch",
        "missing-output",
        "heads-not-divisible",
        "generation-head-ratio",
        "fp8",
        "mla-context",
        "mla-kv-rank",
        "mla-rope-dim",
        "mla-too-many-heads",
        "mla-head-dim",
        "sliding-window",
        "one-token-window",
        "chunked-context",
        "nhd-cache",
        "v1-multi-pool",
        "v2-swa-scratch-reuse",
    ],
)
def test_unsupported_matrix_falls_through(case: dict, expected_reason: str) -> None:
    supported, reason = _support_result(**case)

    assert not supported
    assert expected_reason in reason


@pytest.mark.parametrize(
    ("sm", "cutlass_version", "compiler_version", "expected"),
    [
        (100, "4.7.0", "13.3", True),
        (103, "4.7.0", "13.3", True),
        (100, "4.7.0", "13.4", True),
        (100, "4.7.0", "13.2", False),
        (120, "4.7.0", "13.3", False),
        (100, "4.6.2", "13.3", False),
    ],
)
def test_static_availability_gate(
    monkeypatch: pytest.MonkeyPatch,
    sm: int,
    cutlass_version: str,
    compiler_version: str,
    expected: bool,
) -> None:
    target_version = Mock(
        side_effect=lambda *, min_version: Version(compiler_version) >= Version(min_version)
    )
    cutlass = SimpleNamespace(target_version=target_version)

    def import_cutlass_module(module_name: str) -> object:
        if module_name == "cutlass":
            return cutlass
        assert module_name == "cutlass.experimental.task_scheduling"
        return object()

    monkeypatch.setattr(prims_ts_module, "get_sm_version", lambda: sm)
    monkeypatch.setattr(prims_ts_module, "version", lambda _: cutlass_version)
    monkeypatch.setattr(prims_ts_module, "import_module", import_cutlass_module)
    monkeypatch.setattr(PrimsTSFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: []))

    assert PrimsTSFmha.is_available(_Attention()) is expected
    if sm in (100, 103) and Version(cutlass_version) >= Version("4.7.0"):
        target_version.assert_called_once_with(min_version="13.3")
    else:
        target_version.assert_not_called()


def test_static_availability_gate_fails_closed_when_compiler_query_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cutlass = SimpleNamespace(target_version=Mock(side_effect=RuntimeError("query failed")))
    monkeypatch.setattr(prims_ts_module, "get_sm_version", lambda: 100)
    monkeypatch.setattr(prims_ts_module, "version", lambda _: "4.7.0")
    monkeypatch.setattr(
        prims_ts_module,
        "import_module",
        lambda module_name: cutlass if module_name == "cutlass" else object(),
    )
    monkeypatch.setattr(PrimsTSFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: []))

    assert not PrimsTSFmha.is_available(_Attention())
    cutlass.target_version.assert_called_once_with(min_version="13.3")


def test_unsupported_cutlass_compiler_excludes_prims_ts_from_fmha_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target_version = Mock(
        side_effect=lambda *, min_version: Version("13.2") >= Version(min_version)
    )
    cutlass = SimpleNamespace(target_version=target_version)
    monkeypatch.setenv("TLLM_FMHA_LIBS", "prims_ts,fallback")
    monkeypatch.setattr(prims_ts_module, "get_sm_version", lambda: 100)
    monkeypatch.setattr(prims_ts_module, "version", lambda _: "4.7.0")
    monkeypatch.setattr(
        prims_ts_module,
        "import_module",
        lambda module_name: cutlass if module_name == "cutlass" else object(),
    )
    monkeypatch.setattr(PrimsTSFmha, "_missing_fused_nanobind_ops", staticmethod(lambda: []))

    available_classes = [
        fmha_cls
        for fmha_cls in get_enabled_fmha_lib_classes()
        if fmha_cls.is_available(_Attention())
    ]

    assert available_classes == [FallbackFmha]
    target_version.assert_called_once_with(min_version="13.3")


def test_v2_total_page_bound_is_not_expanded() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    get_page_index_upper_bound = Mock(return_value=4096)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound),
            blocks_in_primary_pool=4096,
            num_local_layers=24,
        )
    )

    assert fmha._get_total_num_blocks(metadata) == 4096
    assert get_page_index_upper_bound.call_args.args[0] == 0


@pytest.mark.parametrize("is_mla", [False, True], ids=["standard", "mla"])
def test_v1_total_page_bound_excludes_slots_before_selected_layer(is_mla: bool) -> None:
    attn = _Attention(is_mla=is_mla)
    attn.local_layer_idx = 3
    fmha = PrimsTSFmha(attn)
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(
                get_primary_pool_data=lambda _: torch.empty(64, dtype=torch.uint8)
            ),
            blocks_in_primary_pool=64,
            num_local_layers=4,
        ),
        host_kv_cache_pool_mapping=torch.tensor(
            [[0, 0], [0, 1], [0, 2], [0, 3]], dtype=torch.int32
        ),
    )

    kv_factor = 1 if is_mla else 2
    assert fmha._get_total_num_blocks(metadata) == 64 * 4 * kv_factor - 3 * kv_factor


def _make_v2_kv_binding_inputs(
    *,
    local_layer_idx: int = 0,
    pool_mapping: torch.Tensor | None = None,
    kv_offsets: torch.Tensor | None = None,
) -> tuple[PrimsTSFmha, SimpleNamespace]:
    if pool_mapping is None:
        pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
    if kv_offsets is None:
        kv_offsets = torch.tensor([64], dtype=torch.int32)
    attn = _Attention()
    attn.local_layer_idx = local_layer_idx
    fmha = PrimsTSFmha(attn)
    manager = _WeakManager(
        impl=SimpleNamespace(get_page_index_upper_bound=lambda *args: 128),
        num_pools=kv_offsets.shape[0],
        enable_swa_scratch_reuse=False,
        kv_offset=kv_offsets,
    )
    metadata = SimpleNamespace(
        _attention_owner=attn,
        kv_cache_manager=manager,
        host_kv_cache_pool_mapping=pool_mapping,
    )
    return fmha, metadata


def test_kv_page_offset_uses_v2_manager_displacement() -> None:
    fmha, metadata = _make_v2_kv_binding_inputs(
        pool_mapping=torch.tensor([[1, 0]], dtype=torch.int32),
        kv_offsets=torch.tensor([0, 128], dtype=torch.int32),
    )

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 128


def test_v2_kv_binding_hit_avoids_tensor_scalar_reads() -> None:
    fmha, metadata = _make_v2_kv_binding_inputs()
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert scalar_read.call_count == 2
    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert scalar_read.call_count == 2


def test_v2_kv_binding_is_shared_with_b1_support_key() -> None:
    fmha, q, metadata, forward_args = _make_b1_context_cache_inputs()
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert (
        fmha._get_b1_context_support_key(q, None, None, fmha.attn, metadata, forward_args)
        is not None
    )
    assert scalar_read.call_count == 2
    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert scalar_read.call_count == 2


@pytest.mark.parametrize("replacement", ["manager", "impl"])
def test_v2_kv_binding_rebinds_after_manager_state_replacement(
    replacement: str,
) -> None:
    fmha, metadata = _make_v2_kv_binding_inputs()
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read
    manager = metadata.kv_cache_manager

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    if replacement == "manager":
        metadata.kv_cache_manager = _WeakManager(
            impl=manager.impl,
            num_pools=manager.num_pools,
            enable_swa_scratch_reuse=False,
            kv_offset=manager.kv_offset,
        )
    else:
        manager.impl = SimpleNamespace(get_page_index_upper_bound=lambda *args: 256)

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert scalar_read.call_count == 4


@pytest.mark.parametrize("replacement", ["mapping", "offsets"])
def test_v2_kv_binding_rebinds_after_tensor_replacement(replacement: str) -> None:
    fmha, metadata = _make_v2_kv_binding_inputs()
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    if replacement == "mapping":
        metadata.host_kv_cache_pool_mapping = torch.tensor([[0, 1]], dtype=torch.int32)
        expected = 64
    else:
        metadata.kv_cache_manager.kv_offset = torch.tensor([96], dtype=torch.int32)
        expected = 96

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == expected
    assert scalar_read.call_count == 4


@pytest.mark.parametrize("mutation", ["mapping", "offsets"])
def test_v2_kv_binding_rebinds_after_inplace_tensor_mutation(mutation: str) -> None:
    fmha, metadata = _make_v2_kv_binding_inputs()
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    if mutation == "mapping":
        metadata.host_kv_cache_pool_mapping[0, 1] = 1
        expected = 64
    else:
        metadata.kv_cache_manager.kv_offset[0] = 96
        expected = 96

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == expected
    assert scalar_read.call_count == 4


def test_v2_kv_binding_rebinds_after_local_layer_change() -> None:
    fmha, metadata = _make_v2_kv_binding_inputs(
        pool_mapping=torch.tensor([[0, 0], [0, 1]], dtype=torch.int32),
    )
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    fmha.attn.local_layer_idx = 1
    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert scalar_read.call_count == 4


def test_v2_multipool_uses_original_per_pool_fallback() -> None:
    fmha, metadata = _make_v2_kv_binding_inputs(
        local_layer_idx=1,
        pool_mapping=torch.tensor([[0, 0], [1, 0]], dtype=torch.int32),
        kv_offsets=torch.tensor([64, 192], dtype=torch.int32),
    )

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 192
    assert fmha._v2_kv_page_offset_binding is None
    assert fmha._kv_page_offset_cache == {(id(metadata.kv_cache_manager), 1): 192}


def test_v2_inference_tensors_resolve_fresh_without_binding() -> None:
    with torch.inference_mode():
        pool_mapping = torch.tensor([[0, 0]], dtype=torch.int32)
        kv_offsets = torch.tensor([64], dtype=torch.int32)
    fmha, metadata = _make_v2_kv_binding_inputs(
        pool_mapping=pool_mapping,
        kv_offsets=kv_offsets,
    )
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert fmha._v2_kv_page_offset_binding is None
    assert scalar_read.call_count == 2

    with torch.inference_mode():
        kv_offsets[0] = 96
    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 96
    assert fmha._v2_kv_page_offset_binding is None
    assert scalar_read.call_count == 4


def test_b1_v2_inference_tensors_remain_supported_without_certificate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    fmha, q, metadata, forward_args = _make_b1_context_cache_inputs()
    with torch.inference_mode():
        metadata.host_kv_cache_pool_mapping = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
        metadata.kv_cache_manager.kv_offset = torch.tensor([64], dtype=torch.int32)
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha.is_supported(q, None, None, metadata, forward_args)
    assert metadata._prims_ts_b1_context_support_key is None
    assert fmha._v2_kv_page_offset_binding is None
    assert scalar_read.call_count == 4

    assert fmha.is_supported(q, None, None, metadata, forward_args)
    assert metadata._prims_ts_b1_context_support_key is None
    assert fmha._v2_kv_page_offset_binding is None
    assert scalar_read.call_count == 8


@pytest.mark.parametrize("invalid", ["pool", "offset"])
def test_b1_v2_binding_fails_closed_for_invalid_single_pool(invalid: str) -> None:
    fmha, q, metadata, forward_args = _make_b1_context_cache_inputs()
    if invalid == "pool":
        metadata.host_kv_cache_pool_mapping[0, 0] = 1
    else:
        metadata.kv_cache_manager.kv_offset[0] = 0

    assert (
        fmha._get_b1_context_support_key(q, None, None, fmha.attn, metadata, forward_args) is None
    )


def test_v2_kv_binding_does_not_retain_dead_manager_or_trust_impl_id() -> None:
    fmha, metadata = _make_v2_kv_binding_inputs()
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read
    manager = metadata.kv_cache_manager
    manager_impl = manager.impl
    manager_ref = weakref.ref(manager)

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    metadata.kv_cache_manager = None
    del manager
    gc.collect()
    assert manager_ref() is None

    metadata.kv_cache_manager = _WeakManager(
        impl=manager_impl,
        num_pools=1,
        enable_swa_scratch_reuse=False,
        kv_offset=fmha._v2_kv_page_offset_binding.kv_offsets,
    )
    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) == 64
    assert scalar_read.call_count == 4
    assert fmha._v2_kv_page_offset_binding.manager_identity.get() is metadata.kv_cache_manager


def test_v2_kv_binding_fails_closed_for_nonweak_manager() -> None:
    fmha, metadata = _make_v2_kv_binding_inputs()
    manager = metadata.kv_cache_manager
    metadata.kv_cache_manager = SimpleNamespace(**manager.__dict__)
    scalar_read = Mock(wraps=fmha._read_host_tensor_scalar)
    fmha._read_host_tensor_scalar = scalar_read

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 0) is None
    assert scalar_read.call_count == 0
    assert fmha._v2_kv_page_offset_binding is None


def test_kv_page_offset_is_inferred_from_v1_host_tables() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    host_offsets = torch.tensor(
        [[[[0, 1, 2], [64, 65, 66]], [[3, 4, 5], [67, 68, 69]]]],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        kv_cache_manager=SimpleNamespace(
            kv_offset=None,
            host_kv_cache_block_offsets=host_offsets,
        ),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
    )

    assert fmha._get_kv_page_offset(fmha.attn, metadata, 1) == 64
    assert fmha._kv_page_offset_cache == {(id(metadata.kv_cache_manager), 0): 64}


def test_context_metadata_stages_live_lengths_and_padded_pages() -> None:
    block_tables = torch.tensor(
        [
            [
                [10, 11, 12, 13, 14, 15, 16, 17],
                [110, 111, 112, 113, 114, 115, 116, 117],
            ],
            [
                [20, 21, 22, 23, 24, 25, 26, 27],
                [120, 121, 122, 123, 124, 125, 126, 127],
            ],
        ],
        dtype=torch.int32,
    )
    cu_kv_seqlens = torch.tensor([0, 33, 97], dtype=torch.int32)
    sequence_lengths = torch.tensor([33, 64], dtype=torch.int32)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        2,
        8,
        32,
        need_context=True,
    )

    logical_kv_indptr, seq_lens, dense_page_table = fmha._stage_context_metadata(
        block_tables,
        cu_kv_seqlens,
        sequence_lengths,
        batch_size=2,
        page_size=32,
        max_kv_len=64,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert logical_kv_indptr is cu_kv_seqlens
    assert seq_lens is sequence_lengths
    assert logical_kv_indptr.data_ptr() == cu_kv_seqlens.data_ptr()
    assert seq_lens.data_ptr() == sequence_lengths.data_ptr()
    torch.testing.assert_close(seq_lens, torch.tensor([33, 64], dtype=torch.int32))
    assert dense_page_table.shape == (2, 2, 4)
    assert dense_page_table.is_contiguous()
    torch.testing.assert_close(
        dense_page_table,
        torch.tensor(
            [
                [[10, 11, 11, 11], [10, 11, 11, 11]],
                [[20, 21, 21, 21], [20, 21, 21, 21]],
            ],
            dtype=torch.int32,
        ),
    )
    cu_kv_seqlens[1] = 34
    sequence_lengths[0] = 34
    torch.testing.assert_close(
        logical_kv_indptr,
        torch.tensor([0, 34, 97], dtype=torch.int32),
    )
    torch.testing.assert_close(
        seq_lens,
        torch.tensor([34, 64], dtype=torch.int32),
    )


def test_context_metadata_batched_mixed_inputs_keep_bounded_views() -> None:
    block_tables = torch.tensor(
        [
            [[10, 11, 12, 13], [110, 111, 112, 113]],
            [[20, 21, 22, 23], [120, 121, 122, 123]],
        ],
        dtype=torch.int32,
    )
    cu_kv_seqlens = torch.tensor([0, 33, 97, 777], dtype=torch.int32)
    sequence_lengths = torch.tensor([33, 64, 777], dtype=torch.int32)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        2,
        4,
        32,
        need_context=True,
    )

    logical_kv_indptr, seq_lens_kv, _ = fmha._stage_context_metadata(
        block_tables,
        cu_kv_seqlens,
        sequence_lengths,
        batch_size=2,
        page_size=32,
        max_kv_len=64,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert logical_kv_indptr is not cu_kv_seqlens
    assert seq_lens_kv is not sequence_lengths
    assert logical_kv_indptr.shape == (3,)
    assert seq_lens_kv.shape == (2,)
    assert logical_kv_indptr.data_ptr() == cu_kv_seqlens.data_ptr()
    assert seq_lens_kv.data_ptr() == sequence_lengths.data_ptr()
    torch.testing.assert_close(
        logical_kv_indptr,
        torch.tensor([0, 33, 97], dtype=torch.int32),
    )
    torch.testing.assert_close(
        seq_lens_kv,
        torch.tensor([33, 64], dtype=torch.int32),
    )
    cu_kv_seqlens[1] = 34
    sequence_lengths[0] = 34
    assert logical_kv_indptr[1] == 34
    assert seq_lens_kv[0] == 34


def test_context_metadata_non_1d_inputs_preserve_slice_behavior() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    cu_kv_seqlens = torch.tensor([[0, 128]], dtype=torch.int32)
    sequence_lengths = torch.tensor([[128]], dtype=torch.int32)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        1,
        4,
        32,
        need_context=True,
    )

    logical_kv_indptr, seq_lens_kv, _ = fmha._stage_context_metadata(
        block_tables,
        cu_kv_seqlens,
        sequence_lengths,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert logical_kv_indptr is not cu_kv_seqlens
    assert seq_lens_kv is not sequence_lengths
    assert logical_kv_indptr.shape == (1, 2)
    assert seq_lens_kv.shape == (1, 1)
    assert logical_kv_indptr.data_ptr() == cu_kv_seqlens.data_ptr()
    assert seq_lens_kv.data_ptr() == sequence_lengths.data_ptr()
    cu_kv_seqlens[0, 1] = 127
    sequence_lengths[0, 0] = 127
    assert logical_kv_indptr[0, 1] == 127
    assert seq_lens_kv[0, 0] == 127


def _stage_b1_context_metadata(
    block_tables: torch.Tensor,
    *,
    head_dim: int = 128,
    max_kv_len: int = 128,
    window_left: int = -1,
) -> tuple[_Attention, PrimsTSFmha, torch.Tensor, torch.Tensor, torch.Tensor]:
    cu_kv_seqlens = torch.tensor([0, max_kv_len], dtype=torch.int32)
    sequence_lengths = torch.tensor([max_kv_len], dtype=torch.int32)
    attn = _Attention(head_dim=head_dim)
    fmha = PrimsTSFmha(attn)
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        1,
        block_tables.shape[-1],
        32,
        need_context=True,
    )

    logical_kv_indptr, seq_lens_kv, dense_page_table = fmha._stage_context_metadata(
        block_tables,
        cu_kv_seqlens,
        sequence_lengths,
        batch_size=1,
        page_size=32,
        max_kv_len=max_kv_len,
        window_left=window_left,
        cache_dense_page_alias=True,
    )
    return attn, fmha, logical_kv_indptr, seq_lens_kv, dense_page_table


def test_context_metadata_b1_d128_full_tile_reuses_cached_alias() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13, 14, 15, 16, 17], [110, 111, 112, 113, 114, 115, 116, 117]]],
        dtype=torch.int32,
    )
    source_k_row = block_tables[0, 0, :4]
    assert source_k_row.data_ptr() % 16 == 0

    attn, fmha, logical_kv_indptr, seq_lens_kv, dense_page_table = _stage_b1_context_metadata(
        block_tables
    )

    assert dense_page_table.shape == (1, 2, 4)
    assert dense_page_table.is_contiguous()
    assert dense_page_table.data_ptr() == source_k_row.data_ptr()
    assert dense_page_table.data_ptr() != fmha._context_page_indices_buffer.data_ptr()
    torch.testing.assert_close(dense_page_table[0, 0], source_k_row)
    cached_alias = fmha._dense_context_page_alias
    assert cached_alias is not None
    assert cached_alias.source is block_tables
    assert cached_alias.dense_page_idx_kv is dense_page_table
    assert cached_alias.source_key == (
        block_tables.device,
        block_tables.dtype,
        block_tables.data_ptr(),
        tuple(block_tables.shape),
        tuple(block_tables.stride()),
        block_tables.storage_offset(),
        block_tables.untyped_storage().nbytes() // block_tables.element_size(),
        32,
        4,
        1,
        128,
        -1,
        4,
        128,
    )

    next_logical_kv_indptr, next_seq_lens_kv, next_dense_page_table = fmha._stage_context_metadata(
        block_tables,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert next_dense_page_table is dense_page_table
    assert fmha.attn is attn
    assert next_logical_kv_indptr is logical_kv_indptr
    assert next_seq_lens_kv is seq_lens_kv
    source_k_row.add_(10)
    torch.testing.assert_close(
        next_dense_page_table[0, 0],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32),
    )


def test_context_metadata_b1_cached_alias_misses_source_replacement() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    attn, fmha, logical_kv_indptr, seq_lens_kv, first_alias = _stage_b1_context_metadata(
        block_tables
    )
    first_key = fmha._dense_context_page_alias.source_key
    replacement = block_tables.clone()

    _, _, replacement_alias = fmha._stage_context_metadata(
        replacement,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert replacement_alias is not first_alias
    assert fmha.attn is attn
    assert fmha._dense_context_page_alias.source is replacement
    assert fmha._dense_context_page_alias.source_key != first_key


def test_context_metadata_b1_cached_alias_reuses_fresh_native_views() -> None:
    base_block_tables = torch.tensor(
        [[[[10, 11, 12, 13], [110, 111, 112, 113]]]],
        dtype=torch.int32,
    )
    first_wrapper = base_block_tables.select(0, 0).narrow(0, 0, 1)
    attn, fmha, logical_kv_indptr, seq_lens_kv, first_alias = _stage_b1_context_metadata(
        first_wrapper
    )
    next_wrapper = base_block_tables.select(0, 0).narrow(0, 0, 1)
    assert next_wrapper is not first_wrapper

    _, _, next_alias = fmha._stage_context_metadata(
        next_wrapper,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert fmha.attn is attn
    assert next_alias is first_alias
    assert fmha._dense_context_page_alias.source is first_wrapper
    base_block_tables[0, 0, 0, :4].add_(10)
    torch.testing.assert_close(
        next_alias[0, 0],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32),
    )


def test_context_metadata_b1_cached_alias_misses_layout_change() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    attn, fmha, logical_kv_indptr, seq_lens_kv, first_alias = _stage_b1_context_metadata(
        block_tables
    )
    first_key = fmha._dense_context_page_alias.source_key
    block_tables.as_strided_((1, 2, 4), (16, 4, 1))
    assert block_tables.is_contiguous()

    _, _, restrided_alias = fmha._stage_context_metadata(
        block_tables,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert restrided_alias is not first_alias
    assert fmha.attn is attn
    assert fmha._dense_context_page_alias.source is block_tables
    assert fmha._dense_context_page_alias.source_key != first_key


def test_context_metadata_b1_cached_alias_misses_storage_geometry_change() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    attn, fmha, logical_kv_indptr, seq_lens_kv, first_alias = _stage_b1_context_metadata(
        block_tables
    )
    first_key = fmha._dense_context_page_alias.source_key
    replacement_storage = torch.arange(16, dtype=torch.int32)
    block_tables.set_(replacement_storage.untyped_storage(), 4, (1, 2, 4), (8, 4, 1))
    assert block_tables.data_ptr() % 16 == 0

    _, _, rebound_alias = fmha._stage_context_metadata(
        block_tables,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert rebound_alias is not first_alias
    assert fmha.attn is attn
    assert fmha._dense_context_page_alias.source is block_tables
    assert fmha._dense_context_page_alias.source_key != first_key
    torch.testing.assert_close(
        rebound_alias[0, 0],
        torch.tensor([4, 5, 6, 7], dtype=torch.int32),
    )


def test_context_metadata_b1_cached_alias_retains_source() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    source_ref = weakref.ref(block_tables)
    _, fmha, _, _, dense_page_table = _stage_b1_context_metadata(block_tables)

    del block_tables
    gc.collect()

    assert source_ref() is fmha._dense_context_page_alias.source
    torch.testing.assert_close(
        dense_page_table[0, 0],
        torch.tensor([10, 11, 12, 13], dtype=torch.int32),
    )

    del fmha, dense_page_table
    gc.collect()
    assert source_ref() is None


def test_context_metadata_b1_estimation_alias_is_not_retained() -> None:
    normal_block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    attn, fmha, logical_kv_indptr, seq_lens_kv, normal_alias = _stage_b1_context_metadata(
        normal_block_tables
    )
    assert fmha._dense_context_page_alias is not None
    estimation_block_tables = torch.tensor(
        [[[20, 21, 22, 23], [120, 121, 122, 123]]],
        dtype=torch.int32,
    )
    estimation_ref = weakref.ref(estimation_block_tables)

    _, _, estimation_alias = fmha._stage_context_metadata(
        estimation_block_tables,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=False,
    )

    assert fmha.attn is attn
    assert fmha._dense_context_page_alias is None
    assert estimation_alias is not normal_alias
    del estimation_alias, estimation_block_tables
    gc.collect()
    assert estimation_ref() is None


def test_context_metadata_b1_cached_alias_survives_tail_fallback() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )
    attn, fmha, logical_kv_indptr, seq_lens_kv, cached_alias = _stage_b1_context_metadata(
        block_tables
    )
    block_tables[0, 0].add_(10)

    _, _, tail_alias = fmha._stage_context_metadata(
        block_tables,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=96,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert tail_alias is not cached_alias
    torch.testing.assert_close(
        tail_alias,
        torch.tensor([[[20, 21, 22, 22], [20, 21, 22, 22]]], dtype=torch.int32),
    )

    _, _, reused_alias = fmha._stage_context_metadata(
        block_tables,
        logical_kv_indptr,
        seq_lens_kv,
        batch_size=1,
        page_size=32,
        max_kv_len=128,
        window_left=-1,
        cache_dense_page_alias=True,
    )

    assert fmha.attn is attn
    assert reused_alias is cached_alias
    torch.testing.assert_close(
        reused_alias[0, 0],
        torch.tensor([20, 21, 22, 23], dtype=torch.int32),
    )


def test_context_metadata_b1_tail_falls_back_to_staged_copy() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )

    _, fmha, _, _, dense_page_table = _stage_b1_context_metadata(block_tables, max_kv_len=96)

    assert fmha._dense_context_page_alias is None
    assert dense_page_table.data_ptr() != block_tables[0, 0].data_ptr()
    expected = torch.tensor([[[10, 11, 12, 12], [10, 11, 12, 12]]], dtype=torch.int32)
    torch.testing.assert_close(dense_page_table, expected)


def test_context_metadata_b1_d256_falls_back_to_staged_copy() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )

    _, fmha, _, _, dense_page_table = _stage_b1_context_metadata(block_tables, head_dim=256)

    assert fmha._dense_context_page_alias is None
    assert dense_page_table.data_ptr() != block_tables[0, 0].data_ptr()
    expected = torch.tensor([[[10, 11, 12, 13], [10, 11, 12, 13]]], dtype=torch.int32)
    torch.testing.assert_close(dense_page_table, expected)


def test_context_metadata_b1_window_falls_back_to_staged_copy() -> None:
    block_tables = torch.tensor(
        [[[10, 11, 12, 13], [110, 111, 112, 113]]],
        dtype=torch.int32,
    )

    _, fmha, _, _, dense_page_table = _stage_b1_context_metadata(block_tables, window_left=64)

    assert fmha._dense_context_page_alias is None
    assert dense_page_table.data_ptr() != block_tables[0, 0].data_ptr()
    expected = torch.tensor([[[10, 11, 12, 13], [10, 11, 12, 13]]], dtype=torch.int32)
    torch.testing.assert_close(dense_page_table, expected)


def test_context_metadata_b1_noncompact_source_falls_back_to_staged_copy() -> None:
    storage = torch.tensor(
        [10, 0, 11, 0, 12, 0, 13, 0, 110, 0, 111, 0, 112, 0, 113],
        dtype=torch.int32,
    )
    block_tables = storage.as_strided((1, 2, 4), (15, 8, 2))
    assert not block_tables[0, 0].is_contiguous()

    _, fmha, _, _, dense_page_table = _stage_b1_context_metadata(block_tables)

    assert fmha._dense_context_page_alias is None
    assert dense_page_table.data_ptr() != block_tables[0, 0].data_ptr()
    expected = torch.tensor([[[10, 11, 12, 13], [10, 11, 12, 13]]], dtype=torch.int32)
    torch.testing.assert_close(dense_page_table, expected)


def test_context_metadata_b1_short_source_storage_falls_back_to_staged_copy() -> None:
    storage = torch.tensor([10, 11, 12, 13], dtype=torch.int32)
    block_tables = storage.as_strided((1, 2, 4), (0, 0, 1))
    assert block_tables[0, 0].is_contiguous()

    _, fmha, _, _, dense_page_table = _stage_b1_context_metadata(block_tables)

    assert fmha._dense_context_page_alias is None
    assert dense_page_table.data_ptr() != block_tables[0, 0].data_ptr()
    expected = torch.tensor([[[10, 11, 12, 13], [10, 11, 12, 13]]], dtype=torch.int32)
    torch.testing.assert_close(dense_page_table, expected)


def test_fixed_stride_csr_without_bound_reuses_compact_storage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    block_tables = torch.tensor(
        [
            [[10, 11, 12], [110, 111, 112]],
            [[20, 21, 22], [120, 121, 122]],
        ],
        dtype=torch.int32,
    )

    indptr, indices = fmha._make_fixed_stride_csr(block_tables, 2, 32)
    first_storage = indices.data_ptr()
    torch.testing.assert_close(indptr, torch.tensor([0, 3, 6], dtype=torch.int32))
    torch.testing.assert_close(
        indices,
        torch.tensor([10, 11, 12, 20, 21, 22], dtype=torch.int32),
    )

    block_tables[:, 0].add_(100)
    _, updated_indices = fmha._make_fixed_stride_csr(block_tables, 2, 32)

    assert updated_indices.data_ptr() == first_storage
    torch.testing.assert_close(
        updated_indices,
        torch.tensor([110, 111, 112, 120, 121, 122], dtype=torch.int32),
    )


@pytest.mark.parametrize("max_kv_len", [None, 96], ids=["mla", "decode"])
def test_fixed_stride_csr_b1_reuses_aligned_k_row(max_kv_len: int | None) -> None:
    fmha = PrimsTSFmha(_Attention())
    block_tables = torch.tensor(
        [[[10, 11, 12], [110, 111, 112]]],
        dtype=torch.int32,
    )
    source_k_row = block_tables[0, 0]
    assert source_k_row.data_ptr() % 16 == 0

    indptr, indices = fmha._make_fixed_stride_csr(
        block_tables,
        1,
        32,
        max_kv_len=max_kv_len,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 3], dtype=torch.int32))
    assert indices.data_ptr() == source_k_row.data_ptr()
    source_k_row.add_(10)
    torch.testing.assert_close(indices, torch.tensor([20, 21, 22], dtype=torch.int32))


def test_fixed_stride_csr_with_safe_mla_bound_reuses_interleaved_tables() -> None:
    fmha = PrimsTSFmha(_Attention())
    block_tables = torch.arange(2 * 2 * 32, dtype=torch.int32).view(2, 2, 32)

    indptr, indices = fmha._make_fixed_stride_csr(
        block_tables,
        2,
        32,
        max_kv_len=128,
        allow_interleaved_tables=True,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 64, 128], dtype=torch.int32))
    assert indices.data_ptr() == block_tables.data_ptr()
    assert indices.numel() == block_tables.numel()
    torch.testing.assert_close(indices.view_as(block_tables), block_tables)


def test_fixed_stride_csr_bound_without_opt_in_keeps_compact_k_tables() -> None:
    fmha = PrimsTSFmha(_Attention())
    block_tables = torch.arange(2 * 2 * 32, dtype=torch.int32).view(2, 2, 32)

    indptr, indices = fmha._make_fixed_stride_csr(
        block_tables,
        2,
        32,
        max_kv_len=128,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 32, 64], dtype=torch.int32))
    assert indices.data_ptr() != block_tables.data_ptr()
    torch.testing.assert_close(indices.view(2, 32), block_tables[:, 0])


def test_fixed_stride_csr_with_small_capacity_falls_back_to_compact_storage() -> None:
    fmha = PrimsTSFmha(_Attention())
    block_tables = torch.arange(2 * 2 * 8, dtype=torch.int32).view(2, 2, 8)

    indptr, indices = fmha._make_fixed_stride_csr(
        block_tables,
        2,
        32,
        max_kv_len=256,
        allow_interleaved_tables=True,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 8, 16], dtype=torch.int32))
    assert indices.data_ptr() != block_tables.data_ptr()
    torch.testing.assert_close(indices.view(2, 8), block_tables[:, 0])


def test_fixed_stride_csr_with_misaligned_tables_falls_back_to_compact_storage() -> None:
    fmha = PrimsTSFmha(_Attention())
    base = torch.arange(1 + 2 * 2 * 32, dtype=torch.int32)
    block_tables = base[1:].view(2, 2, 32)
    assert block_tables.data_ptr() % 16 != 0

    indptr, indices = fmha._make_fixed_stride_csr(
        block_tables,
        2,
        32,
        max_kv_len=128,
        allow_interleaved_tables=True,
    )

    torch.testing.assert_close(indptr, torch.tensor([0, 32, 64], dtype=torch.int32))
    assert indices.data_ptr() != block_tables.data_ptr()
    assert indices.data_ptr() % 16 == 0
    torch.testing.assert_close(indices.view(2, 32), block_tables[:, 0])


def test_mla_aligned_sequence_lengths_use_source_storage() -> None:
    attn = _Attention(head_dim=576, is_mla=True)
    fmha = PrimsTSFmha(attn)
    sequence_lengths = torch.tensor([33, 64], dtype=torch.int32)
    assert sequence_lengths.data_ptr() % 16 == 0

    actual = fmha._get_mla_sequence_lengths(sequence_lengths, 2)

    assert actual.data_ptr() == sequence_lengths.data_ptr()


def test_context_live_unchecked_forwards_exact_compiled_abi() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts.context import BatchPrefillPagedTSWrapper

    wrapper = object.__new__(BatchPrefillPagedTSWrapper)
    wrapper._planned = True
    wrapper._live_metadata = True
    wrapper._scale_softmax_log2 = object()
    wrapper._output_scale = object()
    wrapper._compiled = Mock()
    q, k_cache, v_cache, out = (object() for _ in range(4))
    qo_indptr, logical_kv_indptr, dense_page_idx_kv, seq_lens_kv = (object() for _ in range(4))

    actual = wrapper._run_live_unchecked(
        q,
        k_cache,
        v_cache,
        out,
        qo_indptr,
        logical_kv_indptr,
        dense_page_idx_kv,
        seq_lens_kv,
    )

    assert actual is out
    wrapper._compiled.assert_called_once_with(
        q,
        k_cache,
        v_cache,
        out,
        wrapper._scale_softmax_log2,
        wrapper._output_scale,
        qo_indptr,
        logical_kv_indptr,
        dense_page_idx_kv,
        seq_lens_kv,
    )


@pytest.mark.parametrize(
    ("planned", "live_metadata", "expected_error"),
    [
        (False, False, "plan_live\\(\\) must be called"),
        (True, False, "requires a plan_live\\(\\) plan"),
    ],
    ids=["unplanned", "snapshot-plan"],
)
def test_context_live_unchecked_requires_live_plan(
    planned: bool,
    live_metadata: bool,
    expected_error: str,
) -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts.context import BatchPrefillPagedTSWrapper

    wrapper = object.__new__(BatchPrefillPagedTSWrapper)
    wrapper._planned = planned
    wrapper._live_metadata = live_metadata

    with pytest.raises(RuntimeError, match=expected_error):
        wrapper._run_live_unchecked(*(object() for _ in range(8)))


def test_context_wrapper_plans_once_and_reads_live_staged_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 120
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        2,
        4,
        32,
        need_context=True,
    )
    q_processed = torch.empty((3, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    kv_pool = torch.empty((12, attn.num_kv_heads, 32, attn.head_dim), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2, 3], [6, 7, 8, 9]],
            [[2, 3, 4, 5], [8, 9, 10, 11]],
        ],
        dtype=torch.int32,
    )
    cu_q_seqlens = torch.tensor([0, 1, 3], dtype=torch.int32)
    cu_kv_seqlens = torch.tensor([0, 33, 97], dtype=torch.int32)
    fmha_workspace = torch.empty(64, dtype=torch.uint8)
    context_preprocess = Mock(
        return_value=(
            q_processed,
            kv_pool,
            block_tables,
            None,
            1.0,
            1.0,
            fmha_workspace,
            cu_q_seqlens,
            cu_kv_seqlens,
            2,
            64,
            -1,
        )
    )
    context_postprocess = Mock()
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_context_preprocess",
        context_preprocess,
    )
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_context_postprocess",
        context_postprocess,
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_context_wrapper",
        wrapper_factory,
    )

    host_block_offsets = torch.tensor(
        [
            [
                [[0, 1, 2, 3], [6, 7, 8, 9]],
                [[2, 3, 4, 5], [8, 9, 10, 11]],
                [[4, 5, 0, 0], [10, 11, 0, 0]],
            ]
        ],
        dtype=torch.int32,
    )
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((3, 2, 4), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            kv_offset=None,
            host_kv_cache_block_offsets=host_block_offsets,
        ),
        kv_lens_runtime=torch.tensor([7, 33, 64], dtype=torch.int32),
        max_context_length=8,
        max_seq_len=128,
    )
    output = torch.empty((3, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.context_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(32, dtype=torch.uint8),
        qkv_input=torch.empty(
            (3, (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim),
            dtype=torch.bfloat16,
        ),
        context_buf=output,
        sequence_lengths=torch.tensor([33, 64], dtype=torch.int32),
        context_lengths=torch.tensor([1, 2], dtype=torch.int32),
        input_seq_length=2,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=3,
        seq_offset=1,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=24,
        batch_size=2,
    )

    fmha.run_context(params)

    wrapper_factory.assert_called_once_with(kv_layout="HND")
    wrapper.plan_live.assert_called_once()
    plan_args = wrapper.plan_live.call_args.args
    plan_kwargs = wrapper.plan_live.call_args.kwargs
    assert plan_args[0] is q_processed
    k_cache, v_cache = plan_args[1], plan_args[2]
    assert k_cache.shape == v_cache.shape == (6, attn.num_kv_heads, 32, attn.head_dim)
    assert v_cache.storage_offset() - k_cache.storage_offset() == 6 * math.prod(kv_pool.shape[1:])
    assert plan_kwargs == {
        "batch_size": 2,
        "max_seq_len_q": 8,
        "max_seq_len_k": 128,
        "max_num_pages_per_seq_kv": 4,
        "page_size": 32,
        "mask_type": "causal",
        "window_left": -1,
        "sm_scale": pytest.approx(1.0 / math.sqrt(attn.head_dim)),
        "output_scale": 1.0,
        "out_dtype": torch.bfloat16,
    }
    wrapper._run_live_unchecked.assert_called_once()
    run_args = wrapper._run_live_unchecked.call_args.args
    assert len(run_args) == 8
    first_metadata_ptrs = tuple(run_args[index].data_ptr() for index in (4, 5, 7, 6))
    assert all(
        run_arg is plan_arg for run_arg, plan_arg in zip(run_args[:3], plan_args, strict=True)
    )
    assert run_args[3] is output
    assert run_args[4] is cu_q_seqlens
    assert run_args[5].data_ptr() == cu_kv_seqlens.data_ptr()
    torch.testing.assert_close(
        run_args[7],
        torch.tensor([33, 64], dtype=torch.int32),
    )
    torch.testing.assert_close(
        run_args[6],
        torch.tensor(
            [
                [[0, 1, 1, 1], [0, 1, 1, 1]],
                [[2, 3, 3, 3], [2, 3, 3, 3]],
            ],
            dtype=torch.int32,
        ),
    )
    context_preprocess.assert_called_once()
    context_postprocess.assert_called_once()
    assert context_preprocess.call_args.kwargs["skip_workspace"] is True
    assert context_postprocess.call_args.kwargs["skip_workspace"] is True

    block_tables[:, 0].add_(1)
    cu_kv_seqlens.copy_(torch.tensor([0, 64, 97], dtype=torch.int32))
    params.sequence_lengths.copy_(torch.tensor([64, 33], dtype=torch.int32))
    fmha.run_context(params)

    wrapper_factory.assert_called_once_with(kv_layout="HND")
    wrapper.plan_live.assert_called_once()
    assert wrapper._run_live_unchecked.call_count == 2
    wrapper.run.assert_not_called()
    second_run_args = wrapper._run_live_unchecked.call_args.args
    assert tuple(second_run_args[index].data_ptr() for index in (4, 5, 7, 6)) == first_metadata_ptrs
    torch.testing.assert_close(
        second_run_args[7],
        torch.tensor([64, 33], dtype=torch.int32),
    )
    torch.testing.assert_close(
        second_run_args[6],
        torch.tensor(
            [
                [[1, 2, 2, 2], [1, 2, 2, 2]],
                [[3, 4, 4, 4], [3, 4, 4, 4]],
            ],
            dtype=torch.int32,
        ),
    )


@pytest.mark.parametrize("requires_control_reset", [False, True])
def test_generation_wrapper_plans_once_and_reads_live_native_csr(
    monkeypatch: pytest.MonkeyPatch,
    requires_control_reset: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 120
    fmha_workspace = torch.full((64,), 7, dtype=torch.uint8)
    q_processed = torch.empty((2, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    kv_pool = torch.empty((12, attn.num_kv_heads, 32, attn.head_dim), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2, 3], [6, 7, 8, 9]],
            [[2, 3, 4, 5], [8, 9, 10, 11]],
        ],
        dtype=torch.int32,
    )
    generation_preprocess = Mock(
        return_value=(
            q_processed,
            kv_pool,
            block_tables,
            None,
            1.0,
            1.0,
            fmha_workspace,
            None,
            1,
            64,
            15,
            False,
        )
    )
    wrapper = Mock()
    wrapper._workspace_layout = SimpleNamespace(
        total_bytes=64,
        split_kv_counter=SimpleNamespace(byte_offset=32),
    )
    wrapper._requires_control_reset = requires_control_reset
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "trtllm_gen_generation_preprocess",
        generation_preprocess,
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        Mock(return_value=64),
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_decode_wrapper",
        wrapper_factory,
    )

    get_page_index_upper_bound = Mock(return_value=12)
    metadata = SimpleNamespace(
        beam_width=1,
        max_seq_len=96,
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=_WeakManager(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound),
            num_pools=1,
            enable_swa_scratch_reuse=False,
            kv_offset=torch.tensor([6], dtype=torch.int32),
        ),
    )
    total_num_blocks = fmha._get_total_num_blocks(metadata)
    output = torch.empty((2, attn.num_heads, attn.head_dim), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    sequence_lengths = torch.tensor([0, 33, 64], dtype=torch.int32)[1:]
    assert sequence_lengths.data_ptr() % 16 != 0
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(32, dtype=torch.uint8),
        qkv_input=torch.empty(
            (2, (attn.num_heads + 2 * attn.num_kv_heads) * attn.head_dim),
            dtype=torch.bfloat16,
        ),
        context_buf=output,
        sequence_lengths=sequence_lengths,
        input_seq_length=1,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=2,
        seq_offset=1,
        tokens_per_block=32,
        kv_factor=2,
        total_num_blocks=total_num_blocks,
        batch_size=2,
        num_requests=2,
    )

    fmha.run_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    torch.testing.assert_close(plan_args[0], torch.tensor([0, 4, 8], dtype=torch.int32))
    torch.testing.assert_close(
        plan_args[1],
        torch.tensor([0, 1, 2, 3, 2, 3, 4, 5], dtype=torch.int32),
    )
    assert plan_args[1].data_ptr() == fmha._page_indices_buffer.data_ptr()
    assert plan_args[2] is None
    assert plan_args[3:] == (
        attn.num_heads,
        attn.num_kv_heads,
        attn.head_dim,
        32,
    )
    assert plan_kwargs["workspace_buffer"] is fmha_workspace
    assert {key: value for key, value in plan_kwargs.items() if key != "workspace_buffer"} == {
        "seq_len_q": 1,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "mask_type": "causal",
        "window_left": 15,
        "max_kv_len": 96,
        "live_metadata": True,
        "enable_pdl": fmha._enable_pdl,
    }
    assert fmha._decode_wrappers[2] is wrapper
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0].shape == (2, attn.num_heads, attn.head_dim)
    assert run_args[0].data_ptr() == q_processed.data_ptr()
    k_cache, v_cache = run_args[1]
    assert k_cache.shape == v_cache.shape == (6, attn.num_kv_heads, 32, attn.head_dim)
    assert v_cache.storage_offset() - k_cache.storage_offset() == 6 * math.prod(kv_pool.shape[1:])
    assert run_args[2].data_ptr() == params.sequence_lengths.data_ptr()
    assert run_kwargs["paged_kv_indptr"] is plan_args[0]
    assert run_kwargs["paged_kv_indices"] is plan_args[1]
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(attn.head_dim))
    assert run_kwargs["bmm2_scale"] == 1.0
    assert run_kwargs["out"].shape == (2, attn.num_heads, attn.head_dim)
    assert run_kwargs["out"].data_ptr() == output.data_ptr()
    torch.testing.assert_close(fmha_workspace[:32], torch.full((32,), 7, dtype=torch.uint8))
    if requires_control_reset:
        assert torch.count_nonzero(fmha_workspace[32:]) == 0
    else:
        torch.testing.assert_close(
            fmha_workspace[32:],
            torch.full((32,), 7, dtype=torch.uint8),
        )
    preprocess_args = generation_preprocess.call_args.args
    assert preprocess_args[15] == params.seq_offset
    assert preprocess_args[39] == total_num_blocks
    assert generation_preprocess.call_args.kwargs["skip_workspace"] is True
    get_page_index_upper_bound.assert_called_once()

    block_tables[:, 0].add_(20)
    sequence_lengths.add_(1)
    fmha_workspace.fill_(9)
    fmha.run_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    torch.testing.assert_close(
        plan_args[1],
        torch.tensor([20, 21, 22, 23, 22, 23, 24, 25], dtype=torch.int32),
    )
    torch.testing.assert_close(run_args[2], torch.tensor([34, 65], dtype=torch.int32))
    torch.testing.assert_close(fmha_workspace[:32], torch.full((32,), 9, dtype=torch.uint8))
    if requires_control_reset:
        assert torch.count_nonzero(fmha_workspace[32:]) == 0
    else:
        torch.testing.assert_close(
            fmha_workspace[32:],
            torch.full((32,), 9, dtype=torch.uint8),
        )


def test_decode_pdl_environment_is_snapshotted_and_threaded_to_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    get_env_enable_pdl = Mock(return_value=True)
    monkeypatch.setattr(prims_ts_module, "get_env_enable_pdl", get_env_enable_pdl)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrapper = Mock()
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_decode_wrapper",
        Mock(return_value=wrapper),
    )
    fmha = PrimsTSFmha(_Attention())
    get_env_enable_pdl.return_value = False
    paged_kv_indptr = torch.tensor([0, 2], dtype=torch.int32)
    paged_kv_indices = torch.tensor([0, 1], dtype=torch.int32)
    workspace = torch.empty(64, dtype=torch.uint8)

    first = fmha._get_or_plan_decode_wrapper(
        paged_kv_indptr,
        paged_kv_indices,
        workspace,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=32,
        seq_len_q=1,
        max_kv_len=64,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        mask_type="causal",
        window_left=-1,
    )
    second = fmha._get_or_plan_decode_wrapper(
        paged_kv_indptr,
        paged_kv_indices,
        workspace,
        num_qo_heads=8,
        num_kv_heads=2,
        head_dim=128,
        page_size=32,
        seq_len_q=1,
        max_kv_len=64,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        mask_type="causal",
        window_left=-1,
    )

    assert first is second is wrapper
    get_env_enable_pdl.assert_called_once_with()
    wrapper.plan.assert_called_once()
    assert wrapper.plan.call_args.kwargs["enable_pdl"] is True
    assert fmha._decode_wrappers[1] is wrapper


@pytest.mark.parametrize(
    "enable_pdl",
    [0, 1.0, "true", torch.tensor(True)],
    ids=["integer", "float", "string", "tensor"],
)
def test_decode_plan_requires_an_exact_bool_for_enable_pdl(enable_pdl: object) -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts.decode import BatchDecodePagedTSWrapper

    wrapper = BatchDecodePagedTSWrapper(kv_layout="HND")

    with pytest.raises(TypeError, match="enable_pdl must be a bool"):
        wrapper.plan(
            None,
            None,
            None,
            8,
            2,
            128,
            32,
            enable_pdl=enable_pdl,
        )


def test_decode_pdl_is_a_distinct_compiler_cache_key() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts import decode

    parameters = tuple(inspect.signature(decode._get_compiled_decode).parameters.values())
    assert parameters[-1].name == "enable_pdl"
    assert parameters[-1].default is False

    decode_path = Path(decode.__file__).resolve()
    compiler_source = " ".join(_function_source(decode_path, "_get_compiled_decode").split())
    wrapper_source = " ".join(_function_source(decode_path, "plan").split())
    raw_source = " ".join(
        _function_source(
            decode_path,
            "prims_ts_batch_decode_with_kv_cache",
        ).split()
    )
    assert "dataclasses.replace(spec.config, use_external_pdl=enable_pdl)" in compiler_source
    assert '("use_external_pdl", enable_pdl)' in compiler_source
    assert (
        "_get_compiled_decode( *semantic_key, kv_prefix_mode, kv_lengths_mode, enable_pdl )"
        in wrapper_source
    )
    assert '_get_compiled_decode( *semantic_key, "dynamic", "dynamic", False )' in raw_source


def test_decode_config_combines_external_and_reducer_pdl_without_mutation() -> None:
    from tensorrt_llm._torch.attention_backend.prims_ts.kernels.fmha_decode.fmha_decode_config import (
        FmhaDecodeConfig,
    )

    base = FmhaDecodeConfig()
    external = replace(base, use_external_pdl=True)
    reducer = replace(base, use_separate_reduction_kernel=True)

    assert not base.use_external_pdl
    assert not base.use_main_launch_pdl
    assert external.use_external_pdl
    assert external.use_main_launch_pdl
    assert reducer.use_main_launch_pdl


def test_decode_external_pdl_kernel_handoffs_precede_runtime_memory_reads() -> None:
    source_dir = (
        Path(prims_ts_module.__file__).resolve().parent.parent
        / "prims_ts"
        / "kernels"
        / "fmha_decode"
    )
    main_source = _function_source(source_dir / "fmha_decode_kernel.py", "decode_gen_kernel")
    main_launch_source = _function_source(
        source_dir / "fmha_decode_kernel.py",
        "fmha_decode_launch",
    )
    reducer_source = _function_source(
        source_dir / "reduction.py",
        "decode_gen_parallel_separate_reduction_kernel",
    )

    main_external = main_source.index("if cutlass.const_expr(cfg.use_external_pdl):")
    main_wait = main_source.index("GridDepAction.WAIT", main_external)
    main_launch = main_source.index("GridDepAction.LAUNCH_DEPENDENTS", main_wait)
    main_q_bounds = main_source.index("_q_seq_bounds")
    assert main_source.index("q_group_idx = q_group_cta_idx") < main_external
    assert main_external < main_wait < main_launch < main_q_bounds
    assert "not cfg.use_separate_reduction_kernel" in main_source[main_wait:main_launch]
    assert "use_pdl=cfg.use_main_launch_pdl" in main_launch_source

    reducer_wait = reducer_source.index("GridDepAction.WAIT")
    reducer_external = reducer_source.index(
        "if cutlass.const_expr(cfg.use_external_pdl):",
        reducer_wait,
    )
    reducer_launch = reducer_source.index(
        "GridDepAction.LAUNCH_DEPENDENTS",
        reducer_external,
    )
    reducer_schedule = reducer_source.index(
        "if cutlass.const_expr(cfg.use_compact_parallel_reduction):"
    )
    assert reducer_wait < reducer_external < reducer_launch < reducer_schedule


def test_decode_layer_adapters_bind_the_same_shared_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_decode_wrapper",
        wrapper_factory,
    )
    attentions = [_Attention(), _Attention()]
    layers = [PrimsTSFmha(attn) for attn in attentions]
    shared_workspace = torch.empty(64, dtype=torch.uint8)
    paged_kv_indptr = torch.tensor([0, 2, 4], dtype=torch.int32)
    paged_kv_indices = torch.tensor([0, 1, 2, 3], dtype=torch.int32)

    def get_wrapper(layer: PrimsTSFmha) -> object:
        return layer._get_or_plan_decode_wrapper(
            paged_kv_indptr,
            paged_kv_indices,
            shared_workspace,
            batch_size=2,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            seq_len_q=1,
            max_kv_len=64,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            mask_type="causal",
            window_left=-1,
        )

    first_results = [get_wrapper(layer) for layer in layers]
    second_results = [get_wrapper(layer) for layer in layers]

    assert first_results == second_results == wrappers
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
        assert wrapper.plan.call_args.kwargs["workspace_buffer"] is shared_workspace


def test_context_wrapper_cache_plans_each_batch_once_and_reuses_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_context_wrapper",
        wrapper_factory,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    k_cache = torch.empty((8, 2, 32, 128), dtype=torch.bfloat16)
    v_cache = torch.empty_like(k_cache)

    def get_wrapper(batch_size: int) -> object:
        q = torch.empty((batch_size, 8, 128), dtype=torch.bfloat16)
        return fmha._get_or_plan_context_wrapper(
            q,
            k_cache,
            v_cache,
            batch_size=batch_size,
            max_seq_len_q=128,
            max_seq_len_k=256,
            max_num_pages_per_seq_kv=8,
            page_size=32,
            mask_type="causal",
            window_left=-1,
            sm_scale=1.0 / math.sqrt(128),
            output_dtype=torch.bfloat16,
        )

    first_a = get_wrapper(1)
    profile_b = get_wrapper(2)
    second_a = get_wrapper(1)

    assert first_a is second_a is wrappers[0]
    assert profile_b is wrappers[1]
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan_live.assert_called_once()
    assert set(fmha._context_wrappers) == {1, 2}


def test_decode_wrapper_cache_plans_each_batch_once_and_reuses_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_decode_wrapper",
        wrapper_factory,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    workspace = torch.empty(64, dtype=torch.uint8)

    def get_wrapper(batch_size: int) -> object:
        return fmha._get_or_plan_decode_wrapper(
            torch.arange(batch_size + 1, dtype=torch.int32).mul_(2),
            torch.arange(batch_size * 2, dtype=torch.int32),
            workspace,
            batch_size=batch_size,
            num_qo_heads=8,
            num_kv_heads=2,
            head_dim=128,
            page_size=32,
            seq_len_q=1,
            max_kv_len=256,
            q_dtype=torch.bfloat16,
            kv_dtype=torch.bfloat16,
            output_dtype=torch.bfloat16,
            mask_type="causal",
            window_left=-1,
        )

    first_a = get_wrapper(1)
    profile_b = get_wrapper(2)
    second_a = get_wrapper(1)

    assert first_a is second_a is wrappers[0]
    assert profile_b is wrappers[1]
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
    assert set(fmha._decode_wrappers) == {1, 2}


def test_workspace_allocation_change_invalidates_only_workspace_bound_wrappers() -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    first_workspace = torch.empty(32, dtype=torch.uint8)
    fmha._update_workspace_allocation(first_workspace)
    fmha._context_wrappers[1] = Mock()
    fmha._decode_wrappers[1] = Mock()
    fmha._mla_decode_wrappers[1] = Mock()

    second_workspace = torch.empty(64, dtype=torch.uint8)
    fmha._update_workspace_allocation(second_workspace)

    assert set(fmha._context_wrappers) == {1}
    assert fmha._decode_wrappers == {}
    assert fmha._mla_decode_wrappers == {}


def _get_test_mla_wrapper(
    fmha: PrimsTSFmha,
    block_tables: torch.Tensor,
    seq_lens: torch.Tensor,
    workspace_buffer: torch.Tensor,
    *,
    mask_type: str = "causal",
) -> object:
    return fmha._get_or_plan_mla_decode_wrapper(
        block_tables,
        seq_lens,
        workspace_buffer,
        batch_size=int(block_tables.shape[0]),
        num_heads=4,
        kv_lora_rank=512,
        qk_rope_head_dim=64,
        page_size=32,
        max_seq_len_q=1,
        max_kv_len=96,
        q_dtype=torch.bfloat16,
        kv_dtype=torch.bfloat16,
        output_dtype=torch.bfloat16,
        mask_type=mask_type,
    )


def test_mla_eager_wrapper_plans_once_and_reads_live_staged_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    kv_cache = torch.empty((20, 1, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2], [10, 11, 12]],
            [[3, 4, 5], [13, 14, 15]],
        ],
        dtype=torch.int32,
    )
    build_metadata = Mock(return_value=(kv_cache, block_tables, None))
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "build_trtllm_gen_kv_cache_metadata",
        build_metadata,
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_mla_workspace_size",
        Mock(return_value=64),
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_mla_decode_wrapper",
        wrapper_factory,
    )

    metadata = SimpleNamespace(
        is_cuda_graph=False,
        beam_width=1,
        max_seq_len=80,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
    )
    output = torch.empty((2, attn.num_heads, 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    sequence_lengths = torch.tensor([33, 64], dtype=torch.int32)
    assert sequence_lengths.data_ptr() % 16 == 0
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(64, dtype=torch.uint8),
        qkv_input=torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16),
        context_buf=output,
        sequence_lengths=sequence_lengths,
        input_seq_length=1,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=2,
        seq_offset=2,
        tokens_per_block=32,
        kv_factor=1,
        total_num_blocks=20,
        batch_size=2,
        num_requests=2,
    )

    fmha.run_mla_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    assert plan_args[0].data_ptr() == fmha._page_indices_buffer.data_ptr()
    torch.testing.assert_close(
        plan_args[0],
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
    )
    assert plan_args[1].data_ptr() == sequence_lengths.data_ptr()
    torch.testing.assert_close(plan_args[1], sequence_lengths)
    assert plan_args[2:] == (attn.num_heads, 512, 64, 32)
    workspace_buffer = plan_kwargs["workspace_buffer"]
    assert workspace_buffer.data_ptr() == params.workspace.data_ptr()
    assert workspace_buffer.numel() == 64
    assert {key: value for key, value in plan_kwargs.items() if key != "workspace_buffer"} == {
        "max_seq_len_q": 1,
        "q_data_type": torch.bfloat16,
        "kv_data_type": torch.bfloat16,
        "o_data_type": torch.bfloat16,
        "mask_type": "causal",
        "max_kv_len": 80,
        "live_metadata": True,
    }
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0].shape == (2, 1, attn.num_heads, 576)
    assert run_args[0].data_ptr() == params.qkv_input.data_ptr()
    assert run_args[1] is kv_cache
    assert run_kwargs["block_tables"] is plan_args[0]
    assert run_kwargs["seq_lens"] is plan_args[1]
    assert run_kwargs["out"].shape == (2, 1, attn.num_heads, 512)
    assert run_kwargs["out"].data_ptr() == output.data_ptr()
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(128 + 64))
    assert run_kwargs["bmm2_scale"] == 1.0

    block_tables[:, 0].add_(20)
    sequence_lengths.add_(1)
    fmha.run_mla_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    assert wrapper.run.call_count == 2
    torch.testing.assert_close(
        plan_args[0],
        torch.tensor([[20, 21, 22], [23, 24, 25]], dtype=torch.int32),
    )
    torch.testing.assert_close(plan_args[1], torch.tensor([34, 65], dtype=torch.int32))
    assert fmha._mla_decode_wrappers[2] is wrapper


def test_mla_wrapper_cache_plans_each_batch_once_and_reuses_a_b_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    wrappers = [Mock(), Mock()]
    wrapper_factory = Mock(side_effect=wrappers)
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_mla_decode_wrapper",
        wrapper_factory,
    )
    block_tables = torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32)
    seq_lens = torch.tensor([33, 64], dtype=torch.int32)
    workspace = torch.empty(64, dtype=torch.uint8)

    first = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    cached_first = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    block_pointer_hit = _get_test_mla_wrapper(
        fmha,
        block_tables.clone(),
        seq_lens,
        workspace,
    )
    seq_pointer_hit = _get_test_mla_wrapper(
        fmha,
        block_tables,
        seq_lens.clone(),
        workspace,
    )
    profile_b = _get_test_mla_wrapper(
        fmha,
        block_tables[:1],
        seq_lens[:1],
        workspace,
    )
    profile_a_again = _get_test_mla_wrapper(
        fmha,
        block_tables,
        seq_lens,
        workspace,
    )

    assert all(
        result is wrappers[0]
        for result in (
            first,
            cached_first,
            block_pointer_hit,
            seq_pointer_hit,
            profile_a_again,
        )
    )
    assert profile_b is wrappers[1]
    assert wrapper_factory.call_count == 2
    for wrapper in wrappers:
        wrapper.plan.assert_called_once()
        assert wrapper.plan.call_args.kwargs["workspace_buffer"] is workspace
    assert fmha._mla_decode_wrappers[2] is wrappers[0]
    assert fmha._mla_decode_wrappers[1] is wrappers[1]


def test_mla_wrapper_capture_uses_cached_plan_and_rejects_plan_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    capturing = False
    monkeypatch.setattr(
        torch.cuda,
        "is_current_stream_capturing",
        lambda: capturing,
    )
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_mla_decode_wrapper",
        wrapper_factory,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    block_tables = torch.tensor([[0, 1, 2]], dtype=torch.int32)
    seq_lens = torch.tensor([33], dtype=torch.int32)
    workspace = torch.empty(64, dtype=torch.uint8)
    planned = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    capturing = True

    cached = _get_test_mla_wrapper(fmha, block_tables, seq_lens, workspace)
    with pytest.raises(RuntimeError, match="must be planned before CUDA graph capture"):
        _get_test_mla_wrapper(
            fmha,
            torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
            torch.tensor([33, 64], dtype=torch.int32),
            workspace,
        )

    assert cached is planned is wrapper
    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()


@pytest.mark.parametrize("is_cuda_graph", [False, True], ids=["eager", "cuda-graph"])
def test_mla_wrapper_receives_v2_bound_and_shared_workspace(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda_graph: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    kv_cache = torch.empty((20, 1, 32, 576), dtype=torch.bfloat16)
    block_tables = torch.tensor(
        [
            [[0, 1, 2], [10, 11, 12]],
            [[3, 4, 5], [13, 14, 15]],
        ],
        dtype=torch.int32,
    )
    build_metadata = Mock(return_value=(kv_cache, block_tables, None))
    wrapper = Mock()
    wrapper_factory = Mock(return_value=wrapper)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "build_trtllm_gen_kv_cache_metadata",
        build_metadata,
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_mla_workspace_size",
        Mock(return_value=96),
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_create_prims_mla_decode_wrapper",
        wrapper_factory,
    )

    get_page_index_upper_bound = Mock(return_value=20)
    metadata = SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        beam_width=1,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        host_kv_cache_pool_pointers=torch.tensor([1234], dtype=torch.int64),
        host_kv_cache_pool_mapping=torch.tensor([[0, 0]], dtype=torch.int32),
        kv_cache_manager=SimpleNamespace(
            impl=SimpleNamespace(get_page_index_upper_bound=get_page_index_upper_bound)
        ),
    )
    total_num_blocks = fmha._get_total_num_blocks(metadata)
    output = torch.empty((2, attn.num_heads, 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=64,
        is_fused_qkv=True,
    )
    sequence_lengths = torch.tensor([0, 33, 64], dtype=torch.int32)[1:]
    assert sequence_lengths.data_ptr() % 16 != 0
    params = FmhaParams(
        attn=attn,
        meta=metadata,
        fwd=forward_args,
        workspace=torch.empty(96, dtype=torch.uint8),
        qkv_input=torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16),
        context_buf=output,
        sequence_lengths=sequence_lengths,
        input_seq_length=1,
        max_past_kv_length=64,
        max_attention_window_size=64,
        cyclic_attention_window_size=64,
        num_tokens=2,
        seq_offset=2,
        tokens_per_block=32,
        kv_factor=1,
        total_num_blocks=total_num_blocks,
        batch_size=2,
        num_requests=2,
    )

    fmha.run_mla_generation(params)

    wrapper_factory.assert_called_once_with()
    wrapper.plan.assert_called_once()
    plan_args = wrapper.plan.call_args.args
    plan_kwargs = wrapper.plan.call_args.kwargs
    torch.testing.assert_close(
        plan_args[0],
        torch.tensor([[0, 1, 2], [3, 4, 5]], dtype=torch.int32),
    )
    assert plan_args[0].data_ptr() == fmha._page_indices_buffer.data_ptr()
    torch.testing.assert_close(plan_args[1], params.sequence_lengths)
    assert plan_args[1].data_ptr() == fmha._sequence_lengths_buffer.data_ptr()
    assert plan_args[1].data_ptr() % 16 == 0
    assert plan_args[2:] == (attn.num_heads, 512, 64, 32)
    assert plan_kwargs["live_metadata"] is True
    assert plan_kwargs["workspace_buffer"].data_ptr() == params.workspace.data_ptr()
    assert plan_kwargs["workspace_buffer"].numel() == 96
    wrapper.run.assert_called_once()
    run_args = wrapper.run.call_args.args
    run_kwargs = wrapper.run.call_args.kwargs
    assert run_args[0].shape == (2, 1, attn.num_heads, 576)
    assert run_args[0].data_ptr() == params.qkv_input.data_ptr()
    assert run_args[1] is kv_cache
    assert run_kwargs["block_tables"] is plan_args[0]
    assert run_kwargs["seq_lens"] is plan_args[1]
    assert run_kwargs["out"].shape == (2, 1, attn.num_heads, 512)
    assert run_kwargs["out"].data_ptr() == output.data_ptr()
    assert run_kwargs["bmm1_scale"] == pytest.approx(1.0 / math.sqrt(128 + 64))
    assert run_kwargs["bmm2_scale"] == 1.0
    builder_args = build_metadata.call_args.args
    assert builder_args[8] == total_num_blocks
    assert builder_args[10] == params.seq_offset
    assert builder_args[11] == 2
    assert builder_args[12] == torch.bfloat16
    get_page_index_upper_bound.assert_called_once()


@pytest.mark.parametrize("is_cuda_graph", [False, True], ids=["eager", "cuda-graph"])
def test_mla_prepare_workspace_sizes_caller_owned_workspace(
    monkeypatch: pytest.MonkeyPatch,
    is_cuda_graph: bool,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    workspace_size = Mock(return_value=48)
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_mla_workspace_size",
        workspace_size,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    q = torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        is_cuda_graph=is_cuda_graph,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        max_seq_len=80,
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([33, 64], dtype=torch.int32),
    )
    output = torch.empty((2, attn.num_heads * 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=96,
    )

    workspace = torch.empty(0, dtype=torch.uint8)
    fmha.prepare_workspace(
        q,
        None,
        None,
        metadata,
        forward_args,
        workspace,
    )

    workspace_size.assert_called_once()
    assert workspace_size.call_args.args[5] == 80
    assert workspace.numel() == 48


def test_mla_prepare_workspace_preserves_cached_wrappers_with_stable_allocation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    stream = Mock()
    monkeypatch.setattr(torch.cuda, "current_stream", Mock(return_value=stream))
    workspace_size = Mock(return_value=48)
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_mla_workspace_size",
        workspace_size,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    workspace = torch.empty(48, dtype=torch.uint8)
    fmha._update_workspace_allocation(workspace)
    wrapper = Mock()
    fmha._mla_decode_wrappers[2] = wrapper
    q = torch.empty((2, attn.num_heads * 576), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        is_cuda_graph=True,
        kv_cache_block_offsets=torch.empty((2, 2, 3), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([33, 64], dtype=torch.int32),
    )
    output = torch.empty((2, attn.num_heads * 512), dtype=torch.bfloat16)
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=96,
    )
    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    stream.synchronize.assert_not_called()
    assert fmha._mla_decode_wrappers[2] is wrapper
    assert workspace.numel() == 48
    workspace_size.assert_called_once()


def test_mla_workspace_size_cached_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    workspace_size = Mock(side_effect=lambda batch_size, *args, **kwargs: batch_size * 16)
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_mla_workspace_size",
        workspace_size,
    )
    attn = _Attention(head_dim=576, is_mla=True, num_heads=4)
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    workspace = torch.empty(0, dtype=torch.uint8)

    def prepare(batch_size: int) -> None:
        metadata = SimpleNamespace(
            is_cuda_graph=False,
            kv_cache_block_offsets=torch.empty((3, 2, 3), dtype=torch.int32),
            max_seq_len=80,
            max_num_requests=3,
            num_contexts=0,
            num_generations=batch_size,
            num_ctx_tokens=0,
            tokens_per_block=32,
            kv_lens_runtime=torch.tensor([33, 64, 80], dtype=torch.int32),
        )
        forward_args = AttentionForwardArgs(
            output=torch.empty((batch_size, attn.num_heads * 512), dtype=torch.bfloat16),
            attention_input_type=AttentionInputType.generation_only,
            attention_window_size=96,
        )
        q = torch.empty((batch_size, attn.num_heads * 576), dtype=torch.bfloat16)
        fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    prepare(2)
    prepare(3)
    prepare(2)

    assert [call.args[0] for call in workspace_size.call_args_list] == [2, 3]
    assert fmha._mla_workspace_sizes == {2: 32, 3: 48}
    assert workspace.numel() == 48


def test_decode_workspace_size_cached_per_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {
            "total_size": 64,
            "trtllm_gen_workspace_size": 64,
        },
    )
    workspace_size = Mock(side_effect=lambda batch_size, *args, **kwargs: batch_size * 16)
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        workspace_size,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    workspace = torch.empty(0, dtype=torch.uint8)

    def prepare(batch_size: int) -> None:
        metadata = SimpleNamespace(
            kv_cache_block_offsets=torch.empty((3, 2, 4), dtype=torch.int32),
            max_seq_len=96,
            max_num_requests=3,
            num_contexts=0,
            num_generations=batch_size,
            num_ctx_tokens=0,
            tokens_per_block=32,
        )
        forward_args = AttentionForwardArgs(
            output=torch.empty((batch_size, 8 * 128), dtype=torch.bfloat16),
            attention_input_type=AttentionInputType.generation_only,
            attention_window_size=128,
        )
        q = torch.empty((batch_size, 12 * 128), dtype=torch.bfloat16)
        fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)

    prepare(2)
    prepare(3)
    prepare(2)

    assert [call.args[0] for call in workspace_size.call_args_list] == [2, 3]
    assert fmha._decode_workspace_sizes == {2: 32, 3: 48}
    assert workspace.numel() == 64


def test_decode_prepare_workspace_reserves_aligned_tail_for_large_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {
            "total_size": 64,
            "trtllm_gen_workspace_size": 32,
        },
    )
    workspace_size = Mock(return_value=48)
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        workspace_size,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        max_seq_len=96,
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((2, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=128,
    )
    workspace = torch.empty(0, dtype=torch.uint8)

    fmha.prepare_workspace(
        torch.empty((2, 12 * 128), dtype=torch.bfloat16),
        None,
        None,
        metadata,
        forward_args,
        workspace,
    )

    assert fmha._decode_workspace_offset_bytes == 64
    assert fmha._decode_workspace_required_bytes == 48
    assert workspace_size.call_args.args[5] == 96
    assert workspace.numel() == 112
    decode_workspace = fmha._get_decode_workspace(workspace)
    assert decode_workspace.data_ptr() == workspace.data_ptr() + 64
    assert decode_workspace.numel() == 48


def test_decode_workspace_tail_is_stable_across_mixed_context_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    context_layout = Mock(
        side_effect=(
            {"total_size": 64},
            {"total_size": 320},
        )
    )
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_context_workspace_layout",
        context_layout,
    )
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {
            "total_size": 64,
            "trtllm_gen_workspace_size": 32,
        },
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        lambda *args, **kwargs: 48,
    )
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((4, 2, 4), dtype=torch.int32),
        max_num_requests=4,
        num_contexts=2,
        num_generations=2,
        num_ctx_tokens=4,
        tokens_per_block=32,
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((6, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=128,
    )
    q = torch.empty((6, 12 * 128), dtype=torch.bfloat16)
    workspace = torch.empty(1024, dtype=torch.uint8)

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    first_workspace = fmha._get_decode_workspace(workspace, workspace[:32])
    cached_wrapper = Mock()
    fmha._decode_wrappers[2] = cached_wrapper

    fmha.prepare_workspace(q, None, None, metadata, forward_args, workspace)
    second_workspace = fmha._get_decode_workspace(workspace, workspace[:32])

    assert context_layout.call_count == 2
    assert fmha._decode_wrappers[2] is cached_wrapper
    assert fmha._decode_workspace_offset_bytes == 960
    assert first_workspace.data_ptr() == second_workspace.data_ptr()
    assert first_workspace.numel() == second_workspace.numel() == 48


def test_workspace_cannot_grow_during_capture(monkeypatch: pytest.MonkeyPatch) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._multi_processor_count = 1
    fmha._ensure_metadata_buffers(torch.device("cpu"), 2, 4, 32)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    monkeypatch.setattr(
        prims_ts_module.thop,
        "get_trtllm_gen_generation_workspace_layout",
        lambda *args, **kwargs: {
            "total_size": 32,
            "trtllm_gen_workspace_size": 32,
        },
    )
    monkeypatch.setattr(
        prims_ts_module,
        "_get_prims_decode_workspace_size",
        lambda *args, **kwargs: 32,
    )
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty((2, 2, 4), dtype=torch.int32),
        max_num_requests=2,
        num_contexts=0,
        num_generations=2,
        num_ctx_tokens=0,
        tokens_per_block=32,
        kv_lens_runtime=torch.tensor([64, 96], dtype=torch.int32),
    )
    forward_args = AttentionForwardArgs(
        output=torch.empty((2, 8 * 128), dtype=torch.bfloat16),
        attention_input_type=AttentionInputType.generation_only,
        attention_window_size=128,
    )

    with pytest.raises(
        RuntimeError,
        match="PrimTS caller workspace must be sized before CUDA graph capture",
    ):
        fmha.prepare_workspace(
            torch.empty((2, 12 * 128), dtype=torch.bfloat16),
            None,
            None,
            metadata,
            forward_args,
            torch.empty(16, dtype=torch.uint8),
        )


def test_metadata_buffers_cannot_grow_during_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    with pytest.raises(
        RuntimeError,
        match="PrimTS metadata buffers must be allocated before CUDA graph capture",
    ):
        fmha._ensure_metadata_buffers(torch.device("cpu"), 2, 4, 32)


def test_metadata_buffer_growth_retains_graph_visible_allocations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        2,
        4,
        32,
        need_context=True,
    )
    original_buffers = (
        fmha._page_indices_buffer,
        fmha._fixed_indptr_buffer,
        fmha._interleaved_indptr_buffer,
        fmha._sequence_lengths_buffer,
        fmha._context_page_indices_buffer,
        fmha._context_page_gather_indices_buffer,
        fmha._context_page_columns_buffer,
        fmha._context_last_page_indices_buffer,
    )

    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        3,
        4,
        32,
        need_context=True,
    )
    fmha._ensure_metadata_buffers(
        torch.device("cpu"),
        3,
        4,
        32,
        need_context=True,
    )

    assert len(fmha._retained_metadata_buffers) == 1
    assert all(
        retained is original
        for retained, original in zip(
            fmha._retained_metadata_buffers[0], original_buffers, strict=True
        )
    )
    assert all(
        current is not original
        for current, original in zip(
            (
                fmha._page_indices_buffer,
                fmha._fixed_indptr_buffer,
                fmha._interleaved_indptr_buffer,
                fmha._sequence_lengths_buffer,
                fmha._context_page_indices_buffer,
                fmha._context_page_gather_indices_buffer,
                fmha._context_page_columns_buffer,
                fmha._context_last_page_indices_buffer,
            ),
            original_buffers,
            strict=True,
        )
    )


def test_phased_forward_routes_mixed_batch_to_context_and_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    attn = _Attention()
    fmha = PrimsTSFmha(attn)
    context_calls = []
    generation_calls = []
    run_context = Mock(side_effect=lambda params: context_calls.append(replace(params)))
    run_generation = Mock(side_effect=lambda params: generation_calls.append(replace(params)))
    monkeypatch.setattr(fmha, "prepare_workspace", Mock())
    monkeypatch.setattr(fmha, "run_context", run_context)
    monkeypatch.setattr(fmha, "run_generation", run_generation)

    q = torch.empty((5, 128), dtype=torch.bfloat16)
    output = torch.empty((5, attn.num_heads * attn.head_dim), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        kv_cache_block_offsets=torch.empty(1),
        effective_workspace=torch.empty(0, dtype=torch.int8),
        num_contexts=1,
        num_ctx_tokens=3,
        num_generations=2,
        cache_indirection=None,
        beam_width=1,
        tokens_per_block=32,
        kv_lens_cuda_runtime=torch.tensor([3, 65, 97], dtype=torch.int32),
        kv_lens_runtime=torch.tensor([3, 65, 97], dtype=torch.int32),
        prompt_lens_cuda_runtime=torch.tensor([3, 1, 1], dtype=torch.int32),
        prompt_lens_cpu_runtime=torch.tensor([3, 1, 1], dtype=torch.int32),
        is_spec_decoding_enabled=False,
        is_cross=False,
        kv_cache_manager=None,
    )
    forward_args = AttentionForwardArgs(
        output=output,
        attention_input_type=AttentionInputType.mixed,
        attention_window_size=128,
    )

    fmha.forward(q, None, None, metadata, forward_args)

    run_context.assert_called_once()
    context_params = context_calls[0]
    assert context_params.num_tokens == 3
    assert context_params.seq_offset == 0
    assert context_params.batch_size == 1
    assert context_params.num_requests == 1
    assert context_params.attention_input is not None
    assert context_params.attention_input.shape[0] == 3
    assert context_params.context_buf is not None
    assert context_params.context_buf.shape == (3, attn.num_heads, attn.head_dim)

    run_generation.assert_called_once()
    generation_params = generation_calls[0]
    assert generation_params.num_tokens == 2
    assert generation_params.seq_offset == 1
    assert generation_params.batch_size == 2
    assert generation_params.num_requests == 2
    assert generation_params.attention_input is not None
    assert generation_params.attention_input.shape[0] == 2
    assert generation_params.context_buf is not None
    assert generation_params.context_buf.shape == (2, attn.num_heads, attn.head_dim)
