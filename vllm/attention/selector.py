# SPDX-License-Identifier: Apache-2.0

import os
from contextlib import contextmanager
from functools import cache
from typing import Generator, Optional, Type

import torch

import vllm.envs as envs
from vllm.attention.backends.abstract import AttentionBackend
from vllm.logger import init_logger
from vllm.platforms import _Backend, current_platform
from vllm.utils import STR_BACKEND_ENV_VAR, resolve_obj_by_qualname

logger = init_logger(__name__)


def backend_name_to_enum(backend_name: str) -> Optional[_Backend]:
    """
    Convert a string backend name to a _Backend enum value.

    Returns:
    * _Backend: enum value if backend_name is a valid in-tree type
    * None: otherwise it's an invalid in-tree type or an out-of-tree platform is
            loaded.
    """
    assert backend_name is not None
    return _Backend[backend_name] if backend_name in _Backend.__members__ else \
          None


def get_env_variable_attn_backend() -> Optional[_Backend]:
    '''
    Get the backend override specified by the vLLM attention
    backend environment variable, if one is specified.

    Returns:

    * _Backend enum value if an override is specified
    * None otherwise
    '''
    backend_name = os.environ.get(STR_BACKEND_ENV_VAR)
    return (None
            if backend_name is None else backend_name_to_enum(backend_name))


# Global state allows a particular choice of backend
# to be forced, overriding the logic which auto-selects
# a backend based on system & workload configuration
# (default behavior if this variable is None)
#
# THIS SELECTION TAKES PRECEDENCE OVER THE
# VLLM_ATTENTION_BACKEND ENVIRONMENT VARIABLE
forced_attn_backend: Optional[_Backend] = None


def global_force_attn_backend(attn_backend: Optional[_Backend]) -> None:
    '''
    Force all attention operations to use a specified backend.

    Passing `None` for the argument re-enables automatic
    backend selection.,

    Arguments:

    * attn_backend: backend selection (None to revert to auto)
    '''
    global forced_attn_backend
    forced_attn_backend = attn_backend


def get_global_forced_attn_backend() -> Optional[_Backend]:
    '''
    Get the currently-forced choice of attention backend,
    or None if auto-selection is currently enabled.
    '''
    return forced_attn_backend


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
    """选择 attention 后端，并延迟加载。"""
    # 在`@lru_cache`装饰器后面访问`envs.*`可能会导致在两次调用之间值发生变化时，从缓存中返回错误的值。
    # 为了避免这种情况，我们在这里读取`envs.VLLM_USE_V1`，并将其显式传递给私有函数。
    return _cached_get_attn_backend(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        block_size=block_size,
        is_attention_free=is_attention_free,
        is_blocksparse=is_blocksparse,
        use_v1=envs.VLLM_USE_V1,
        use_mla=use_mla,
    )


@cache
def _cached_get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: Optional[str],
    block_size: int,
    is_attention_free: bool,
    is_blocksparse: bool = False,
    use_v1: bool = False,
    use_mla: bool = False,
) -> Type[AttentionBackend]:
    if is_blocksparse:
        logger.info("Using BlocksparseFlashAttention backend.")
        from vllm.attention.backends.blocksparse_attn import (
            BlocksparseFlashAttentionBackend)
        return BlocksparseFlashAttentionBackend

    # 如果没有注意力层（例如，我们正在运行 Mamba），则使用占位符`NO_ATTENTION`。
    if is_attention_free:
        from vllm.attention.backends.placeholder_attn import (
            PlaceholderAttentionBackend)
        return PlaceholderAttentionBackend

    # 检查是否之前曾经强制指定了某个特定的后端选项。
    #
    # 此选择将覆盖`VLLM_ATTENTION_BACKEND`环境变量。
    selected_backend = None
    backend_by_global_setting: Optional[_Backend] = (
        get_global_forced_attn_backend())
    if backend_by_global_setting is not None:
        selected_backend = backend_by_global_setting
    else:
        # 检查环境变量，并在指定时覆盖。
        backend_by_env_var: Optional[str] = envs.VLLM_ATTENTION_BACKEND
        if backend_by_env_var is not None:
            selected_backend = backend_name_to_enum(backend_by_env_var)

    # 获取特定设备的`attn_backend`（注意力后端）。
    attention_cls = current_platform.get_attn_backend_cls(
        selected_backend, head_size, dtype, kv_cache_dtype, block_size, use_v1,
        use_mla)
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}")
    return resolve_obj_by_qualname(attention_cls)


@contextmanager
def global_force_attn_backend_context_manager(
        attn_backend: _Backend) -> Generator[None, None, None]:
    '''
    Globally force a vLLM attention backend override within a
    context manager, reverting the global attention backend
    override to its prior state upon exiting the context
    manager.

    Arguments:

    * attn_backend: attention backend to force

    Returns:

    * Generator
    '''

    # Save the current state of the global backend override (if any)
    original_value = get_global_forced_attn_backend()

    # Globally force the new backend override
    global_force_attn_backend(attn_backend)

    # Yield control back to the enclosed code block
    try:
        yield
    finally:
        # Revert the original global backend override, if any
        global_force_attn_backend(original_value)
