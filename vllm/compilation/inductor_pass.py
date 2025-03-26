# SPDX-License-Identifier: Apache-2.0

import hashlib
import importlib.metadata
import inspect
import json
import types
from typing import Any, Callable, Dict, Optional, Union

import torch
from packaging.version import Version
from torch import fx

if Version(importlib.metadata.version('torch')) >= Version("2.6"):
    from torch._inductor.custom_graph_pass import CustomGraphPass
else:
    # CustomGraphPass is not present in 2.5 or lower, import our version
    from .torch25_custom_graph_pass import (  # noqa: yapf
        Torch25CustomGraphPass as CustomGraphPass)


class InductorPass(CustomGraphPass):
    """
    这是一个自定义图传递（graph pass），它使用其源代码的哈希值作为 UUID。  
    这被定义为一种便利的方式，在大多数情况下应该可以正常工作。
    """

    def uuid(self) -> Any:
        """
        为该传递（pass）提供一个唯一的标识符，用于 Inductor 代码缓存中。  
        该标识符应当依赖于传递的实现，以便在传递发生变化时触发重新编译。  
        默认情况下，会对对象的源代码进行哈希处理。
        """
        return InductorPass.hash_source(self)

    @staticmethod
    def hash_source(*srcs: Union[str, Any]):
        """用于对函数或对象的源代码进行哈希处理的工具方法。  
        - **参数**：  
        - `srcs`：要加入哈希的字符串或对象。对象和函数会检查其源代码。
        :return:
        """
        hasher = hashlib.sha256()
        for src in srcs:
            if isinstance(src, str):
                src_str = src
            elif isinstance(src, types.FunctionType):
                src_str = inspect.getsource(src)
            else:
                src_str = inspect.getsource(src.__class__)
            hasher.update(src_str.encode("utf-8"))
        return hasher.hexdigest()

    @staticmethod
    def hash_dict(dict_: Dict[Any, Any]):
        """
        Utility method to hash a dictionary, can alternatively be used for uuid.
        :return: A sha256 hash of the json rep of the dictionary.
        """
        encoded = json.dumps(dict_, sort_keys=True).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


class CallableInductorPass(InductorPass):
    """
    This class is a wrapper for a callable that automatically provides an
    implementation of the UUID.
    """

    def __init__(self,
                 callable: Callable[[fx.Graph], None],
                 uuid: Optional[Any] = None):
        self.callable = callable
        self._uuid = self.hash_source(callable) if uuid is None else uuid

    def __call__(self, graph: torch.fx.Graph):
        self.callable(graph)

    def uuid(self) -> Any:
        return self._uuid
