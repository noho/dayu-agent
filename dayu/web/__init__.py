"""Web UI 适配层占位包。

本包当前提供 FastAPI 适配入口，Web 组合根应显式注入所需 Service 协议，
请求期路由不应再依赖全局 service locator。

``create_fastapi_app`` 在运行期通过 ``__getattr__`` 惰性加载，避免仅使用
Streamlit 子树时拉取 FastAPI 装配链；类型检查期仍解析到真实符号以满足
``__all__`` 与静态分析。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = ["create_fastapi_app"]

if TYPE_CHECKING:
    from dayu.web.fastapi_app import create_fastapi_app
else:

    def __getattr__(name: str) -> object:
        """惰性加载 ``create_fastapi_app``。"""

        if name == "create_fastapi_app":
            from dayu.web.fastapi_app import create_fastapi_app as created

            return created
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
