"""`python -m dayu.web` 入口。"""

from __future__ import annotations

from dayu.web.streamlit_app import run_streamlit


def main() -> int:
    """运行 Streamlit Web 模块入口。

    Args:
        无。

    Returns:
        Streamlit 子进程退出码。

    Raises:
        无。异常由下层 ``run_streamlit()`` 透传。
    """

    return run_streamlit()


if __name__ == "__main__":
    raise SystemExit(main())
