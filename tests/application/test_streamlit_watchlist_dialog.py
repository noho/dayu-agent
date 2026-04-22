"""Streamlit 自选股管理对话框行为测试。"""

from __future__ import annotations

from types import SimpleNamespace

from dayu.web.streamlit.components import watchlist_dialog


def test_trigger_sidebar_refresh_sets_flag_and_reruns(monkeypatch) -> None:
    """保存成功后应设置刷新标记并触发 rerun。"""

    rerun_called = {"value": False}

    def _fake_rerun() -> None:
        rerun_called["value"] = True

    fake_streamlit = SimpleNamespace(
        session_state={},
        rerun=_fake_rerun,
    )
    monkeypatch.setattr(watchlist_dialog, "st", fake_streamlit)

    watchlist_dialog._trigger_sidebar_refresh()

    assert fake_streamlit.session_state["watchlist_needs_refresh"] is True
    assert rerun_called["value"] is True
