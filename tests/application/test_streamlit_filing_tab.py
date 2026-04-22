"""Streamlit 财报页面辅助函数测试。"""

from types import SimpleNamespace

from dayu.contracts.fins import DownloadFilingResultItem
from dayu.web.streamlit.pages import filing_tab
from dayu.web.streamlit.pages.filing_tab import (
    _build_file_downloaded_message,
    _build_filing_completed_message,
    _get_download_header_button_text,
    _format_download_size,
    _should_show_download_settings_for_ticker,
)


def test_format_download_size_when_size_is_none() -> None:
    """验证未知大小展示文案。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    assert _format_download_size(None) is None


def test_format_download_size_when_size_is_zero() -> None:
    """验证零字节文件展示文案。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    assert _format_download_size(0) == "0 字节"


def test_format_download_size_when_size_is_positive() -> None:
    """验证正常文件大小展示文案。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    assert _format_download_size(12345) == "12345 字节"


def test_build_file_downloaded_message_without_size() -> None:
    """验证未知大小时不显示文件大小。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    assert _build_file_downloaded_message("a.htm", None) == "已下载 a.htm"


def test_build_file_downloaded_message_with_size() -> None:
    """验证已知大小时显示文件大小。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    assert _build_file_downloaded_message("a.htm", 10) == "已下载 a.htm (10 字节)"


def test_build_filing_completed_message_for_skipped_with_reason() -> None:
    """验证 skipped 结果会展示跳过原因。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    filing_result = DownloadFilingResultItem(
        document_id="fil_1",
        status="skipped",
        form_type="10-K",
        reason_message="本地已有完整下载结果，跳过重新下载",
    )
    message, level = _build_filing_completed_message("10-K", filing_result, None)
    assert message == "跳过下载 10-K: 本地已有完整下载结果，跳过重新下载"
    assert level == "warning"


def test_build_filing_completed_message_for_downloaded_with_count() -> None:
    """验证 downloaded 结果会展示文件数量。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """
    filing_result = DownloadFilingResultItem(
        document_id="fil_2",
        status="downloaded",
        form_type="10-Q",
        downloaded_files=3,
    )
    message, level = _build_filing_completed_message("10-Q", filing_result, None)
    assert message == "完成下载 10-Q（3 个文件）"
    assert level == "info"


def test_should_show_download_settings_for_ticker_when_current_ticker_is_expanded() -> None:
    """验证当前股票展开下载设置时返回 True。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    filing_tab.st = SimpleNamespace(
        session_state={
            "show_download_settings": True,
            "download_settings_ticker": "AAPL",
        }
    )

    assert _should_show_download_settings_for_ticker("AAPL") is True


def test_should_show_download_settings_for_ticker_when_other_ticker_is_expanded() -> None:
    """验证其他股票展开下载设置时当前股票返回 False。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    filing_tab.st = SimpleNamespace(
        session_state={
            "show_download_settings": True,
            "download_settings_ticker": "MSFT",
        }
    )

    assert _should_show_download_settings_for_ticker("AAPL") is False


def test_get_download_header_button_text_matches_expanded_state() -> None:
    """验证页头按钮文案会跟随下载设置展开状态切换。

    Args:
        无。

    Returns:
        无。

    Raises:
        无。
    """

    filing_tab.st = SimpleNamespace(
        session_state={
            "show_download_settings": False,
            "download_settings_ticker": "AAPL",
        }
    )
    assert _get_download_header_button_text("AAPL") == "📥 下载财报"

    filing_tab.st = SimpleNamespace(
        session_state={
            "show_download_settings": True,
            "download_settings_ticker": "AAPL",
        }
    )
    assert _get_download_header_button_text("AAPL") == "❌ 关闭下载"
