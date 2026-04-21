"""SEC HTTP 响应磁盘缓存单元测试。

覆盖 sec_pipeline 模块中的缓存辅助函数：
- _sec_cache_path
- _read_sec_cache
- _write_sec_cache
- _browse_edgar_filings_to_dicts
- _dicts_to_browse_edgar_filings

以及与下载流程集成的路径：
- _filter_filings 中 history pages 缓存命中/写入
- _extend_with_browse_edgar_sc13 中 browse-edgar 缓存命中/写入
- _retry_sc13_if_empty 中 retry 自动复用缓存（不重复 HTTP）
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from dayu.fins.downloaders.sec_downloader import BrowseEdgarFiling
from dayu.fins.pipelines import sec_pipeline
from dayu.fins.pipelines.sec_download_state import (
    _SEC_CACHE_CATEGORY_BROWSE_EDGAR,
    _SEC_CACHE_CATEGORY_SUBMISSIONS,
    _SEC_CACHE_DIR,
    _SEC_CACHE_TTL_HOURS,
    _browse_edgar_filings_to_dicts,
    _dicts_to_browse_edgar_filings,
    _read_sec_cache,
    _sec_cache_path,
    _write_sec_cache,
)
from dayu.fins.pipelines.sec_filing_collection import FilingRecord
from dayu.fins.pipelines.sec_pipeline import SecPipeline
from dayu.fins.processors.registry import build_fins_processor_registry


# ---------------------------------------------------------------------------
# 测试用桩
# ---------------------------------------------------------------------------


class _StubRepository:
    """最小存储库桩，满足 SecPipeline 构造函数要求。"""

    def save_document(self, *args: Any, **kwargs: Any) -> None:
        """空操作。"""

    def load_document(self, *args: Any, **kwargs: Any) -> None:
        """空操作。"""


def _make_pipeline(tmp_path: Path) -> SecPipeline:
    """创建最小 SecPipeline 用于方法测试。"""
    return SecPipeline(
        workspace_root=tmp_path,
        processor_registry=build_fins_processor_registry(),
    )


# ---------------------------------------------------------------------------
# _sec_cache_path
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestSecCachePath:
    """_sec_cache_path 路径构造测试。"""

    def test_basic_path(self, tmp_path: Path) -> None:
        """基本路径构造正确。"""
        path = _sec_cache_path(tmp_path, "submissions", "CIK0000019617-submissions-001")
        expected = tmp_path / _SEC_CACHE_DIR / "submissions" / "CIK0000019617-submissions-001.json"
        assert path == expected

    def test_slash_in_key_replaced(self, tmp_path: Path) -> None:
        """key 中的斜杠被替换为下划线，防止路径穿越。"""
        path = _sec_cache_path(tmp_path, "submissions", "a/b/c")
        assert "/" not in path.name
        assert path.name == "a_b_c.json"

    def test_backslash_in_key_replaced(self, tmp_path: Path) -> None:
        """key 中的反斜杠被替换为下划线。"""
        path = _sec_cache_path(tmp_path, "browse_edgar", r"005\12345")
        assert "\\" not in path.name


# ---------------------------------------------------------------------------
# _read_sec_cache / _write_sec_cache
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestReadWriteSecCache:
    """缓存读写往返测试。"""

    def test_read_returns_none_when_not_exists(self, tmp_path: Path) -> None:
        """文件不存在时返回 None。"""
        result = _read_sec_cache(tmp_path, "submissions", "nonexistent")
        assert result is None

    def test_write_then_read_roundtrip(self, tmp_path: Path) -> None:
        """写入后读取应返回相同数据。"""
        data = {"form": ["10-K", "SC 13D"], "filingDate": ["2025-01-15", "2025-02-10"]}
        _write_sec_cache(tmp_path, "submissions", "test_key", data)
        loaded = _read_sec_cache(tmp_path, "submissions", "test_key")
        assert loaded == data

    def test_write_creates_parent_directories(self, tmp_path: Path) -> None:
        """写入时自动创建父目录。"""
        _write_sec_cache(tmp_path, "submissions", "deep_key", {"hello": "world"})
        path = _sec_cache_path(tmp_path, "submissions", "deep_key")
        assert path.exists()

    def test_read_returns_none_when_expired(self, tmp_path: Path) -> None:
        """缓存过期时返回 None。"""
        _write_sec_cache(tmp_path, "submissions", "old_key", {"data": 1})
        path = _sec_cache_path(tmp_path, "submissions", "old_key")
        # 将 mtime 设为 25 小时前
        old_time = time.time() - (_SEC_CACHE_TTL_HOURS + 1) * 3600
        os.utime(path, (old_time, old_time))
        result = _read_sec_cache(tmp_path, "submissions", "old_key")
        assert result is None

    def test_read_valid_within_ttl(self, tmp_path: Path) -> None:
        """TTL 内的缓存应正常读取。"""
        _write_sec_cache(tmp_path, "submissions", "fresh_key", {"val": 42})
        result = _read_sec_cache(tmp_path, "submissions", "fresh_key")
        assert result == {"val": 42}

    def test_read_returns_none_on_corrupt_json(self, tmp_path: Path) -> None:
        """缓存文件内容损坏时返回 None。"""
        path = _sec_cache_path(tmp_path, "submissions", "corrupt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{{{invalid json", encoding="utf-8")
        result = _read_sec_cache(tmp_path, "submissions", "corrupt")
        assert result is None

    def test_custom_ttl(self, tmp_path: Path) -> None:
        """自定义 TTL 生效。"""
        _write_sec_cache(tmp_path, "test_cat", "ttl_key", {"v": 1})
        path = _sec_cache_path(tmp_path, "test_cat", "ttl_key")
        # 设为 2 小时前
        old_time = time.time() - 2 * 3600
        os.utime(path, (old_time, old_time))
        # ttl=1h → 过期
        assert _read_sec_cache(tmp_path, "test_cat", "ttl_key", ttl_hours=1) is None
        # ttl=3h → 未过期
        assert _read_sec_cache(tmp_path, "test_cat", "ttl_key", ttl_hours=3) == {"v": 1}

    def test_write_failure_does_not_raise(self, tmp_path: Path) -> None:
        """写入失败（如只读目录）不抛异常。"""
        # 使用一个不可能写入的路径
        bad_root = Path("/dev/null/impossible_path")
        # 不应抛出异常
        _write_sec_cache(bad_root, "cat", "key", {"data": 1})


# ---------------------------------------------------------------------------
# _browse_edgar_filings_to_dicts / _dicts_to_browse_edgar_filings
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestBrowseEdgarSerialization:
    """BrowseEdgarFiling 序列化/反序列化测试。"""

    def test_roundtrip(self) -> None:
        """序列化后反序列化应还原对象。"""
        entries = [
            BrowseEdgarFiling(
                form_type="SC 13D",
                filing_date="2025-01-15",
                accession_number="0001234567-25-000001",
                cik="0000320193",
                index_url="https://www.sec.gov/Archives/edgar/data/...",
            ),
            BrowseEdgarFiling(
                form_type="SC 13G/A",
                filing_date="2025-06-01",
                accession_number="0001234567-25-000002",
                cik="0000886982",
                index_url="https://www.sec.gov/Archives/edgar/data/...",
            ),
        ]
        dicts = _browse_edgar_filings_to_dicts(entries)
        restored = _dicts_to_browse_edgar_filings(dicts)
        assert len(restored) == 2
        assert restored[0].form_type == "SC 13D"
        assert restored[0].accession_number == "0001234567-25-000001"
        assert restored[1].form_type == "SC 13G/A"
        assert restored[1].cik == "0000886982"

    def test_empty_list(self) -> None:
        """空列表往返正常。"""
        assert _browse_edgar_filings_to_dicts([]) == []
        assert _dicts_to_browse_edgar_filings([]) == []

    def test_serialized_is_json_compatible(self) -> None:
        """序列化结果可直接 JSON 编码。"""
        entries = [
            BrowseEdgarFiling(
                form_type="SC 13D",
                filing_date="2025-01-15",
                accession_number="0001234567-25-000001",
                cik="0000320193",
                index_url="https://example.com",
            ),
        ]
        dicts = _browse_edgar_filings_to_dicts(entries)
        json_str = json.dumps(dicts, ensure_ascii=False)
        restored_dicts = json.loads(json_str)
        assert restored_dicts == dicts


# ---------------------------------------------------------------------------
# _filter_filings — history pages 缓存集成
# ---------------------------------------------------------------------------


def _make_submissions(
    recent_forms: Optional[list[str]] = None,
    history_files: Optional[list[dict[str, str]]] = None,
) -> dict[str, Any]:
    """构造最小 submissions 数据。"""

    recent_forms = recent_forms or []
    count = len(recent_forms)
    recent = {
        "form": recent_forms,
        "filingDate": [f"2025-01-{(i+1):02d}" for i in range(count)],
        "reportDate": ["" for _ in range(count)],
        "accessionNumber": [f"0001234567-25-{(i+100):06d}" for i in range(count)],
        "primaryDocument": [f"doc{i}.htm" for i in range(count)],
        "fileNumber": [f"001-{(i+1000):05d}" for i in range(count)],
    }
    return {
        "filings": {
            "recent": recent,
            "files": history_files or [],
        },
    }


def _make_history_page(
    forms: list[str],
    accession_base: int = 200,
) -> dict[str, Any]:
    """构造 history page 数据。"""

    count = len(forms)
    return {
        "form": forms,
        "filingDate": [f"2024-06-{(i+1):02d}" for i in range(count)],
        "reportDate": ["" for _ in range(count)],
        "accessionNumber": [f"0001234567-24-{(accession_base+i):06d}" for i in range(count)],
        "primaryDocument": [f"hist{i}.htm" for i in range(count)],
        "fileNumber": [f"001-{(accession_base+i):05d}" for i in range(count)],
    }


@pytest.mark.unit
@pytest.mark.asyncio
class TestFilterFilingsCache:
    """_filter_filings 中 history pages 缓存集成测试。"""

    async def test_cache_miss_fetches_and_writes(self, tmp_path: Path) -> None:
        """缓存未命中时应调用 HTTP 并写入缓存。"""
        pipeline = _make_pipeline(tmp_path)
        history_data = _make_history_page(["10-K"])
        submissions = _make_submissions(
            recent_forms=["10-K"],
            history_files=[{"name": "CIK0000019617-submissions-001.json"}],
        )

        # mock downloader.fetch_json 返回 history page
        async def fake_fetch_json(url: str) -> dict[str, Any]:
            return history_data

        pipeline._downloader.fetch_json = fake_fetch_json  # type: ignore[assignment]

        import datetime as dt
        form_windows = {"10-K": dt.date(2024, 1, 1)}
        filings, filenums = await pipeline._filter_filings(
            ticker="AAPL",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="19617",
        )

        # 应至少有 history page 中的 1 条 10-K
        assert any(f.form_type == "10-K" for f in filings)

        # 缓存文件应已写入
        cache_path = _sec_cache_path(
            tmp_path, _SEC_CACHE_CATEGORY_SUBMISSIONS, "CIK0000019617-submissions-001",
        )
        assert cache_path.exists()
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        assert cached["form"] == ["10-K"]

    async def test_cache_hit_skips_http(self, tmp_path: Path) -> None:
        """缓存命中时不应调用 HTTP。"""
        pipeline = _make_pipeline(tmp_path)
        history_data = _make_history_page(["10-K", "DEF 14A"])

        # 预写入缓存
        _write_sec_cache(
            tmp_path, _SEC_CACHE_CATEGORY_SUBMISSIONS,
            "CIK0000019617-submissions-001", history_data,
        )

        submissions = _make_submissions(
            recent_forms=[],
            history_files=[{"name": "CIK0000019617-submissions-001.json"}],
        )
        call_count = 0

        async def counting_fetch_json(url: str) -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"form": [], "filingDate": [], "reportDate": [],
                    "accessionNumber": [], "primaryDocument": [], "fileNumber": []}

        pipeline._downloader.fetch_json = counting_fetch_json  # type: ignore[assignment]

        import datetime as dt
        form_windows = {"10-K": dt.date(2024, 1, 1), "DEF 14A": dt.date(2023, 1, 1)}
        filings, _ = await pipeline._filter_filings(
            ticker="AAPL",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="19617",
        )

        # HTTP 不应被调用
        assert call_count == 0
        # 缓存中的数据应被使用
        assert any(f.form_type == "10-K" for f in filings)

    async def test_expired_cache_re_fetches(self, tmp_path: Path) -> None:
        """过期缓存应重新 HTTP 拉取。"""
        pipeline = _make_pipeline(tmp_path)
        old_data = _make_history_page(["DEF 14A"])
        new_data = _make_history_page(["10-K"], accession_base=300)

        # 写入过期缓存
        _write_sec_cache(
            tmp_path, _SEC_CACHE_CATEGORY_SUBMISSIONS,
            "CIK0000019617-submissions-001", old_data,
        )
        cache_path = _sec_cache_path(
            tmp_path, _SEC_CACHE_CATEGORY_SUBMISSIONS, "CIK0000019617-submissions-001",
        )
        old_time = time.time() - (_SEC_CACHE_TTL_HOURS + 1) * 3600
        os.utime(cache_path, (old_time, old_time))

        submissions = _make_submissions(
            history_files=[{"name": "CIK0000019617-submissions-001.json"}],
        )

        async def fake_fetch_json(url: str) -> dict[str, Any]:
            return new_data

        pipeline._downloader.fetch_json = fake_fetch_json  # type: ignore[assignment]

        import datetime as dt
        form_windows = {"10-K": dt.date(2024, 1, 1)}
        filings, _ = await pipeline._filter_filings(
            ticker="AAPL",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="19617",
        )

        # 应使用新数据
        assert any(f.form_type == "10-K" for f in filings)

    async def test_multiple_history_files_cached_independently(self, tmp_path: Path) -> None:
        """多个 history files 各自独立缓存。"""
        pipeline = _make_pipeline(tmp_path)
        page1 = _make_history_page(["10-K"], accession_base=400)
        page2 = _make_history_page(["DEF 14A"], accession_base=500)

        call_urls: list[str] = []

        async def fake_fetch_json(url: str) -> dict[str, Any]:
            call_urls.append(url)
            if "001" in url:
                return page1
            return page2

        pipeline._downloader.fetch_json = fake_fetch_json  # type: ignore[assignment]

        submissions = _make_submissions(
            history_files=[
                {"name": "CIK0000019617-submissions-001.json"},
                {"name": "CIK0000019617-submissions-002.json"},
            ],
        )

        import datetime as dt
        form_windows = {"10-K": dt.date(2024, 1, 1), "DEF 14A": dt.date(2023, 1, 1)}
        await pipeline._filter_filings(
            ticker="AAPL",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="19617",
        )

        # 两个都应被 HTTP 拉取
        assert len(call_urls) == 2

        # 两个缓存文件都应存在
        assert _sec_cache_path(
            tmp_path, _SEC_CACHE_CATEGORY_SUBMISSIONS, "CIK0000019617-submissions-001",
        ).exists()
        assert _sec_cache_path(
            tmp_path, _SEC_CACHE_CATEGORY_SUBMISSIONS, "CIK0000019617-submissions-002",
        ).exists()


# ---------------------------------------------------------------------------
# _extend_with_browse_edgar_sc13 — browse-edgar 缓存集成
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestBrowseEdgarSc13Cache:
    """_extend_with_browse_edgar_sc13 中 browse-edgar 缓存集成测试。"""

    async def test_cache_miss_fetches_and_writes(self, tmp_path: Path) -> None:
        """缓存未命中时应调用 HTTP 并写入缓存。"""
        pipeline = _make_pipeline(tmp_path)
        browse_entries = [
            BrowseEdgarFiling(
                form_type="SC 13D",
                filing_date="2025-01-15",
                accession_number="0001234567-25-000001",
                cik="0000999999",
                index_url="https://www.sec.gov/Archives/...",
            ),
        ]

        async def fake_fetch_browse(filenum: str, count: int = 100) -> list:
            return browse_entries

        # mock _should_keep_sc13_direction → True
        async def fake_keep(*args: Any, **kwargs: Any) -> bool:
            return True

        # mock resolve_primary_document
        async def fake_resolve_primary(*args: Any, **kwargs: Any) -> str:
            return "sc13d.htm"

        pipeline._downloader.fetch_browse_edgar_filenum = fake_fetch_browse  # type: ignore[assignment]
        pipeline._should_keep_sc13_direction = fake_keep  # type: ignore[assignment]
        pipeline._downloader.resolve_primary_document = fake_resolve_primary  # type: ignore[assignment]

        import datetime as dt
        form_windows = {"SC 13D": dt.date(2024, 1, 1)}
        filings = await pipeline._extend_with_browse_edgar_sc13(
            ticker="AAPL",
            filings=[],
            filenums={"005-12345"},
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="320193",
        )

        # 应包含 browse-edgar 补充的 filing
        assert len(filings) >= 1
        assert any(f.form_type == "SC 13D" for f in filings)

        # 缓存文件应已写入
        cache_path = _sec_cache_path(
            tmp_path, _SEC_CACHE_CATEGORY_BROWSE_EDGAR, "005-12345",
        )
        assert cache_path.exists()

    async def test_cache_hit_skips_http(self, tmp_path: Path) -> None:
        """缓存命中时不应调用 browse-edgar HTTP。"""
        pipeline = _make_pipeline(tmp_path)
        cached_entries = [
            {
                "form_type": "SC 13G",
                "filing_date": "2025-03-01",
                "accession_number": "0009999999-25-000010",
                "cik": "0000888888",
                "index_url": "https://www.sec.gov/Archives/...",
            },
        ]
        # 预写入缓存
        _write_sec_cache(
            tmp_path, _SEC_CACHE_CATEGORY_BROWSE_EDGAR, "005-67890", cached_entries,
        )

        call_count = 0

        async def counting_fetch_browse(filenum: str, count: int = 100) -> list:
            nonlocal call_count
            call_count += 1
            return []

        # mock _should_keep_sc13_direction → True
        async def fake_keep(*args: Any, **kwargs: Any) -> bool:
            return True

        async def fake_resolve_primary(*args: Any, **kwargs: Any) -> str:
            return "sc13g.htm"

        pipeline._downloader.fetch_browse_edgar_filenum = counting_fetch_browse  # type: ignore[assignment]
        pipeline._should_keep_sc13_direction = fake_keep  # type: ignore[assignment]
        pipeline._downloader.resolve_primary_document = fake_resolve_primary  # type: ignore[assignment]

        import datetime as dt
        form_windows = {"SC 13G": dt.date(2024, 1, 1)}
        filings = await pipeline._extend_with_browse_edgar_sc13(
            ticker="AAPL",
            filings=[],
            filenums={"005-67890"},
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="320193",
        )

        # HTTP 不应被调用
        assert call_count == 0
        # 缓存中的数据应被使用
        assert len(filings) >= 1

    async def test_non_005_filenums_skipped(self, tmp_path: Path) -> None:
        """非 005- 开头的 filenum 不触发 browse-edgar。"""
        pipeline = _make_pipeline(tmp_path)
        call_count = 0

        async def counting_fetch_browse(filenum: str, count: int = 100) -> list:
            nonlocal call_count
            call_count += 1
            return []

        pipeline._downloader.fetch_browse_edgar_filenum = counting_fetch_browse  # type: ignore[assignment]

        import datetime as dt
        form_windows = {"SC 13D": dt.date(2024, 1, 1)}
        filings = await pipeline._extend_with_browse_edgar_sc13(
            ticker="AAPL",
            filings=[],
            filenums={"001-12345", "002-67890"},  # 无 005- 前缀
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="320193",
        )

        assert call_count == 0
        assert filings == []


# ---------------------------------------------------------------------------
# _retry_sc13_if_empty — retry 复用缓存（不重复 HTTP）集成
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
class TestRetrySc13CacheReuse:
    """验证 SC13 渐进式回溯过程中 retry 自动复用已缓存的 history pages。"""

    async def test_retry_reuses_cache_no_extra_http(self, tmp_path: Path) -> None:
        """retry 时 history pages 应从缓存读取，不再发起 HTTP。"""
        pipeline = _make_pipeline(tmp_path)

        # 构造一个没有 SC13 的 submissions（触发 retry）
        history_data = _make_history_page(["10-K", "DEF 14A"])
        submissions = _make_submissions(
            recent_forms=["10-K"],
            history_files=[{"name": "CIK0000019617-submissions-001.json"}],
        )

        http_call_count = 0

        async def counting_fetch_json(url: str) -> dict[str, Any]:
            nonlocal http_call_count
            http_call_count += 1
            return history_data

        # browse-edgar 无 005- filenum（不触发）
        async def fake_fetch_browse(filenum: str, count: int = 100) -> list:
            return []

        pipeline._downloader.fetch_json = counting_fetch_json  # type: ignore[assignment]
        pipeline._downloader.fetch_browse_edgar_filenum = fake_fetch_browse  # type: ignore[assignment]

        import datetime as dt
        form_windows = {
            "10-K": dt.date(2024, 1, 1),
            "SC 13D": dt.date(2025, 1, 1),
            "SC 13D/A": dt.date(2025, 1, 1),
            "SC 13G": dt.date(2025, 1, 1),
            "SC 13G/A": dt.date(2025, 1, 1),
        }
        filenums: set[str] = set()

        # 先调用 _filter_filings 一次（触发 HTTP + 写入缓存）
        filings, filenums = await pipeline._filter_filings(
            ticker="AAPL",
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="19617",
        )
        initial_http = http_call_count
        assert initial_http == 1  # 首次 HTTP

        # 调用 _retry_sc13_if_empty（应触发 retry 但走缓存）
        filings = await pipeline._retry_sc13_if_empty(
            ticker="AAPL",
            filings=filings,
            filenums=filenums,
            submissions=submissions,
            form_windows=form_windows,
            end_date=dt.date(2026, 12, 31),
            target_cik="19617",
        )

        # retry 不应产生额外 HTTP 请求（全部从缓存读取）
        assert http_call_count == initial_http
