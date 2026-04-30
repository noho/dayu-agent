"""run.json 迁移：补齐 ``host_config.lane.write_chapter``。

2026-04 架构调整把写作流水线的并发上限从 in-process 常量
``_MIDDLE_CHAPTER_MAX_WORKERS`` 移交给 Host ``write_chapter`` lane。
旧工作区的 ``workspace/config/run.json`` 在 ``host_config.lane`` 下
没有该 key，启动期会缺失业务默认值。

本迁移只做一件事：在 ``host_config.lane`` 缺少 ``write_chapter`` 时补 5；
存在则一律尊重用户取值，绝不覆写。

写入采用「写临时文件 + ``os.replace`` 原子替换」：在 init 持有 workspace
advisory lock 的前提下额外抵御进程被强制终止时残留半截写入的边界情况。
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


_RUN_JSON_FILENAME = "run.json"
_HOST_CONFIG_KEY = "host_config"
_LANE_KEY = "lane"
_WRITE_CHAPTER_LANE = "write_chapter"
_DEFAULT_WRITE_CHAPTER_CONCURRENCY = 5


def migrate_run_json_add_write_chapter_lane(config_dir: Path) -> bool:
    """为旧工作区的 ``run.json`` 补齐 ``write_chapter`` lane 默认值。

    Args:
        config_dir: 工作区配置目录，即 ``workspace/config``。

    Returns:
        True 表示实际改写了文件；False 表示无需变更或文件不存在。

    Raises:
        OSError: 读取或写入 ``run.json`` 失败时抛出，由 init 命令显式失败。
        json.JSONDecodeError: ``run.json`` 既存但 JSON 解析失败；不再吞错，
            由上层决定是否继续。
    """

    run_json_path = config_dir / _RUN_JSON_FILENAME
    if not run_json_path.exists():
        return False

    raw_text = run_json_path.read_text(encoding="utf-8")
    payload: Any = json.loads(raw_text)
    if not isinstance(payload, dict):
        return False

    host_config = payload.get(_HOST_CONFIG_KEY)
    if not isinstance(host_config, dict):
        return False

    lane_section = host_config.get(_LANE_KEY)
    if not isinstance(lane_section, dict):
        return False

    if _WRITE_CHAPTER_LANE in lane_section:
        return False

    lane_section[_WRITE_CHAPTER_LANE] = _DEFAULT_WRITE_CHAPTER_CONCURRENCY
    new_text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(run_json_path, new_text)
    return True


def _atomic_write_text(target_path: Path, content: str) -> None:
    """使用临时文件 + ``os.replace`` 原子替换 ``target_path``。

    Args:
        target_path: 目标文件绝对路径。
        content: 待写入文本。

    Returns:
        无。

    Raises:
        OSError: 写入或重命名失败时抛出。
    """

    temp_path = target_path.with_name(f".{target_path.name}.migrate.tmp")
    try:
        with open(temp_path, "w", encoding="utf-8") as fp:
            fp.write(content)
            fp.flush()
            os.fsync(fp.fileno())
        os.replace(temp_path, target_path)
    finally:
        # 仅清理悬挂临时文件；replace 成功后该路径已不存在，unlink 会 ENOENT。
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass
