"""run.json 迁移：补齐 ``host_config.lane.write_chapter``。

2026-04 架构调整把写作流水线的并发上限从 in-process 常量
``_MIDDLE_CHAPTER_MAX_WORKERS`` 移交给 Host ``write_chapter`` lane。
旧工作区的 ``workspace/config/run.json`` 在 ``host_config.lane`` 下
没有该 key，启动期会缺失业务默认值。

本迁移只做一件事：在 ``host_config.lane`` 缺少 ``write_chapter`` 时补 5；
存在则一律尊重用户取值，绝不覆写。
"""

from __future__ import annotations

import json
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
        无：文件缺失、解析失败或结构不符预期均安静跳过，由上层决定是否告警。
    """

    run_json_path = config_dir / _RUN_JSON_FILENAME
    if not run_json_path.exists():
        return False

    try:
        raw_text = run_json_path.read_text(encoding="utf-8")
    except OSError:
        return False

    try:
        payload: Any = json.loads(raw_text)
    except json.JSONDecodeError:
        return False
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
    run_json_path.write_text(new_text, encoding="utf-8")
    return True
