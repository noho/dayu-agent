"""run.json 迁移：补齐 A 股与港股下载业务 lane。

CN/HK 财报下载接入 Host direct operation 后，需要与美股
``sec_download`` 一样通过 ``host_config.lane`` 配置跨进程并发上限。
旧工作区的 ``workspace/config/run.json`` 没有 ``cn_download`` 与
``hk_download``，会导致服务启动时只能依赖内置默认值，用户配置文件
无法显式呈现当前 schema。

本迁移只在缺少 key 时补默认值 1；已存在的用户取值一律保留。
"""

from __future__ import annotations

from pathlib import Path

from dayu.cli.workspace_migrations.run_json_utils import (
    as_json_object,
    load_json_value,
    write_json_value,
)


_RUN_JSON_FILENAME = "run.json"
_HOST_CONFIG_KEY = "host_config"
_LANE_KEY = "lane"
_CN_DOWNLOAD_LANE = "cn_download"
_HK_DOWNLOAD_LANE = "hk_download"
_DEFAULT_MARKET_DOWNLOAD_CONCURRENCY = 1


def migrate_run_json_add_market_download_lanes(config_dir: Path) -> bool:
    """为旧工作区的 ``run.json`` 补齐 CN/HK 下载 lane 默认值。

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

    payload = load_json_value(run_json_path)
    payload_obj = as_json_object(payload)
    if payload_obj is None:
        return False

    host_config = as_json_object(payload_obj.get(_HOST_CONFIG_KEY))
    if host_config is None:
        return False

    lane_section = as_json_object(host_config.get(_LANE_KEY))
    if lane_section is None:
        return False

    changed = False
    if _CN_DOWNLOAD_LANE not in lane_section:
        lane_section[_CN_DOWNLOAD_LANE] = _DEFAULT_MARKET_DOWNLOAD_CONCURRENCY
        changed = True
    if _HK_DOWNLOAD_LANE not in lane_section:
        lane_section[_HK_DOWNLOAD_LANE] = _DEFAULT_MARKET_DOWNLOAD_CONCURRENCY
        changed = True

    if not changed:
        return False

    write_json_value(run_json_path, payload)
    return True
