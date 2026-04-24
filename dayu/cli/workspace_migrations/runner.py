"""工作区迁移统一入口。

所有一次性修复脚本在这里串联。新增规则只需：

1. 在 ``dayu/cli/workspace_migrations/`` 下新建模块；
2. 暴露一个幂等函数；
3. 在 :func:`apply_all_workspace_migrations` 里按顺序调用并打印结果。

**不要**把规则写回 ``init.py``；也不要在此文件里做除"调度 + 汇报"以外的事。
"""

from __future__ import annotations

from pathlib import Path

from dayu.cli.workspace_migrations.host_store_rename_concurrency_lane import (
    migrate_host_store_rename_concurrency_lane,
)
from dayu.cli.workspace_migrations.host_store_strip_max_output_tokens import (
    migrate_host_store_strip_max_output_tokens,
)
from dayu.cli.workspace_migrations.run_json_write_chapter_lane import (
    migrate_run_json_add_write_chapter_lane,
)
from dayu.workspace_paths import build_host_store_default_path


def apply_all_workspace_migrations(*, base_dir: Path, config_dir: Path) -> None:
    """按顺序执行全部已登记的工作区迁移，并把结果打印到 stdout。

    Args:
        base_dir: 工作区根目录，用于解析 Host SQLite 数据库位置。
        config_dir: 工作区配置目录（通常是 ``base_dir / "config"``）。

    Returns:
        无。

    Raises:
        无：单条迁移失败不应阻塞 init 主流程，异常由各迁移自行吞掉。
    """

    if migrate_run_json_add_write_chapter_lane(config_dir):
        print("✓ 工作区迁移: run.json 已补齐 host_config.lane.write_chapter=5")

    host_db_path = build_host_store_default_path(base_dir)
    rewritten = migrate_host_store_rename_concurrency_lane(host_db_path)
    if rewritten > 0:
        print(
            "✓ 工作区迁移: Host SQLite pending turn 快照 "
            f"concurrency_lane → business_concurrency_lane（共 {rewritten} 行）"
        )

    stripped = migrate_host_store_strip_max_output_tokens(host_db_path)
    if stripped > 0:
        print(
            "✓ 工作区迁移: Host SQLite pending turn 快照已剥离 "
            f"max_output_tokens 字段（共 {stripped} 行）"
        )
