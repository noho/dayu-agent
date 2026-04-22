"""工作区一次性迁移脚本集合。

该包集中存放 `dayu-cli init` 在幂等检查阶段需要执行的、
针对旧工作区的一次性修复脚本。放这里而非 ``dayu/cli/commands/init.py``
的目的：

- 让这类"未来会累积、随时可能调整"的迁移规则与 init 常规流程解耦；
- 每条迁移单独一个模块，便于单独测试、单独评估是否保留；
- init 入口只调用 :func:`apply_all_workspace_migrations`，不再承载具体规则。

迁移函数必须满足：

- **幂等**：重复执行必须等价于执行一次；
- **保守**：只在检测到旧结构时才改写，不触碰用户自定义；
- **只向前**：不做旧 schema 兼容读取，只把旧数据原地改成当前 schema。
"""

from __future__ import annotations

from dayu.cli.workspace_migrations.runner import apply_all_workspace_migrations


__all__ = ["apply_all_workspace_migrations"]
