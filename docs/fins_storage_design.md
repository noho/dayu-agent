# fins/storage 仓储层设计文档

## 背景

`dayu/fins/storage/` 已经是 dayu agent 的财报文档存取**唯一真源**——CLAUDE.md
明令「财报文档存取必须且只能通过 `dayu.fins.storage` 下的仓储协议与仓储实现完成」。
但当前**设计意图散落在三处**:

- 6 个 Protocol 的 docstring 各写一小段(`repository_protocols.py` 337 行)
- `_fs_storage_core.py` 的 mixin 钻石继承注释
- 散落在 1580 行 `_fs_storage_infra.py` 内部注释里的 batch 三阶段 commit / journal 4 phase / 两层文件锁 ACID 设计

这导致两个工程问题:

1. **新成员读 5254 行 core 找不到入口** —— 没有顶层 invariants 索引
2. **测试只覆盖 batch_recovery + split,逐 protocol 不变量级 spec 缺** ——
   单个 protocol 改一个字段没有显式回归网

本文档把 6 Protocol × N 不变量、batch ACID 状态机、跨进程并发治理一次性
固化为**契约真源**,与 `dayu/host/README.md §10` 同位等阶。

## 设计原则

### 一、Repository 窄协议(已成形)

按真实职责簇拆分 6 个 Protocol,不抽象单一 `Repository`:

| Protocol | 职责 | 行 |
|---|---|---|
| `BatchingRepositoryProtocol` | 批处理事务边界(begin/commit/rollback/recover) | 7 method |
| `CompanyMetaRepositoryProtocol` | 公司级元数据(scan/get/upsert/resolve_ticker) | 4 method |
| `SourceDocumentRepositoryProtocol` | 源文档 CRUD + 三态(create/update/delete/reset/restore) | 14 method |
| `ProcessedDocumentRepositoryProtocol` | processed 产物 CRUD + reprocess 标记 | 8 method |
| `DocumentBlobRepositoryProtocol` | 文件对象读写(handle 级越界校验) | 5 method |
| `FilingMaintenanceRepositoryProtocol` | 拒绝注册表 + 清理治理 | 9 method |

**理由**:与 `dayu/host/lease.py` 的「不抽 LeaseManager」同型——
batch 事务、公司元数据、文档 CRUD、blob、维护治理状态机不同,
**硬抽 `Repository` 公共层只会制造胶水**。

### 二、共享 core 装配(`_fs_repository_factory.py`)

```python
def build_fs_repository_set(*, workspace_root, file_store=None,
                            repository_set=None, create_directories=True) -> _FsRepositorySet
```

- **6 个公开窄仓储共享同一 `FsStorageCore` 实例** —— 否则多 repo 各自 init 会
  破坏 batch/cache 共享语义(同一进程内同一 ticker 不能在 repo A 看到 batch
  但 repo B 看不到)
- `_FsRepositorySet` 是 `@dataclass(frozen=True)`,持有 `core: FsStorageCore`
- 这是与 `conversation_memory` phase「显式装配不能比默认弱」同型契约 ——
  显式注入路径必须自动接 batch 屏障

### 三、文件系统 ACID(`_fs_storage_infra.py`)

文件系统天然无事务,通过 **5 套机制** 模拟 ACID:

1. **Staging / target 读写分离** —— batch 活跃时 read/write 都走
   `staging_ticker_dir`,无 batch 走 `target_ticker_dir`(L1115-1152)
2. **三阶段 commit** —— 备份 → 交接 → 删除备份(L224-242)
3. **Journal phase 4 转移** —— `_PHASE_STARTED → _PHASE_BACKED_UP_TARGET →
   _PHASE_SWAPPED_TARGET → _PHASE_COMMITTED`(L188 起)
4. **原子 JSON 写入** —— temp file + fsync + atomic replace + dir fsync(utils L439-463)
5. **两层文件锁** —— ticker 级 non-blocking(`begin_batch`) +
   global 恢复锁 blocking(`recover_orphan_batches`)(L443-493 / L369)

## 6 Protocol × N 不变量(spec test 必须钉死)

### BatchingRepositoryProtocol(7 不变量)

1. **唯一 token 结构** —— `begin_batch` 返回 BatchToken 必含
   `token_id (UUID) / ticker / staging_root_dir / ticker_lock_path`(必存且可读写)
   - test:`test_batch_paths_move_under_dayu_root` L240
2. **同进程同 ticker 重复拒绝** —— `RuntimeError("已存在活动 batch")`
   - test:`test_begin_batch_rejects_same_ticker_in_same_process` L707
3. **跨进程同 ticker 拒绝** —— `ticker.lock` 文件锁 non-blocking,争用抛
   `RuntimeError("跨进程活动 batch")`
   - test:`test_begin_batch_rejects_cross_process_same_ticker` L534
4. **不同 ticker 可并发** —— 锁粒度=ticker,非全局
   - test:`test_begin_batch_allows_different_ticker_while_other_process_holds_lock` L575
5. **无效 token 拒绝** —— commit/rollback 对不在 `_active_batches` 的 token
   抛 `ValueError("无效的 batch token")`
   - test:`test_commit_batch_rejects_invalid_token` L721 / `test_rollback_batch_rejects_invalid_token` L754
6. **dry_run 非破坏** —— `recover_orphan_batches(dry_run=True)` 返回拟动作列表
   但不动文件系统
   - test:`test_recover_orphan_batches_dry_run_is_non_destructive` L328
7. **活跃 ticker 锁保护恢复** —— recovery 遇活跃 ticker 锁必须 skip(防误删)
   - test:`test_recover_orphan_batches_skips_live_locked_batch` L553

### CompanyMetaRepositoryProtocol(5 不变量)

1. **缺失 / 损坏 / IO 失败的三套异常分流** —— `FileNotFoundError` / `ValueError` / `OSError`
   - Protocol L68-80
2. **upsert 幂等** —— 重复写覆盖,语义不变
   - test:`test_company_meta_repository_roundtrip` L50
3. **resolve_existing_ticker 冲突显式** —— 多公司含相同 alias 时抛
   `ValueError("命中多个公司目录")`,**不返回首个匹配**
   - test:`test_company_meta_repository_resolves_existing_ticker_via_alias_and_detects_conflicts` L122
4. **scan_inventory 三态** —— `status` 必为 `available / missing_meta / hidden_directory`,
   `.dayu/` 必标 `hidden_directory`
   - test:`test_company_meta_repository_scan_inventory_records_skipped_directories` L77
5. **装配纯转发** —— `FsCompanyMetaRepository` 6 method 全部 `self._repository_set.core.<method>(*args)`,
   无参数变换、无条件分支

### SourceDocumentRepositoryProtocol(6 不变量)

1. **source_kind 分支装配** —— create/update 在装配层按 `SourceKind` 分支到
   `core.create_filing` / `create_material` —— **装配层唯一含分支的仓储**
   - L308-328
2. **delete / restore 逻辑可逆** —— `is_deleted` 字段切换,二者均可重复调用
   - test:`test_source_document_repository_supports_generic_create_delete_restore` L268
3. **reset 物理重置幂等** —— 目录 + manifest 条目清,目标不存在保持幂等不抛
   - test:`test_source_document_repository_reset_source_document_tolerates_missing_target` L707
4. **has_source_storage_root** —— 路径非目录抛 `NotADirectoryError("source root")`
   - test:`test_source_document_repository_directory_and_filing_checks_cover_errors` L587
5. **has_filing_xbrl_instance 二态校验** —— 目录不存在 `FileNotFoundError`,
   路径是文件 `NotADirectoryError`
6. **get_source / get_primary_source 走 list_files** —— 不直接读盘,通过
   `_find_file_meta_by_filename` 定位
   - L406-417

### ProcessedDocumentRepositoryProtocol(4 不变量,**3 测试空白**)

1. **mark_reprocess_required(required=False) 是空操作** —— 装配层显式
   `if not required: return`,只在 `required=True` 时调 core(L89-94)
2. **list_processed_documents 走 query 过滤** —— `DocumentQuery` 条件下传
   - test:`test_processed_document_repository_uses_explicit_processed_listing` L420
3. **create/update 返回 DocumentHandle,delete 返回 None** —— 见 Protocol L191-201
4. **clear_processed_documents 幂等** —— **⚠️ 当前测试空白**,需补 spec
5. **(空白)`get_processed_handle` 对不存在文档** —— 行为未测

### DocumentBlobRepositoryProtocol(4 不变量,**2 测试空白**)

1. **store_file / read_file_bytes 字节级 roundtrip** —— `b"hello world"` 入 → 出等值
   - test:`test_document_blob_repository_can_store_and_read_source_file` L289
2. **uri ↔ filename 可逆** —— `_infer_filename_from_uri` 能从 `FileObjectMeta.uri`
   反推 filename(L218-239)
3. **list_entries(直系) vs list_files(文件元数据) 区分** —— 一个返目录条目
   一个返 FileObjectMeta
4. **delete_entry 异常契约** —— **⚠️ 当前测试空白**,目标不存在行为未钉
5. **(空白)handle 越界 ValueError** —— `..` / `/` / `\` 在 `_normalize_entry_name`
   被拒,但**仓储装配层未测**

### FilingMaintenanceRepositoryProtocol(5 不变量,**3 测试空白**)

1. **rejection registry 二层 dict** —— `dict[str, dict[str, str]]` 结构,
   load/save 形成读写对
   - test:`test_filing_maintenance_repository_persists_rejection_registry` L453
2. **upsert_rejected_filing_artifact 覆盖** —— 同 `(ticker, document_id)` 后值覆盖前值
   - test:`test_filing_maintenance_repository_roundtrip_rejected_artifact` L477
3. **list_rejected_filing_artifacts 不跨 ticker** —— 严格按 ticker 隔离
4. **cleanup_stale_filing_documents 返回清理条数** —— **⚠️ 测试空白**,
   `active_form_types` ∩ `valid_document_ids` 补集语义未显式测
5. **clear_filing_documents 与 delete_processed 职责分离** —— **⚠️ 测试空白**,
   两者操作不同目录(filings vs processed),不能互相覆盖

## 字段表

| 字段 | 类型 | 来源 | 不变量 |
|---|---|---|---|
| `BatchToken.token_id` | UUID hex | `begin_batch` | 全局唯一 |
| `BatchToken.ticker` | str(规范化) | `begin_batch(ticker)` | `_normalize_ticker` 处理 |
| `BatchToken.staging_root_dir` | Path | `_target_ticker_dir.parent / .staging` | 与 target 在同一 portfolio 下 |
| `BatchToken.ticker_lock_path` | Path | `.dayu/locks/{ticker}.lock` | OS file lock 文件 |
| `BatchToken.journal_path` | Path | `.dayu/repo_batches/{token_id}/journal.json` | crash recovery 真源 |
| `CompanyMeta.ticker` | str(规范化) | upsert 调用方 | upsert 写入时再次 normalize |
| `CompanyMeta.aliases` | tuple[str,...] | LLM/用户输入 | resolve_existing_ticker 多公司含相同 alias 抛 ValueError |
| `DocumentMeta.is_deleted` | bool | delete/restore 切换 | reset 后该字段失效 |
| `SourceHandle.source_kind` | SourceKind | 调用方传入 | 装配层按此分支 filing/material |
| `ProcessedHandle.document_id` | str | create_processed 生成 | `mark_reprocess_required` 改不动 |
| `RejectedFilingArtifact.upserted_at` | str | upsert 时盖戳 | 同 (ticker, document_id) 覆盖语义 |
| `FileObjectMeta.uri` | str | store_file 生成 | `_infer_filename_from_uri` 可逆 |

## Batch 跨阶段算账(journal × 目录存在性 → 恢复动作)

| journal phase | backup 存在 | target 存在 | recovery 动作 | 出处 |
|---|---|---|---|---|
| `_PHASE_STARTED` | 否 | 是 | 删 staging,目标已是稳定态 | L188 |
| `_PHASE_BACKED_UP_TARGET` | 是 | 否 | restore backup → target | L626 |
| `_PHASE_SWAPPED_TARGET` | 是 | 是 | delete backup(交接已完成) | L631 |
| 未知 / 损坏 | 是 | 否 | restore backup → target(保守) | L635-639 |
| 未知 / 损坏 | 是 | 是 | delete backup(保守) | L640-643 |
| `_PHASE_COMMITTED` | 否 | 是 | noop,清 staging 即可 | — |

**关键设计**:`_PHASE_SWAPPED_TARGET` 后若写 journal 失败,**不再 rollback** ——
保留目标目录,只发 warn 日志(L246-249)。理由:journal 丢失不能损害数据真源。

## 跨档算账:文档生命周期三态

`SourceDocument` 三态:

```text
        create_source_document
              ↓
        ┌─────────┐
        │ EXISTS  │ ←─────────┐
        └────┬────┘           │
             │ delete         │ restore
             ↓                │
        ┌──────────────┐      │
        │ LOGICALLY    │──────┘
        │ DELETED      │
        └──────┬───────┘
               │ reset(物理重置)
               ↓
        ┌──────────────┐
        │ GONE(目录无) │ ← reset 也可从 EXISTS 直接到
        └──────────────┘
```

- **delete** —— `is_deleted=True`,保留目录与 meta
- **restore** —— `is_deleted=False`,撤销逻辑删除
- **reset** —— 物理删除目录 + manifest 条目,**幂等可重入**(目标不存在不抛)

这与 conversation_memory phase 的 `PartiallyApplied` 4 二阶断言同型:
**失败后语义不留半生不熟**(reset 后 delete/restore 都失去意义)。

## 调参面(已成形的可注入项)

| 参数 | 默认 | 调参语境 |
|---|---|---|
| `workspace_root` | 必填 | 单实例锁绑定的 workspace |
| `file_store` | `LocalFileStore` 自动构建 | 测试或云存储替换 |
| `repository_set` | None(新建) | 已有 set 直接复用,**避免破坏 batch 共享** |
| `create_directories` | True | 测试环境可关 |
| `BatchingRepositoryProtocol.recover_orphan_batches(dry_run)` | False | 排障时设 True 看拟动作 |
| `FilingMaintenanceRepositoryProtocol.cleanup_stale_filing_documents(active_form_types, valid_document_ids)` | 调用方传集合 | active set 越小清理越激进 |

## 测试空白与已知缺口

| 缺口 | 严重度 | 建议补救 |
|---|---|---|
| `clear_processed_documents` 无测试 | P0 | 补 `test_processed_clear_removes_all_and_idempotent` |
| `delete_entry` 异常契约无测试 | P0 | 补 `test_blob_delete_entry_rejects_missing_and_directory` |
| `handle 越界 ValueError` 装配层无测试 | P0 | 补 `test_blob_handle_path_traversal_rejected`(尝试 `../../etc/passwd`) |
| `cleanup_stale_filing_documents` 返回值无测试 | P1 | 补 `test_maintenance_cleanup_returns_count_for_complement_of_active_sets` |
| `clear_filing_documents` 无测试 | P1 | 补 `test_maintenance_clear_filing_does_not_touch_processed` |
| `get_processed_handle` 对不存在文档 | P2 | 补行为锚定测试 |

## 与 conversation_memory phase 的同型契约

| 维度 | conversation_memory | fins/storage |
|---|---|---|
| 公共原语极简 | `lease.py` 32 行 | `_fs_repository_factory.py` 57 行 |
| 拒绝胶水抽象 | 不抽 LeaseManager | 不抽 Repository |
| 双条件 CAS | SQL `state + lease_id` | journal phase + 目录存在性 |
| 装配漏洞回归 | 显式注入接 session_registry 屏障 | 显式 `repository_set` 接共享 core |
| 失败语义不半成品 | `PartiallyApplied` 4 二阶断言 | reset / restore / delete 三态 |
| 跨进程基础设施 | 真 subprocess + WAL shm | OS file lock + journal recovery |
| 字段重命名表达层分离 | `assistant_final → assistant_text` | `staging_dir → target_dir`(三阶段交接) |

下一阶段 README §X 与 spec test 必须钉死这套对应关系。
