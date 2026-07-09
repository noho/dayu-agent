# dayu/fins/storage 仓储层 deepreview prompt 模板

## 范围

- `dayu/fins/storage/` 全量
- `tests/fins/storage_testkit.py`
- `tests/fins/test_storage_batch_recovery.py`(1332 行)
- `tests/fins/test_storage_split_repositories.py`(732 行)
- 后续补的 6 个 `tests/fins/test_*_repository.py` spec test

## 补充约束(避免误报)

- 本模块是 `dayu.fins` 唯一财报文档存取真源(CLAUDE.md 明令)
- mixin 钻石继承(`_fs_storage_core.py`)对外保持 `FsStorageCore` 不变,内部按职责拆分 — 不要把"应再拆"当 finding
- `lease.py`/`pending_turn_store.py` 风格上"不抽 LeaseManager",storage 同款"不抽 Repository" — 不要把"应抽公共 Repository 基类"当 finding
- 跨平台文件锁 POSIX/Windows 实现差异在 `dayu/file_lock.py`,不在 storage 内 — 不要在 storage 内找 fcntl/msvcrt
- design_doc 在 `docs/fins_storage_design.md`,test_doc 在 `docs/fins_storage_test.md`,
  reviewer 必读后再审

## review 结果

```text
### 编号-未修复-[严重程度(低/中/高/严重)]-finding简述

**finding 类别**:[正确性 / 并发 / 安全 / 类型 / 设计 / 测试 / 性能 / 文档]

**位置**:
- `dayu/fins/storage/_fs_xxx_core.py:L123-L145`

**问题**:
(具体,挂源码段,反引现有测试名或测试空白)

**修复建议**:
(具体改法,涉及契约改动须先动 design_doc)

**回归挂点**:
- 新增 / 强化测试:`tests/fins/test_xxx.py::test_yyy_zzz`
- (若 design_doc 改) `docs/fins_storage_design.md §X`
```

## 审查要求(极其严格)

1. 对照 design_doc §3 「6 Protocol × N 不变量」逐条审查,缺一个不变量没钉死 spec test → finding
2. 对照 design_doc §5 「Batch 跨阶段算账」表逐 phase 审查,任何漏处理的 phase 转移 → finding
3. 对照 `code_review §6.9 grep -rn` 同款 — storage 内 **禁止兼容性代码** ,见 §9
4. 反引现有测试名;缺测试本身是 finding(对应 design_doc §测试空白与已知缺口)

## 检查项

### 1. 仓储协议契约一致

#### 1.1 6 Protocol method 签名与实现一致

- `repository_protocols.py` 内 6 Protocol 的每个 method 必须有对应 `fs_*_repository.py` 实现
- 装配层 method 必须**纯转发**(`self._repository_set.core.<method>(*args)`),例外仅 `FsSourceDocumentRepository` 按 `SourceKind` 分支(L308-328)
- 实现层任何分支逻辑必须在 `_fs_*_core.py` mixin 内,不在公开装配层

可执行检查:
```bash
grep -rn "def.*self.*-> " dayu/fins/storage/fs_*_repository.py | wc -l
# 应 ≈ Protocol 总 method 数 + __init__
```

#### 1.2 异常契约对齐

- `get_company_meta` / `get_source_meta` 缺失 → `FileNotFoundError`(不是 KeyError)
- meta 格式非法 → `ValueError`
- 底层 IO 失败 → `OSError`
- 句柄越界 → `ValueError("条目名称越界")`
- batch token 无效 → `ValueError("无效的 batch token")`
- 同 ticker 并发 batch → `RuntimeError`(分进程内 / 跨进程两个 message)
- lease 概念在 storage 内不复用(那是 host 的)

#### 1.3 装配漏洞回归

- 显式注入 `repository_set` 路径必须自动接 batch / cache 共享语义
- 不能比默认 `build_fs_repository_set()` 弱(同型 conv_memory B-06)

回归挂点测试:
- `test_storage_split_repositories::test_explicit_repository_set_shares_batch_visibility`(若无,**finding 测试空白**)

---

### 2. Batch 三阶段 ACID

#### 2.1 Journal 4 phase 不可跳跃

- `_PHASE_STARTED → _PHASE_BACKED_UP_TARGET → _PHASE_SWAPPED_TARGET → _PHASE_COMMITTED`
- 任何 commit 路径不能跳 phase
- crash 恢复决策必须基于 journal phase + 目录存在性,见 design_doc §5 表
- `_PHASE_SWAPPED_TARGET` 后写 journal 失败 **不再 rollback**(L246-249) —— 保留目标,只发 warn

可执行检查:
```bash
grep -rn "_PHASE_" dayu/fins/storage/_fs_storage_infra.py
# 应该出现:_PHASE_STARTED, _PHASE_BACKED_UP_TARGET, _PHASE_SWAPPED_TARGET, _PHASE_COMMITTED
```

回归挂点测试:
- `test_storage_batch_recovery::test_recover_started_batch_cleans_orphan_staging` L252
- `test_storage_batch_recovery::test_recover_orphan_batches_dry_run_is_non_destructive` L328

#### 2.2 dry_run 必须非破坏

- `recover_orphan_batches(dry_run=True)` 返回拟动作 tuple,**不动文件系统**
- 实测剧本 B 验证

#### 2.3 活跃 ticker 锁保护恢复

- recovery 遇活跃 ticker 锁必须 skip,**不能误删**
- 否则杀活跃进程数据

回归挂点:
- `test_storage_batch_recovery::test_recover_orphan_batches_skips_live_locked_batch` L553

#### 2.4 rollback 是补偿路径

- 写 `rolled_back` journal 失败也必须继续清理 staging 并释放 ticker 锁
- 在 `_execute_with_auto_batch` 内 rollback 失败只能附注暴露,**不能覆盖原始写操作异常**
- 参 fins/README §7 第二条 storage 边界

---

### 3. 跨进程并发与文件锁

#### 3.1 ticker 锁非阻塞 + 跨平台

- `begin_batch` 必须用 `dayu/file_lock.py` 的 `acquire_text_file_lock(blocking=False)`
- POSIX 走 `fcntl.flock`,Windows 走 `msvcrt.locking`
- **Windows 上不能退化为重试**(否则在高并发下会假阻塞)

#### 3.2 recovery 全局锁 blocking

- `recover_orphan_batches` 必须用全局恢复锁 `blocking=True`
- Windows 上必须真等到锁,**不能偷换成 `LK_LOCK` 有限重试**
- 参 fins/README §7 第二条 storage 边界(L509)

#### 3.3 ticker 锁释放语义边界

- `transaction.json` 中残留 `owner_pid` **不能**当成"仍持锁"充分条件
- 真正权威是 ticker 锁文件本身

回归挂点:
- `test_storage_batch_recovery::test_recover_treats_freed_lock_with_stale_owner_pid_as_orphan` (若无,**finding**)

---

### 4. 安全边界

#### 4.1 Handle 越界三层校验

校验链(`_resolve_handle_child_path` L761 + `_normalize_entry_name` utils L81-101):
1. 空字符串拒绝
2. `.` / `..` 拒绝
3. `/` / `\` 路径分隔符拒绝
4. 拼接后用 `candidate.relative_to(base_dir)` 终极校验

任一漏 → P0 安全 bug。

可执行检查:
```bash
grep -rn "relative_to\|raise.*ValueError.*越界" dayu/fins/storage/
```

回归挂点测试(必须有):
- `test_blob_handle_path_traversal_rejected` 覆盖 6 类越界(`../etc/passwd` / `/etc/passwd` /
  `..\\windows\\system32` / `""` / `.` / `..`)
- 若无 → **测试空白 finding,P0**

#### 4.2 越界绝不退化为 FileNotFoundError

- 任何越界 attack 经 normalize 后**不能落到合法但不存在路径** → 抛 `FileNotFoundError`
- 必须早 fail 在 normalize 层,**抛 ValueError**

---

### 5. 文档生命周期(三态可逆 + 物理重置幂等)

#### 5.1 create / delete / restore / reset 语义边界

- `delete` 设 `is_deleted=True`,保留目录 + meta
- `restore` 设 `is_deleted=False`,字段**逐字段恢复 = delete 前快照**
- `reset` 物理删除目录 + manifest 条目,**幂等可重入**
- reset 后 delete/restore 都失去意义(目标已 GONE),如调用应安静幂等不抛

#### 5.2 reset 幂等三态

reset 必须对以下三状态都不抛:
- 目录存在 + manifest 有 → 清理两者
- 目录不存在 + manifest 无 → noop
- 目录已被外部删除 + manifest 残留 → 清 manifest

回归挂点:
- `test_source_document_repository_reset_source_document_tolerates_missing_target` L707 ✓

---

### 6. 公司元数据治理

#### 6.1 ticker alias 冲突显式

- `resolve_existing_ticker(candidates)` 多公司含相同 alias **必抛 `ValueError("命中多个公司目录")`**
- 严禁静默返首个匹配(数据完整性陷阱)

回归挂点:
- `test_company_meta_repository_resolves_existing_ticker_via_alias_and_detects_conflicts` L122 ✓

#### 6.2 scan_inventory 三态

- `available / missing_meta / hidden_directory` 三态完整
- `.dayu/` / `.staging/` / `.backup/` 等隐藏目录必标 `hidden_directory`,**不能漏标为 missing_meta**(会让 inventory 把内部目录当公司)

回归挂点:
- `test_company_meta_repository_scan_inventory_records_skipped_directories` L77 ✓

#### 6.3 ticker 规范化降级

- `_normalize_ticker` 先调 `try_normalize_ticker()`
- 失败回退 `.strip().upper()`,**不能直接抛** —— 仓储宽容,业务校验上层
- 见 utils L48-51

---

### 7. processed / maintenance 测试空白(P0)

design_doc §测试空白与已知缺口逐条审查:

| 缺口 | 严重度 | 必须补救的测试名 |
|---|---|---|
| `clear_processed_documents` 无测试 | P0 | `test_processed_clear_removes_all_and_idempotent` |
| `delete_entry` 无测试 | P0 | `test_blob_delete_entry_rejects_missing_and_directory` |
| `handle 越界 ValueError` 装配层无 | P0 | `test_blob_handle_path_traversal_rejected`(6 类) |
| `cleanup_stale_filing_documents` 返回值 | P1 | `test_maintenance_cleanup_returns_count_for_complement_of_active_sets` |
| `clear_filing_documents` 无测试 | P1 | `test_maintenance_clear_filing_does_not_touch_processed` |
| `get_processed_handle` 不存在文档行为 | P2 | `test_processed_get_handle_for_missing_document` |

任一缺失 → finding。

---

### 8. 与 host 模块的层级边界

#### 8.1 storage 不依赖 host

- `dayu/fins/storage/` 不得 `import dayu.host.*`
- 反向 host 可依赖 storage(但走的是 protocol,非具体实现)

可执行检查:
```bash
grep -rn "from dayu.host\|import dayu.host" dayu/fins/storage/
# 应该为空
```

#### 8.2 storage 不依赖 LLM / agent

- 不得 `import dayu.engine.agent` / `dayu.host.executor`

```bash
grep -rn "from dayu.engine\|from dayu.host" dayu/fins/storage/
# 仅允许 `dayu.engine.processors.source.Source`(纯类型)
```

---

### 9. 兼容性代码 grep -rn 应返回空

CLAUDE.md 「禁止兼容性代码」可执行检查:

```bash
# 兼容性 re-export
grep -rn "^from .* import .*  # re-export" dayu/fins/storage/
# 兼容性常量
grep -rn "# 兼容\|# compat\|# legacy" dayu/fins/storage/
# 兼容性 wrapper / facade
grep -rn "def.*self.*:\s*\n.*self\._impl\." dayu/fins/storage/
# 任何输出非空都是 finding
```

#### 9.1 旧 DocumentRepository 不应残留

- 旧 `DocumentRepository / FsDocumentRepository` 已删除(参 fins/README §7 L493)
- grep 应返回空:
```bash
grep -rn "class DocumentRepository\|class FsDocumentRepository" dayu/
```

---

### 10. 类型检查与 pyright

- 所有 protocol method 必须有完整 type hint(Optional / 联合类型显式)
- 禁止 `Any` / `object` 偷懒
- `hasattr / getattr` 只在必要时用,docstring 必须解释 why
- 装配层 `__init__` 必须接受具名 `repository_set: _FsRepositorySet`

可执行检查:
```bash
source .venv/bin/activate
pyright dayu/fins/storage/ 2>&1 | tail -20
# 不得有 new error,不得扩散
```

---

### 11. 反引测试名审查(缺测试本身是 finding)

reviewer 必须确认以下测试名实际存在(grep test_xxx):

**已存在(必查命中)**:
- `test_storage_batch_recovery::test_batch_paths_move_under_dayu_root`
- `test_storage_batch_recovery::test_begin_batch_rejects_same_ticker_in_same_process`
- `test_storage_batch_recovery::test_begin_batch_rejects_cross_process_same_ticker`
- `test_storage_batch_recovery::test_begin_batch_allows_different_ticker_while_other_process_holds_lock`
- `test_storage_batch_recovery::test_commit_batch_rejects_invalid_token`
- `test_storage_batch_recovery::test_rollback_batch_rejects_invalid_token`
- `test_storage_batch_recovery::test_recover_orphan_batches_dry_run_is_non_destructive`
- `test_storage_batch_recovery::test_recover_started_batch_cleans_orphan_staging`
- `test_storage_batch_recovery::test_recover_orphan_batches_skips_live_locked_batch`
- `test_storage_split_repositories::test_company_meta_repository_roundtrip`
- `test_storage_split_repositories::test_company_meta_repository_resolves_existing_ticker_via_alias_and_detects_conflicts`
- `test_storage_split_repositories::test_company_meta_repository_scan_inventory_records_skipped_directories`
- `test_storage_split_repositories::test_source_document_repository_supports_generic_create_delete_restore`
- `test_storage_split_repositories::test_source_document_repository_reset_source_document_tolerates_missing_target`
- `test_storage_split_repositories::test_source_document_repository_directory_and_filing_checks_cover_errors`
- `test_storage_split_repositories::test_source_document_repository_can_resolve_primary_file_and_source`
- `test_storage_split_repositories::test_document_blob_repository_can_store_and_read_source_file`
- `test_storage_split_repositories::test_processed_document_repository_uses_explicit_processed_listing`
- `test_storage_split_repositories::test_filing_maintenance_repository_persists_rejection_registry`
- `test_storage_split_repositories::test_filing_maintenance_repository_roundtrip_rejected_artifact`

**必须新增(缺则 finding)**:
- `test_blob_handle_path_traversal_rejected`(6 类越界)
- `test_processed_clear_removes_all_and_idempotent`
- `test_blob_delete_entry_rejects_missing_and_directory`
- `test_maintenance_cleanup_returns_count_for_complement_of_active_sets`
- `test_maintenance_clear_filing_does_not_touch_processed`

---

### 12. 性能(静态分析,无需 profiling)

- `list_processed_documents` 等列表操作不得在装配层做 N+1(每文档一次仓储调用)
- batch journal 写盘必须批量,而非每个 entry 一次 fsync
- handle 解析路径应缓存而非每次 stat()

---

## 产出要求

每条 finding 必须:

1. 编号 + 严重程度 + 是否未修复 + 一句话简述
2. **挂源码行号**(`_fs_xxx_core.py:L1234-L1245`)
3. **反引现有测试名**或明示**测试空白 P0/P1**
4. 与 design_doc / test_doc 中对应章节交叉引用
5. 修复建议挂回归测试名(可以是建议名,在 PR 中实创)

reviewer 反例(应当避免):
- 「应该增加测试覆盖」 → ✗ 太泛
- 「`test_blob_handle_path_traversal_rejected` 缺失,需覆盖 6 类越界 attack,挂 `_resolve_handle_child_path` L761」 → ✓

## 与 conversation_memory phase code_review.md 的同型对照

| 维度 | conv_memory §6 | storage §1-12 |
|---|---|---|
| 测试反引深度 | §6.12 反引 11 测试名 | §11 反引 20 测试名 + 5 必新增 |
| grep -rn 可执行 | §6.9 移除字段硬切换 | §9 兼容性代码 |
| 不变量逐条 | §6.1-6.11 | §1.1-1.3 / §2.1-2.4 / §5.1-5.2 / §6.1-6.3 |
| 缺测试本身是 finding | 是 | 是 |
| 业务语境绑技术决策 | 财报对话 pinned_state | 财报存取 batch ACID |
