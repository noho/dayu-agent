# fins/storage 仓储实测剧本

## 目标

实测验证 6 Protocol × 30 条不变量 + batch 三阶段 ACID 在真实工作区上的表现。
重点观察四件事:

1. **batch 三阶段 commit 在 crash 后能否幂等恢复**(journal × 目录存在性)
2. **跨进程同 ticker 锁是否真的非阻塞拒绝**(POSIX fcntl / Windows msvcrt)
3. **source 文档三态(create → delete → restore → reset)语义可逆 + reset 幂等**
4. **handle 越界校验**(`../`、`/`、`\` 拒绝)

启动方式:

```bash
source .venv/bin/activate
python -m dayu.cli init --base /tmp/dayu_storage_test
cd /tmp/dayu_storage_test
```

每组测试结束后,结合 sqlite3 / 文件系统快照 / journal.json 交叉确认。

---

## 测试组 A:Batch 三阶段 commit 一致性

**目的**:验证 `begin_batch → 写 staging → commit_batch` 三阶段交接后,
读路径能立即看到 staging 内容(L1115-1152 staging/target 切换)。

观察项:
- A-1 begin 后 read 走 staging
- A-3 commit 后 read 走 target
- **A-3 = A-1 数据内容一致**(只是路径切换,无丢失)

```python
from pathlib import Path
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from dayu.fins.storage.fs_source_document_repository import FsSourceDocumentRepository
from dayu.fins.domain.document_models import SourceDocumentUpsertRequest
from dayu.fins.domain.enums import SourceKind

ws = Path("/tmp/dayu_storage_test")
repo_set = build_fs_repository_set(workspace_root=ws)
src_repo = FsSourceDocumentRepository(repository_set=repo_set)
batch_repo = ...  # 通过同 repo_set 装配

# A-1
token = batch_repo.begin_batch("AAPL")
handle = src_repo.create_source_document(
    SourceDocumentUpsertRequest(ticker="AAPL", document_id="fil_test_001", ...),
    source_kind=SourceKind.FILING,
)
# 读取应走 staging
meta_during_batch = src_repo.get_source_meta("AAPL", "fil_test_001", SourceKind.FILING)

# A-2 staging 目录存在
assert (token.staging_root_dir / "AAPL" / "filings" / "fil_test_001").exists()

# A-3
batch_repo.commit_batch(token)
# 读取走 target,数据一致
meta_after_commit = src_repo.get_source_meta("AAPL", "fil_test_001", SourceKind.FILING)
assert meta_during_batch == meta_after_commit  # A-3 = A-1
assert not token.staging_root_dir.exists()  # staging 已清
```

**预期**:`meta_during_batch == meta_after_commit`,无字段丢失。staging 目录 commit 后被清理。

---

## 测试组 B:Crash 恢复(模拟 batch 中 commit 一半)

**目的**:验证 `recover_orphan_batches` 按 journal phase × 目录存在性决策恢复动作
(design_doc §5 跨阶段算账)。

观察项:
- B-1 写 staging + backup 之后强杀进程模拟 crash
- B-2 重启 → `ensure_batch_recovery` 自动决策
- **B-2 后目标目录状态 = B-1 commit 完成时该有的状态**

```bash
# 终端 1:启动 batch,写 staging,在 commit 三阶段中间停
python -c "
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from pathlib import Path
import time, os
ws = Path('/tmp/dayu_storage_test')
repo_set = build_fs_repository_set(workspace_root=ws)
core = repo_set.core
token = core.begin_batch('TSLA')
# 写 staging
(token.staging_root_dir / 'TSLA' / 'filings' / 'fil_crash_001').mkdir(parents=True, exist_ok=True)
(token.staging_root_dir / 'TSLA' / 'filings' / 'fil_crash_001' / 'meta.json').write_text('{}')
# 手动写到 BACKED_UP_TARGET phase 但不进入 SWAPPED
core._write_batch_journal(token, '_PHASE_BACKED_UP_TARGET')
print(f'journal at: {token.journal_path}')
os._exit(1)  # 强杀
"

# 终端 2:重启 + 自动 recovery
python -c "
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from pathlib import Path
ws = Path('/tmp/dayu_storage_test')
repo_set = build_fs_repository_set(workspace_root=ws)  # 自动跑 ensure_batch_recovery
core = repo_set.core
actions = core.recover_orphan_batches(dry_run=True)  # 已恢复时应空
print(f'remaining orphan actions: {actions}')
assert actions == ()  # B-2 自动恢复完毕
"
```

**预期**:B-2 重启时 `ensure_batch_recovery` 按 `_PHASE_BACKED_UP_TARGET` + backup 存在 + target 不存在 → restore backup → target,然后 `recover_orphan_batches(dry_run=True)` 返回空(无剩余 orphan)。

---

## 测试组 C:跨进程同 ticker 锁拒绝

**目的**:验证 `begin_batch` 对同一 ticker 跨进程并发抛
`RuntimeError("跨进程活动 batch")`(非阻塞 OS file lock,
POSIX `fcntl.flock` / Windows `msvcrt.locking`)。

观察项:
- C-1 进程 A 持锁中,进程 B `begin_batch("MSFT")` 必抛
- C-2 不同 ticker `begin_batch("GOOGL")` 在进程 B 可成功
- **锁粒度 = ticker,非全局**

```bash
# 终端 1
python -c "
import time
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from pathlib import Path
repo_set = build_fs_repository_set(workspace_root=Path('/tmp/dayu_storage_test'))
token = repo_set.core.begin_batch('MSFT')
print(f'A held {token.token_id}')
time.sleep(60)
"

# 终端 2(终端 1 跑着时启动)
python -c "
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from pathlib import Path
repo_set = build_fs_repository_set(workspace_root=Path('/tmp/dayu_storage_test'))
# C-1 同 ticker 必抛
try:
    repo_set.core.begin_batch('MSFT')
    raise AssertionError('应抛 RuntimeError')
except RuntimeError as e:
    print(f'C-1 OK: {e}')
# C-2 不同 ticker 可
token = repo_set.core.begin_batch('GOOGL')
print(f'C-2 OK: GOOGL token {token.token_id}')
repo_set.core.rollback_batch(token)
"
```

**预期**:C-1 抛"跨进程活动 batch",C-2 成功。**Windows 上必须真等到 fcntl 等效行为,
不能退化为重试 → `LK_LOCK` 有限次失败**(design_doc 文件锁不变量)。

---

## 测试组 D:Source 三态 (create → delete → restore → reset) 可逆 + 幂等

**目的**:验证 `delete_source_document` 是逻辑删除可恢复,`reset_source_document`
是物理重置幂等。

观察项:
- D-1 create → meta 存在,`is_deleted=False`
- D-2 delete → meta 存在,`is_deleted=True`
- D-4 restore → meta 存在,**`is_deleted=False`,数据 = D-1**
- D-6 reset(重复 3 次) → 目录 + manifest 条目都清,**3 次调用都不抛**
- **D-4 = D-1 数据等价(可逆契约)**
- **D-7 reset 后再 reset 仍幂等**

```python
from pathlib import Path
from dayu.fins.storage._fs_repository_factory import build_fs_repository_set
from dayu.fins.storage.fs_source_document_repository import FsSourceDocumentRepository
from dayu.fins.domain.document_models import (
    SourceDocumentUpsertRequest, SourceDocumentStateChangeRequest,
)
from dayu.fins.domain.enums import SourceKind

ws = Path("/tmp/dayu_storage_test")
repo_set = build_fs_repository_set(workspace_root=ws)
src_repo = FsSourceDocumentRepository(repository_set=repo_set)

# D-1
handle = src_repo.create_source_document(
    SourceDocumentUpsertRequest(ticker="NVDA", document_id="fil_d_001", ...),
    source_kind=SourceKind.FILING,
)
meta_d1 = src_repo.get_source_meta("NVDA", "fil_d_001", SourceKind.FILING)
assert meta_d1.is_deleted is False

# D-2
src_repo.delete_source_document(
    SourceDocumentStateChangeRequest(ticker="NVDA", document_id="fil_d_001", ...),
)
meta_d2 = src_repo.get_source_meta("NVDA", "fil_d_001", SourceKind.FILING)
assert meta_d2.is_deleted is True

# D-4 restore
src_repo.restore_source_document(
    SourceDocumentStateChangeRequest(ticker="NVDA", document_id="fil_d_001", ...),
)
meta_d4 = src_repo.get_source_meta("NVDA", "fil_d_001", SourceKind.FILING)
assert meta_d4.is_deleted is False
assert meta_d4 == meta_d1  # D-4 = D-1

# D-6 reset 3 次幂等
for i in range(3):
    src_repo.reset_source_document("NVDA", "fil_d_001", SourceKind.FILING)
# D-7 目录已无,meta 应抛 FileNotFoundError
try:
    src_repo.get_source_meta("NVDA", "fil_d_001", SourceKind.FILING)
    raise AssertionError("reset 后 meta 应不可读")
except FileNotFoundError:
    pass

# D-8 不存在 ticker reset 也幂等
src_repo.reset_source_document("NVDA", "fil_does_not_exist", SourceKind.FILING)  # 不抛
```

**预期**:D-4 = D-1 字段等价(reverse 可逆);D-6 三次 reset 不抛;D-8 reset 不存在文档也幂等。

---

## 测试组 E:Handle 越界 path traversal 拒绝

**目的**:验证 `_resolve_handle_child_path` 拒绝 `../`、绝对路径、`/`、`\`、
空字符串四类越界(`_normalize_entry_name` utils L81-101)。

观察项:
- E-1 `../etc/passwd` 拒绝
- E-2 `/etc/passwd` 拒绝(含路径分隔符)
- E-3 `..\\windows\\system32` 拒绝
- E-4 空字符串拒绝
- E-5 单点 `.` 拒绝
- E-6 双点 `..` 拒绝
- **所有 5 种都抛 `ValueError`**

```python
from dayu.fins.storage.fs_document_blob_repository import FsDocumentBlobRepository
from dayu.fins.domain.document_models import SourceHandle
from dayu.fins.domain.enums import SourceKind

# 准备一个合法 handle(测试组 D 后续)
handle = SourceHandle(ticker="NVDA", document_id="fil_e_001", source_kind=SourceKind.FILING)
blob_repo = FsDocumentBlobRepository(repository_set=repo_set)

attacks = [
    "../etc/passwd",
    "/etc/passwd",
    "..\\windows\\system32\\drivers\\etc\\hosts",
    "",
    ".",
    "..",
]
for name in attacks:
    try:
        blob_repo.read_file_bytes(handle, name)
        raise AssertionError(f"应拒绝越界: {name!r}")
    except ValueError as e:
        print(f"E-x OK: {name!r} → {e}")
    except FileNotFoundError:
        # 部分越界经过 normalize 后落到合法相对位置但文件不存在 — 这是 design bug
        raise AssertionError(f"越界 {name!r} 经 normalize 后落到合法位置,缺校验")
```

**预期**:6 种越界全部抛 `ValueError`,无一漏网。若任一抛 `FileNotFoundError`
说明 normalize 把越界变成"合法但找不到",**缺校验,是 P0 bug**。

---

## 测试组 F:共享 core 装配一致性

**目的**:验证 6 公开仓储共享同一 `FsStorageCore` 后,**任一 repo 看到的 batch
其它 repo 必看到**(design_doc §装配 共享 core 不变量)。

观察项:
- F-1 通过 batch_repo 起 batch
- F-2 src_repo / processed_repo / blob_repo 三个都能写入该 batch 的 staging
- **F-3 commit 后所有 repo 读到的数据一致**

```python
repo_set = build_fs_repository_set(workspace_root=ws)
batch_repo = FsBatchingRepository(repository_set=repo_set)
src_repo = FsSourceDocumentRepository(repository_set=repo_set)
processed_repo = FsProcessedDocumentRepository(repository_set=repo_set)
blob_repo = FsDocumentBlobRepository(repository_set=repo_set)

# F-1
token = batch_repo.begin_batch("AMD")

# F-2 三 repo 同时写 staging
src_handle = src_repo.create_source_document(...)
processed_handle = processed_repo.create_processed(...)
blob_repo.store_file(src_handle, "test.txt", BytesIO(b"shared core"))

# F-3 commit + 三 repo 读
batch_repo.commit_batch(token)
src_meta = src_repo.get_source_meta("AMD", src_handle.document_id, SourceKind.FILING)
processed_meta = processed_repo.get_processed_meta("AMD", processed_handle.document_id)
file_bytes = blob_repo.read_file_bytes(src_handle, "test.txt")
assert file_bytes == b"shared core"
# F-3 三 repo 都看到 commit 后的稳定状态
```

**预期**:共享 core 保证 batch 边界一致;若三 repo 各自 init 独立 core,
processed_meta 会读不到(staging 路径不同)。**这是与 conversation_memory phase
「显式注入装配漏洞 B-06」同型回归点**。

---

## 测试组 G:CompanyMeta 跨 ticker alias 冲突拒绝

**目的**:验证 `resolve_existing_ticker(candidates)` 在多个公司含同一 alias 时
抛 `ValueError("命中多个公司目录")`,**不静默返首个匹配**。

观察项:
- G-1 两公司各自 upsert 含 alias `OLDCO` 的 meta
- G-2 `resolve_existing_ticker(["OLDCO"])` 必抛
- **不静默选第一个**

```python
from dayu.fins.storage.fs_company_meta_repository import FsCompanyMetaRepository
from dayu.fins.domain.document_models import CompanyMeta

cm_repo = FsCompanyMetaRepository(repository_set=repo_set)

cm_repo.upsert_company_meta(CompanyMeta(ticker="NEWCO_A", aliases=("OLDCO", "ALPHA"), ...))
cm_repo.upsert_company_meta(CompanyMeta(ticker="NEWCO_B", aliases=("OLDCO", "BETA"), ...))

try:
    cm_repo.resolve_existing_ticker(["OLDCO"])
    raise AssertionError("G-2 应抛 ValueError")
except ValueError as e:
    assert "命中多个公司目录" in str(e)
    print(f"G-2 OK: {e}")

# G-3 单一匹配的 alias 应正常返回
result = cm_repo.resolve_existing_ticker(["ALPHA"])
assert result == "NEWCO_A"
```

**预期**:G-2 抛 ValueError 明示冲突,G-3 单匹配正常返回。

---

## 测试组 H:scan_inventory 三态分类

**目的**:验证 `scan_company_meta_inventory()` 三态分类:
`available / missing_meta / hidden_directory`。

观察项:
- `portfolio/AAPL/` 含 meta.json → `available`
- `portfolio/TSLA/` 无 meta.json → `missing_meta`
- `portfolio/.dayu/` 隐藏目录 → `hidden_directory`(必标)
- **三态不能漏 hidden_directory,不能把 `.dayu` 当成 missing_meta**

```python
import json
from pathlib import Path

portfolio = ws / "portfolio"
(portfolio / "AAPL").mkdir(parents=True, exist_ok=True)
(portfolio / "AAPL" / "meta.json").write_text(json.dumps({"ticker": "AAPL", "aliases": []}))
(portfolio / "TSLA").mkdir(parents=True, exist_ok=True)
# TSLA 故意无 meta.json
(portfolio / ".dayu" / "locks").mkdir(parents=True, exist_ok=True)

entries = cm_repo.scan_company_meta_inventory()
status_by_dir = {e.directory_name: e.status for e in entries}
assert status_by_dir.get("AAPL") == "available"
assert status_by_dir.get("TSLA") == "missing_meta"
assert status_by_dir.get(".dayu") == "hidden_directory"
```

**预期**:三态准确,`.dayu` 必标 hidden_directory **(关键 — 防 inventory 把内部目录当公司)**。

---

## 验证清单(每组测试结束后核对)

- [ ] 测试组 A:`meta_during_batch == meta_after_commit`(staging/target 切换无丢失)
- [ ] 测试组 B:`ensure_batch_recovery` 自动决策后 `recover_orphan_batches(dry_run=True) == ()`
- [ ] 测试组 C:C-1 抛 RuntimeError,C-2 成功(锁粒度 = ticker)
- [ ] 测试组 D:D-4 = D-1 数据等价,D-6 三次 reset 不抛,D-8 不存在文档 reset 不抛
- [ ] 测试组 E:6 种越界全部 ValueError,无一 FileNotFoundError
- [ ] 测试组 F:共享 core 装配后三 repo 看到同一 batch
- [ ] 测试组 G:多公司同 alias 抛 ValueError("命中多个公司目录")
- [ ] 测试组 H:`.dayu` 必标 hidden_directory

## 故障信号判读

| 现象 | 调整 |
|---|---|
| 测试组 A `meta_during_batch != meta_after_commit` | staging/target 切换逻辑出错,查 `_ticker_dir_for_read` |
| 测试组 B 重启后 orphan 仍残留 | `ensure_batch_recovery` 没被装配触发,查 `build_fs_repository_set(create_directories=True)` |
| 测试组 C Windows 上 C-1 不抛 | `dayu/file_lock.py` Windows 实现退化,查 `msvcrt.locking` blocking 真等 |
| 测试组 D D-4 ≠ D-1 字段 | restore 路径漏字段,查 `_FsSourceDocumentMixin` restore 实现 |
| 测试组 E 任一抛 FileNotFoundError | normalize 漏校验,**P0 安全 bug** |
| 测试组 F 三 repo 数据不一致 | 多 core 装配错误,检查 `_FsRepositorySet` 共享 |
| 测试组 G 静默返首匹配 | `resolve_existing_ticker` 未实现冲突检测,**P0 业务 bug** |
| 测试组 H `.dayu` 漏标 | `scan_inventory` 把隐藏目录当 missing_meta,**P1 inventory bug** |

## 与 conversation_memory phase 实测剧本的同型对照

| 维度 | conversation_memory(A-G 5 组) | fins/storage(A-H 8 组) |
|---|---|---|
| 一致校验 | A-7 = A-3 / D-13 = D-2 | A-3 = A-1 / D-4 = D-1 |
| 兜底保底 | recent_turns_floor / minimum_preserve | reset 幂等 / scan hidden_directory |
| 跨段连贯 | compaction 后 confirmed_facts | batch crash recovery → 数据等价 |
| 攻防边界 | 不漂移 / 反幻觉 | path traversal / 跨进程锁 / alias 冲突 |
| 调参信号 | budget cap / trigger ratio | 锁实现退化 / normalize 漏校验 |
