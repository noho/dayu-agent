# GitHub 主线工作流

本文档面向：
- 把当前仓库首次开源到 GitHub 的切割操作。
- 开源之后的日常开发、贡献、同步流程。

核心假设：
- GitHub 远端名：`github`，地址 `git@github.com:noho/dayu-agent.git`。
- 局域网 Gitea 远端名：`lan`。
- 本地曾经以 `lan-main` 作为主线，有大量非公开开发历史。

---

## 一、整体策略

### 为什么需要切割

开源前的本地历史是你的私产：
- 含实验性提交、调试痕迹、私人路径；
- rewrite 成本远高于一次性切断；
- 对外部读者没有阅读价值。

因此：**用 orphan 提交作为公开起点，一次性切断历史，旧历史以 `archive/*` 归档。**

### 两个远端的角色分工

| 远端 | 角色 | 含哪些分支 |
|---|---|---|
| `github` | **唯一公开主线** | `main`（干净），各类 `feat/*` `fix/*` PR 分支 |
| `lan` | GitHub 镜像 + 私人备份 | `main`（镜像公开主线）、`archive/*`（旧历史）、可选 `wip/*`（私人 WIP） |

### 切割后的不变式

- **本地 `main` 绝不直接 commit**，任何修改都走 `feat/xxx` → PR。
- **GitHub `main` 是唯一事实源**，`lan/main` 只是它的只读镜像。
- 日常开发与外部贡献走**同一套 PR 流程**，不搞双轨制。

---

## 二、一次性切割（只做一次）

### 前置检查

```bash
git remote -v
git status
git log --oneline -1
git fetch github
```

确认：
- `github` 远端地址正确。
- 工作树干净（否则先 commit 或 stash）。
- 知道当前 HEAD 指向哪个 commit（这个 commit 的代码会成为公开起点）。
- 已经 fetch 到远端 `github/main`，这样后面的 `--force-with-lease` 才能生效。

### 第 1 步：归档旧历史

```bash
git branch -m lan-main archive/pre-opensource
git push lan archive/pre-opensource
git tag archive/pre-opensource-tag   # 额外加 tag 兜底，避免误删分支丢失历史
git push lan archive/pre-opensource-tag
```

做完这步：
- 本地 `lan-main` 改名为 `archive/pre-opensource`，语义清晰。
- 旧历史已同时存在于本地、`lan` 远端、和 tag 三处，几乎不可能丢。

### 第 2 步：基于当前代码创建 orphan 公开起点

不要用 `git switch --orphan` + `git add .`。这条路径有一个隐蔽陷阱：`git switch --orphan` 会把原分支 tracked 的部分 root 文件（例如 `README.md`、`LICENSE`）从工作区移除，`git add .` 随后只捕获残留文件，结果会做出一个缺失关键文件的首提交。

稳妥做法是直接用 `git commit-tree` 从当前 HEAD 的 tree 构造一个无父提交，完全不动工作区：

```bash
TREE=$(git rev-parse HEAD^{tree})
ORPHAN=$(git commit-tree "$TREE" -m "Initial open-source release")
git branch main "$ORPHAN"
git switch main
```

命令说明：
- `git rev-parse HEAD^{tree}` 取出当前 HEAD 的 tree 对象 SHA，这就是"当前代码快照"。
- `git commit-tree` 用这个 tree 新建一个 commit，不指定 `-p`，所以没有父提交，天然是 orphan。
- `git branch main "$ORPHAN"` 让新分支 `main` 指向这个 orphan commit。
- 最后 `git switch main`，此时工作区不动，`git status` 应该是干净的。

做完后应该看到：
- `git log --oneline` 只有 1 个 commit：`Initial open-source release`。
- `git ls-files | wc -l` 与 `archive/pre-opensource` 分支完全一致。
- `git status` 显示 `nothing to commit, working tree clean`。

如果上面三项任一不满足，停下排查，不要继续推送。

### 第 3 步：推到 GitHub main

```bash
git push -u --force-with-lease github main
```

- `-u` 建立本地 `main` 与 `github/main` 的 tracking 关系，以后直接 `git push` / `git pull` 就行。
- `--force-with-lease` 覆盖 GitHub 初始化时自动生成的提交；比 `--force` 安全，远端如果在你 fetch 之后又被改动会拒绝推送。

### 第 4 步：镜像到 lan

```bash
git push lan main:main
```

从此 `lan/main` 作为 GitHub 的只读镜像存在。

### 第 5 步：GitHub 仓库设置（网页操作）

去 `github.com/noho/dayu-agent` 的 Settings → General：
- **Default branch** 设为 `main`。
- Settings → Branches → Add branch protection rule：
  - Branch name pattern：`main`
  - 勾选 `Require a pull request before merging`
  - 勾选 `Require status checks to pass before merging`（有 CI 后再开）
  - 勾选 `Do not allow bypassing the above settings`（防止自己手滑直推）

做完切割后的仓库状态：

```
本地分支：
  main                        ← tracking github/main，公开主线
  archive/pre-opensource      ← 旧历史归档，不再开发

本地 tag：
  archive/pre-opensource-tag  ← 旧历史 HEAD 兜底

github 远端：
  main（干净）

lan 远端：
  main（公开主线镜像）
  archive/pre-opensource（旧历史）
  archive/pre-opensource-tag
```

---

## 三、日常开发流程

**铁律**：
本地 `main` 绝不直接 commit，只 pull。
所有改动走 feat/* → PR → merge。
跑长任务用 run/* tag + worktree，不要开长期分支。

### 1. 开工前同步主线

```bash
git switch main
git pull
```

### 2. 新建功能分支

```bash
git switch -c <feat/short-topic>
```

命名约定：
- `feat/xxx` 新功能
- `fix/xxx` 修 bug
- `docs/xxx` 仅文档
- `refactor/xxx` 重构
- `chore/xxx` 杂活（依赖升级、构建脚本等）

名字短而具体：`feat/sec-cache` 好过 `feat/update`。

### 3. 开发并提交

```bash
git status
git add <具体文件>
git commit -m "feat: 简短说明"
```

不用 `git add .`：你对仓库还不够熟时，`.` 容易把不想要的文件带进去。

一个功能分支可以有多个小 commit，最终通过 PR 的 "Squash and merge" 合成一个干净提交。

### 4. 推到 GitHub 并开 PR

```bash
git branch --show-current
git push -u github <feat/short-topic>
gh pr create --fill          # 或去网页开
```

如果想让 CI 额外跑扩展集成测试和 PR 可执行的平台完整验证：
```bash
git branch --show-current
git push -u github <feat/short-topic>
gh pr create --fill --label full-integration
```

或者 PR 创建后再加：
```bash
gh pr edit --add-label full-integration
```

`-u` 只在该分支**第一次** push 时加，后续直接 `git push`。

- **CI 失败或验证发现问题**：在功能分支上继续改、commit、push，PR 自动更新，CI 重跑。
PR 未 merge 前，功能分支还活着，直接在上面继续改：

```bash
git switch fix/xxx              # 确保在功能分支上
# 改代码、本地验证…
git add <具体文件>
git commit -m "fix: 修复 xxx"
git push
```
push 后 PR 自动更新，CI 重新跑。最终 merge 时用 **Squash and merge**，多个 commit 会合成一个干净提交。  
详见[场景 H](#h-pr-还没-mergeci-失败或验证发现问题)。

- **CI 通过**：Squash merge 并删除远端分支：
  ```bash
  gh pr merge <PR号> --squash --delete-branch \
  --subject "v0.1.2 — 提供离线安装；支持MiMo Plan海外；bug fix" \
  --body "最终提交说明"
  ```

### 4.1 CI 分三层运行

当前 GitHub Actions 已固定为三层验证，不靠创建 PR 时手工挑选：

1. `PR 必跑`
   - 触发：所有 `pull_request`
   - 内容：
     - `pyright`
     - 最低支持版本 `min-compat`
     - 快速 `pytest` 主链：`not integration and not slow and not e2e`
     - `linux-x64` 锁定环境下的最小真实 Docling 集成 smoke
     - `linux-x64` 离线安装包构建与 smoke
   - 目标：几分钟内给出主反馈，不把所有慢测试塞进每个 PR

2. `PR 扩展`
   - 默认不跑
   - 触发方式：
     - 给 PR 加 label：`full-integration`
   - 内容：
     - Ubuntu 下完整 `integration` 测试层
     - PR 可执行的平台完整验证矩阵：
       - `linux-x64`
       - `windows-x64`
       - `macos-arm64`
     - `macos-x64` 不在 PR 层阻塞，避免长期排队占用反馈时间

3. `主线 / 定时 / 手工完整验证`
   - `push main`：自动跑扩展层与主线默认平台完整验证
   - `schedule`：每日定时跑扩展层与主线默认平台完整验证
   - `workflow_dispatch`：可手工触发
     - 默认只跑快主链
     - 勾选 `run_extended_integration=true` 时再跑扩展 integration 层
     - 勾选 `run_full_matrix=true` 时再跑主线默认平台完整验证
     - 只有再额外勾选 `include_macos_x64=true` 时，才补跑 `macos-x64`

这里的“主线默认平台”固定为：

- `linux-x64`
- `windows-x64`
- `macos-arm64`

`macos-x64` 因为 GitHub runner 长期稀缺，不放进 `PR`、`push main`、`schedule` 的阻塞层，只保留在：

- `workflow_dispatch(run_full_matrix=true, include_macos_x64=true)`
- `release` 正式发布工作流

这三层的设计原则是：

- 主链 CI 必须包含少量真实集成 smoke
- 更慢的真实集成测试与完整平台矩阵分层运行
- 稀缺 runner（当前是 `macos-x64`）不放进日常阻塞层，只在手工完整验证和正式发布层收口
- 发布前完整验证仍以 Release workflow 为准

### 5. PR merge 后本地同步

`gh pr merge --delete-branch` 会自动删除本地和远端分支并切回 main，只需拉取最新：

```bash
git pull
```

### 6. 定期镜像到 lan（可选但推荐）

```bash
git push lan main
```

放进 cron 或者每次 PR merge 后顺手做一次都行。

### 7. PR 前做四平台安装包验证

在向 GitHub 提 PR 前，除了本地 `pyright`、受影响测试和常规命令 smoke 外，还要补一轮**离线安装包验证**。

当前平台矩阵与验证方式固定如下：

- `macos-arm64`：在当前本机直接构建并验证
- `macos-x64`：在 Intel macOS 宿主机验证
- `linux-x64`：在 Docker 容器中验证
- `windows-x64`：在 Windows x64 宿主机验证

这里有一个前提要明确：

- 当前 [utils/build_offline_bundle.py](</Users/leo/workspace/dayu-agent/utils/build_offline_bundle.py>) 会基于**当前运行平台**下载 wheelhouse
- `--platform-id` 主要用于命名、安装脚本分支和归档后缀
- 它**不是**“一台机器交叉构建任意平台离线包”的开关

因此，不要在一台 `macos-arm64` 机器上强行本地构建 `windows-x64` 或 `macos-x64` 离线包；那不是当前工具链支持的验证路径。

#### 通用步骤

每个平台都按同一套路做：

1. 构建项目 wheel
2. 基于对应平台的 `constraints/lock-<platform>-py311.txt` 构建离线安装包
3. 运行 [utils/smoke_test_offline_bundle.py](</Users/leo/workspace/dayu-agent/utils/smoke_test_offline_bundle.py>) 做干净虚拟环境 smoke

示例中的 `dist/offline` 可以按需替换成别的输出目录。

#### 7.1 `macos-arm64`（当前本机）

```bash
source .venv/bin/activate
python -m pip install --upgrade pip build
rm -rf dist build
python -m build --wheel
python utils/build_offline_bundle.py \
  --wheel "$(ls dist/dayu_agent-*.whl | tail -n1)" \
  --constraints constraints/lock-macos-arm64-py311.txt \
  --platform-id macos-arm64 \
  --output-dir dist/offline
python utils/smoke_test_offline_bundle.py \
  --archive "$(ls dist/offline/dayu-agent-*-macos-arm64-offline.tar.gz | tail -n1)"
```

#### 7.2 `macos-x64`

在 Intel macOS 宿主机上执行与上面相同的流程，只把 constraints 和平台标识换掉：

```bash
source .venv/bin/activate
python -m pip install --upgrade pip build
rm -rf dist build
python -m build --wheel
python utils/build_offline_bundle.py \
  --wheel "$(ls dist/dayu_agent-*.whl | tail -n1)" \
  --constraints constraints/lock-macos-x64-py311.txt \
  --platform-id macos-x64 \
  --output-dir dist/offline
python utils/smoke_test_offline_bundle.py \
  --archive "$(ls dist/offline/dayu-agent-*-macos-x64-offline.tar.gz | tail -n1)"
```

#### 7.3 `linux-x64`（Docker）

仓库内提供了 Linux 验证容器：

- [docker/linux-x64-offline-verify/Dockerfile](</Users/leo/workspace/dayu-agent/docker/linux-x64-offline-verify/Dockerfile>)

如果当前开发机是 Apple Silicon（`macos-arm64`），这里必须显式指定 `linux/amd64`。
否则 Docker 默认会起 `linux/arm64` 容器，最终下载到的是 `aarch64` wheel，但文件名仍会被误标成 `linux-x64`，验证结果失真。

先构建镜像（只需构建一次）：

```bash
docker build \
  --platform linux/amd64 \
  -f docker/linux-x64-offline-verify/Dockerfile \
  -t dayu-linux-x64-verify .
```

再启动容器：

```bash
mkdir -p "$HOME/.cache/dayu-agent/linux-x64-offline-verify/pip"
docker run --rm -it \
  --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp/home \
  -v "$HOME/.cache/dayu-agent/linux-x64-offline-verify/pip:/tmp/pip-cache" \
  -v "$PWD:/dayu-agent" \
  -w /dayu-agent \
  dayu-linux-x64-verify \
  /bin/sh -lc 'mkdir -p "$HOME" "$PIP_CACHE_DIR" && exec /bin/bash'
```

说明：

- 首次执行 `python utils/build_offline_bundle.py` 仍可能下载很久；这是正常的，因为需要把完整 Linux wheelhouse 拉到本机缓存。
- 只要继续复用同一个宿主机缓存目录 `~/.cache/dayu-agent/linux-x64-offline-verify/pip`，后续再次构建同平台、同 Python 版本、且依赖集合高度重叠的离线包时，`pip download` 会优先复用已缓存的 wheel 与 HTTP 元数据，通常会明显更快。
- 如果你删掉这个缓存目录、切换到另一台机器、修改锁定文件导致依赖版本大幅变化，或者上游只提供 sdist 需要重新构建，那么下载时间仍可能重新变长。

容器内执行：

```bash
rm -rf dist build
python -m build --wheel
python utils/build_offline_bundle.py \
  --wheel "$(ls dist/dayu_agent-*.whl | tail -n1)" \
  --constraints constraints/lock-linux-x64-py311.txt \
  --platform-id linux-x64 \
  --output-dir dist/offline
python utils/smoke_test_offline_bundle.py \
  --archive "$(ls dist/offline/dayu-agent-*-linux-x64-offline.tar.gz | tail -n1)"
```

#### 7.4 `windows-x64`

在 Windows x64 宿主机上执行同样流程，示意如下：

```powershell
python -m pip install --upgrade pip build
Remove-Item -Recurse -Force dist, build -ErrorAction SilentlyContinue
python -m build --wheel
$wheel = Get-ChildItem dist/dayu_agent-*.whl | Select-Object -Last 1
python utils/build_offline_bundle.py `
  --wheel $wheel.FullName `
  --constraints constraints/lock-windows-x64-py311.txt `
  --platform-id windows-x64 `
  --output-dir dist/offline
$archive = Get-ChildItem dist/offline/dayu-agent-*-windows-x64-offline.zip | Select-Object -Last 1
python utils/smoke_test_offline_bundle.py --archive $archive.FullName
```

#### 什么时候可以放心开 PR

理想情况是四个平台都先人工验证一遍，再开 PR。

如果时间有限，最低建议是：

- 当前开发机先验证 `macos-arm64`
- Docker 再验证 `linux-x64`
- `windows-x64` 至少确认对应宿主机流程可跑，剩余收口交给 GitHub Actions
- `macos-x64` 如需提前验证，需要在 Intel macOS 宿主机手工完成；正式 Release 会继续构建并上传该平台离线包

如果 PR 的改动直接触及离线打包、安装脚本、平台差异依赖或 CLI 入口，则尽量不要跳过任何平台。

### 8. 发布 release tag（按需）

积累到一个阶段性版本时，发布流程已经改成：

- **GitHub Release 是发布触发器**
- **离线安装包由 GitHub Actions 自动构建并上传**
- **不再手工本地 build wheel 再上传到 Release**

#### 产物形态

当前正式发布资产包括：

- 一个在线安装用的通用 wheel：
  - `dayu_agent-<version>-py3-none-any.whl`
- 4 个平台离线安装包：
  - `dayu-agent-<version>-macos-arm64-offline.tar.gz`
  - `dayu-agent-<version>-macos-x64-offline.tar.gz`
  - `dayu-agent-<version>-linux-x64-offline.tar.gz`
  - `dayu-agent-<version>-windows-x64-offline.zip`

这些资产由 `.github/workflows/release-offline.yml` 在 **GitHub Release 发布后** 自动生成并上传。

#### 发布步骤

##### 第 1 步：在功能分支里准备版本发布改动

在功能分支中：

1. 修改 `pyproject.toml` 中的 `version`（例如 `"0.2.0"`）。
2. 更新与当前版本相关的用户文档：
   - 根目录 `README.md` 中的离线安装包示例版本号；
   - 如有必要，更新开发文档或内部流程文档中的版本示例。
3. 提交、push、走 PR 流程，等待 CI 全绿后 merge。

##### 第 2 步：同步主线并打 tag

```bash
git switch main
git pull
git tag -a v0.1.2 -m "v0.1.2 — 提供离线安装；支持MiMo Plan海外；bug fix"
git push github v0.1.2
git push lan v0.1.2
```

##### 第 3 步：创建 GitHub Release

推荐用 `gh`，也可以直接在 GitHub 网页操作。

```bash
gh release create v0.1.2 \
  --title "v0.1.2 — 提供离线安装；支持MiMo Plan海外；bug fix" \
  --notes "$(cat <<'EOF'
## 安装

```bash
pip install https://github.com/noho/dayu-agent/releases/download/v0.1.2/dayu_agent-0.1.2-py3-none-any.whl
```

## 更新到新版本

```bash
pip install --upgrade https://github.com/noho/dayu-agent/releases/download/v0.1.2/dayu_agent-0.1.2-py3-none-any.whl
```

## 离线安装

从 [Releases](https://github.com/noho/dayu-agent/releases/tag/v0.1.2) 页面下载对应平台的离线安装包：

- macOS ARM64: `dayu-agent-0.1.2-macos-arm64-offline.tar.gz`
- macOS x64: `dayu-agent-0.1.2-macos-x64-offline.tar.gz`
- Linux x64: `dayu-agent-0.1.2-linux-x64-offline.tar.gz`
- Windows x64: `dayu-agent-0.1.2-windows-x64-offline.zip`

## 本次更新

- 提供离线安装包，支持 4 个平台（macOS ARM64/x64、Linux x64、Windows x64）
- 支持 MiMo Plan 海外（需运行 dayu-cli init --overwrite）
- bug fix

安装后可用命令：
- `dayu-cli init` — 初始化配置
- `dayu-cli` — 财报分析 CLI
- `dayu-wechat` — WeChat 服务
- `dayu-render` — 报告渲染
EOF
)"
```

说明：

- 不需要在本地手工构建 wheel 或上传文件。
- `gh release create` 完成后，GitHub 会触发 `Release Offline Bundles` workflow。
- 该 workflow 会自动执行两类发布任务：
  - 在单独的 wheel job 中构建通用 `py3-none-any` wheel，并上传到当前 Release
  - 在 4 个平台上分别构建离线安装包、执行 smoke test，并上传到当前 Release

##### 第 4 步：等待 Release workflow 完成

到 GitHub Actions 查看 `Release Offline Bundles` 是否全部成功。

只有当它全部成功后，这次 release 才算真正完成。最终 Release 页面应看到这 5 个资产：

- `dayu_agent-<version>-py3-none-any.whl`
- `dayu-agent-<version>-macos-arm64-offline.tar.gz`
- `dayu-agent-<version>-macos-x64-offline.tar.gz`
- `dayu-agent-<version>-linux-x64-offline.tar.gz`
- `dayu-agent-<version>-windows-x64-offline.zip`

如果只是想手工试跑发布流程而不污染正式 Release，可以手动触发 `workflow_dispatch`。  
但要注意：`workflow_dispatch` 只会生成 workflow artifact，不会把文件上传到 GitHub Release。

#### 移动已有 tag（例如发布后又合了紧急修复）

```bash
# 删掉旧 tag，重建指向 HEAD
git tag -d v0.2.0
git tag -a v0.2.0 -m "v0.2.0 — 简短描述"
# 更新远端 tag
git push github :refs/tags/v0.2.0 && git push github v0.2.0
git push lan :refs/tags/v0.2.0 && git push lan v0.2.0
# 删掉旧 Release，重新创建 Release 以重新触发自动构建上传
gh release delete v0.2.0 -y
gh release create v0.2.0 \
  --title "v0.2.0 — 简短描述" \
  --notes "..."
```

---

## 四、常见场景

### A. 远端 `main` 有别人合进来的提交

如果当时不在功能分支上：
```bash
git switch main
git pull
```

**如果当时正在功能分支上开发，先提交当前进度再同步**：

```bash
# 1. 保存当前功能分支的改动
git add <当前改动>
git commit -m "feat: WIP"

# 2. 同步 main
git switch main
git pull

# 3. 回到功能分支，rebase 到最新 main
git branch
git switch feat/xxx
git rebase main
```

推荐 `rebase`：功能分支保持线性，PR review 更直观；只要分支还没 push / 没人协作就安全。已经 push 到 GitHub 的分支做 rebase，push 时要加 `--force-with-lease`。

如果功能分支和新 merge 的 PR 没有文件冲突，也可以不急着 rebase，等开 PR 时 GitHub 会自动做 merge check。

### B. 同时开发多个功能 / 同时修 bug 和做 feature

每个任务一个独立分支，互不干扰。用 `git switch` 切换：

```bash
# 正在做 feat/xxx，突然要修 bug
git add <当前改动>
git commit -m "feat: WIP"       # 先提交当前进度
git switch main
git pull
git switch -c fix/yyy           # 从最新 main 开新分支修 bug
# 修完提交、push、开 PR...

# 回到之前的 feature 继续
git switch feat/xxx
```

WIP commit 留在功能分支历史里，最终 squash merge 时会合并掉，不影响 main。

要点：
- **每个分支从 `main` 开出**，不要从另一个功能分支开，避免依赖链。
- 切换前**先 commit 或 stash**，否则未提交修改会跟着带到另一个分支。
- 多个 PR 可以同时开着，各自独立 review 和 merge。

### C. 在功能分支上搞废了，想重来

```bash
git switch main
git branch -D feat/xxx
git switch -c feat/xxx
```

没 push 过的分支随便删，push 过的分支同样能删但顺手清一下远端：
```bash
git push github --delete feat/xxx
```

### D. 我忘了开分支，已经在 main 上改了代码（还没 commit）

```bash
git switch -c feat/xxx    # 带着未提交修改切到新分支
```

`git switch -c` 会把未提交修改一起带到新分支，`main` 回到干净状态。

### E. 我忘了开分支，已经在 main 上 commit 了（还没 push）

```bash
git switch -c feat/xxx    # 在 commit 上建新分支
git switch main
git reset --hard @{u}     # main 对齐到远端（github/main），丢掉本地 commit
git switch feat/xxx
```

`@{u}` 表示当前分支的上游（upstream），这里就是 `github/main`。

### F. 想临时把某个 WIP 分支备份到 lan，不想让 GitHub 看到

```bash
git push lan feat/xxx:wip/xxx
```

`lan` 作为私人备份，WIP 分支用 `wip/` 前缀区分，永远不推到 `github`。

### G. 想查开源前的旧历史

```bash
git log archive/pre-opensource
git checkout archive/pre-opensource   # 看完记得切回去
git switch main
```

### H. PR 还没 merge，CI 失败或验证发现问题

PR 未 merge 前，功能分支还活着，直接在上面继续改：

```bash
git switch fix/xxx              # 确保在功能分支上
# 改代码、本地验证…
git add <具体文件>
git commit -m "fix: 修复 xxx"
git push
```

push 后 PR 自动更新，CI 重新跑。最终 merge 时用 **Squash and merge**，多个 commit 会合成一个干净提交。

如果 PR 已经 merge 后才发现问题，开一个新的 `fix/` 分支走同样的 PR 流程。

### I. 跑长任务（download / process / write 等），需要工作区隔离

长任务（下载全量 SEC 财报、批量 process、跑完整 write pipeline）跑几个小时到几天，期间 `main` 可能还在改。不要开一条"长期稳定分支"来解决这个问题（见第五节反模式），正确做法是 **worktree 物理隔离**，按需决定是否打 tag。

#### 最简用法：直接基于当前 HEAD 开 worktree

不需要事后回溯时，不打 tag，直接用 HEAD：

```bash
# 1. 在独立目录展开当前代码
git worktree add ../dayu-agent-runs/download-aapl HEAD

# 2. 进入该目录跑任务，原目录继续开发
cd ../dayu-agent-runs/download-aapl
python -m dayu.cli download --ticker AAPL

# 3. 任务结束后清理 worktree
cd -
git worktree remove ../dayu-agent-runs/download-aapl
```

也可以基于已有的发布 tag（如 `v0.1.0`）开 worktree：
```bash
git worktree add ../dayu-agent-runs/prod-v0.1.0 v0.1.0
```

#### 已有 worktree 想更新到最新代码

```bash
cd ../dayu-agent-runs/download-aapl
git reset --hard main
```

#### 需要事后精确回溯：打 tag 冻结版本

任务跑崩或结果有疑问时，需要知道当时用的是哪个版本。这时先打 tag 再开 worktree：

```bash
# 1. 冻结当次任务的代码版本
git tag run/download-aapl-20260417
git push lan run/download-aapl-20260417   # 可选：推到 lan 留底，便于事后复盘

# 2. 在独立目录展开该 tag 的工作区
git worktree add ../dayu-agent-runs/download-aapl run/download-aapl-20260417

# 3. 进入该目录跑任务，原目录继续开发
cd ../dayu-agent-runs/download-aapl
python -m dayu.cli download --ticker AAPL

# 4. 任务结束后清理 worktree，tag 保留
cd -
git worktree remove ../dayu-agent-runs/download-aapl
```

#### 约定

- worktree 目录统一放在仓库同级的 `../dayu-agent-runs/<名字>` 下，便于集中管理。
- `run/*` tag 只推 `lan`（私人备份），不推 `github`（对外部贡献者无价值）。
- 同时跑多个长任务时，每个任务一个 worktree，互不干扰。

查看当前所有 worktree：
```bash
git worktree list
```

查看历史任务 tag：
```bash
git tag --list 'run/*'
```

---

## 五、反模式（都别做）

- ❌ **直接在 `main` 上改代码并 commit**：破坏 PR 流程，日后加 branch protection 会卡住自己。
- ❌ **长期维护 `lan-main` + `main` 两条主线**：cherry-pick 成本随时间指数上升。
- ❌ **为长任务留一条"稳定分支"**（例如 `base-01`）：分支会 diverge、需要手动同步 bug fix、违反主线纪律。用场景 I 的 tag + worktree 替代。
- ❌ **`git push --mirror`**：会把所有本地分支（包括实验分支、WIP 分支）推到 GitHub。
- ❌ **已 push 的分支用 `git push --force`**：覆盖其他人的提交；必须用 `--force-with-lease`。
- ❌ **在功能分支里 `git pull github main`**：容易产生嵌套 merge commit。要同步用 `rebase main` 或先切回 `main` 再 merge。

---

## 六、命令速查

```bash
# 看状态
git status
git branch --show-current
git log --oneline -5

# 同步主线
git switch main && git pull

# 开分支
git switch -c feat/xxx

# 提交
git add <file> && git commit -m "..."

# 推 PR
git push -u github feat/xxx

# PR merge 后清理
git switch main && git pull
git branch -d feat/xxx

# 镜像到 lan
git push lan main

# 看旧历史
git log archive/pre-opensource
```

---

## 七、远端/分支命名总览（固定下来别改）

| 类型 | 名字 | 含义 |
|---|---|---|
| 远端 | `github` | 公开主线 |
| 远端 | `lan` | 私人备份 / 镜像 |
| 分支 | `main` | 唯一长期主线 |
| 分支 | `feat/*` `fix/*` `docs/*` `refactor/*` `chore/*` | 功能分支，短期存活 |
| 分支 | `archive/*` | 归档，只读 |
| 分支 | `wip/*` | 私人 WIP，只推 `lan`，永不推 `github` |
| tag | `v*.*.*` | 发布 tag |
| tag | `archive/*` | 历史兜底 tag |
| tag | `run/*` | 长任务版本冻结 tag，仅推 `lan` |

---

## 八、术语表

| 术语 | 全称 | 含义 |
|---|---|---|
| WIP | Work In Progress | 还没做完，临时保存。常用于 commit message（如 `feat: WIP`），squash merge 时会被合并掉 |
| LGTM | Looks Good To Me | PR review 时表示"通过，没问题" |
| PTAL | Please Take A Look | 请人 review 时用 |
| NITS | Nitpicks | review 时指出不影响功能的小瑕疵（格式、命名等） |
| TBD | To Be Determined | 待定，还没决定 |
| FIXME | — | 代码注释里标记已知问题，需要修复 |
| TODO | — | 代码注释里标记待实现的功能 |
| HACK | — | 标记不优雅但暂时能用的方案 |
| PR | Pull Request | 合并请求，功能分支进入 main 的正式流程 |
| CI | Continuous Integration | 持续集成，push / 开 PR 后自动跑测试和类型检查 |
| squash | — | 把多个 commit 压成一个。GitHub 的 "Squash and merge" 会把功能分支的所有 commit 合并为一条 |
| rebase | — | 把功能分支的 commit 重新接到 main 最新位置，保持线性历史 |
| orphan | — | 没有父提交的 commit，用于切断历史 |
