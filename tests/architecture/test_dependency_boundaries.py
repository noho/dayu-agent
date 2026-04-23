"""架构依赖边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


_REPO_ROOT = Path(__file__).resolve().parents[2]
_APP_SRC = _REPO_ROOT / "dayu"


def _iter_python_files(base_dir: Path) -> list[Path]:
    """收集目录下全部 Python 文件。"""

    return [path for path in base_dir.rglob("*.py") if path.is_file()]


def _collect_forbidden_imports(file_path: Path, forbidden_prefixes: tuple[str, ...]) -> list[str]:
    """提取文件中的受限导入。"""

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name.startswith(forbidden_prefixes):
                    hits.append(module_name)
            continue
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name.startswith(forbidden_prefixes):
                hits.append(module_name)
    return hits


def _collect_exact_imports(file_path: Path, forbidden_modules: tuple[str, ...]) -> list[str]:
    """提取文件中的精确受限导入。"""

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name in forbidden_modules:
                    hits.append(module_name)
            continue
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in forbidden_modules:
                hits.append(module_name)
    return hits


def _attribute_chain(node: ast.Attribute) -> tuple[str, ...]:
    """把 attribute 节点还原成属性链。"""

    parts: list[str] = []
    current: ast.AST = node
    while isinstance(current, ast.Attribute):
        parts.append(current.attr)
        current = current.value
    if isinstance(current, ast.Name):
        parts.append(current.id)
    return tuple(reversed(parts))


def _collect_forbidden_host_attribute_accesses(file_path: Path, forbidden_attrs: tuple[str, ...]) -> list[str]:
    """提取 Service 层对 Host 内部子组件的属性访问。"""

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[str] = []

    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        chain = _attribute_chain(node)
        if not chain:
            continue
        if chain[-1] not in forbidden_attrs:
            continue
        if "host" not in chain[:-1]:
            continue
        hits.append(".".join(chain))
    return hits


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_dir,forbidden_prefixes",
    [
        (_APP_SRC / "engine", ("dayu.host", "dayu.services", "dayu.startup", "dayu.cli", "dayu.web", "dayu.wechat")),
        (_APP_SRC / "host", ("dayu.services", "dayu.cli", "dayu.web", "dayu.wechat")),
        (
            _APP_SRC / "services",
            ("dayu.engine", "dayu.cli", "dayu.web", "dayu.wechat", "dayu.application", "dayu.runtime", "dayu.capabilities"),
        ),
        (_APP_SRC / "cli", ("dayu.application", "dayu.runtime", "dayu.capabilities")),
        (_APP_SRC / "web", ("dayu.application", "dayu.runtime", "dayu.capabilities")),
        (_APP_SRC / "wechat", ("dayu.application", "dayu.runtime", "dayu.capabilities")),
    ],
)
def test_layers_do_not_import_forbidden_modules(
    target_dir: Path,
    forbidden_prefixes: tuple[str, ...],
) -> None:
    """验证四层链路不存在反向依赖或旧命名空间依赖。"""

    violations: list[str] = []
    for file_path in _iter_python_files(target_dir):
        hits = _collect_forbidden_imports(file_path, forbidden_prefixes)
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现非法依赖：", *violations])


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_dir,forbidden_prefixes",
    [
        (_APP_SRC / "contracts", ("dayu.engine",)),
        (_APP_SRC / "execution", ("dayu.engine",)),
    ],
)
def test_contract_and_execution_layers_do_not_import_engine_modules(
    target_dir: Path,
    forbidden_prefixes: tuple[str, ...],
) -> None:
    """验证 contracts / execution 不再反向依赖 engine。"""

    violations: list[str] = []
    for file_path in _iter_python_files(target_dir):
        hits = _collect_forbidden_imports(file_path, forbidden_prefixes)
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现 contracts/execution 对 engine 的反向依赖：", *violations])


@pytest.mark.unit
@pytest.mark.parametrize(
    "entrypoint_path",
    [
        _APP_SRC / "cli" / "main.py",
        _APP_SRC / "wechat" / "main.py",
    ],
)
def test_ui_request_entrypoints_do_not_import_limit_model_types(entrypoint_path: Path) -> None:
    """验证 UI 请求入口不再直接依赖 execution/fins 的具体 limits 类型。"""

    hits = _collect_exact_imports(
        entrypoint_path,
        (
            "dayu.execution.doc_limits",
            "dayu.execution.web_limits",
            "dayu.fins.tools.fins_limits",
            "dayu.tool_limits",
        ),
    )
    assert hits == [], "\n".join(
        [
            "发现 UI 请求入口直接依赖具体 limits 类型：",
            f"{entrypoint_path.relative_to(_REPO_ROOT)}: {hits}",
        ]
    )


@pytest.mark.unit
def test_codebase_does_not_keep_hidden_service_assembly_terms() -> None:
    """验证代码库中不再保留旧装配术语。"""

    forbidden_tokens = (
        "ServiceRegistry",
        "ServicePlugin",
        "ServiceRuntimeContext",
        "ApplicationFactory",
        "RuntimeBundle",
        "CapabilityPlugin",
    )
    offenders: list[str] = []
    for file_path in _iter_python_files(_APP_SRC):
        content = file_path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden_tokens):
            offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现旧装配术语残留：", *offenders])


@pytest.mark.unit
def test_only_host_constructs_async_agent_from_agent_create_args() -> None:
    """验证 AgentCreateArgs 到 AsyncAgent 的装配只发生在 Host 内部。"""

    allowed_files = {
        (_APP_SRC / "host" / "agent_builder.py").resolve(),
        (_APP_SRC / "host" / "executor.py").resolve(),
        (_APP_SRC / "host" / "scene_preparer.py").resolve(),
    }
    offenders: list[str] = []
    for file_path in _iter_python_files(_APP_SRC):
        content = file_path.read_text(encoding="utf-8")
        if "build_async_agent(" not in content and "AsyncAgent(" not in content:
            continue
        if file_path.resolve() in allowed_files:
            continue
        if file_path.parts[-2] == "engine":
            continue
        offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现 Host 外部的 Agent 构造：", *offenders])


@pytest.mark.unit
def test_startup_preparation_does_not_construct_host() -> None:
    """验证 startup preparation 只提供稳定依赖，不直接构造 Host。"""

    startup_root = _APP_SRC / "startup"
    forbidden_tokens = (
        "Host(",
        "DefaultHostExecutor(",
        "build_host(",
    )
    offenders: list[str] = []
    for file_path in _iter_python_files(startup_root):
        content = file_path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden_tokens):
            offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现 startup 内部的 Host 构造：", *offenders])


@pytest.mark.unit
def test_startup_preparation_only_uses_public_service_and_host_preparation_modules() -> None:
    """验证 startup preparation 不再直接依赖 Service / Host 内部实现模块。"""

    startup_root = _APP_SRC / "startup"
    forbidden_prefixes = (
        "dayu.services.conversation_policy_reader",
        "dayu.services.scene_definition_reader",
        "dayu.services.scene_execution_acceptance",
        "dayu.host.concurrency",
    )
    violations: list[str] = []
    for file_path in _iter_python_files(startup_root):
        hits = _collect_forbidden_imports(file_path, forbidden_prefixes)
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现 startup 直接依赖下层内部实现：", *violations])


@pytest.mark.unit
def test_startup_preparation_does_not_construct_service_or_host_internal_components() -> None:
    """验证 startup preparation 不再显式实例化下层内部实现。"""

    startup_root = _APP_SRC / "startup"
    forbidden_tokens = (
        "ConversationPolicyReader(",
        "SceneDefinitionReader(",
        "SceneExecutionAcceptancePreparer(",
        "DEFAULT_LANE_CONFIG",
    )
    offenders: list[str] = []
    for file_path in _iter_python_files(startup_root):
        content = file_path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden_tokens):
            offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现 startup 泄漏下层内部实现装配：", *offenders])


@pytest.mark.unit
def test_ui_layers_do_not_construct_host_internal_default_components() -> None:
    """验证 UI 层不再显式构造 Host 默认内部实现细节。"""

    ui_roots = (
        _APP_SRC / "cli",
        _APP_SRC / "wechat",
        _APP_SRC / "web",
    )
    forbidden_tokens = (
        "HostStore(",
        "SQLiteSessionRegistry(",
        "SQLiteRunRegistry(",
        "SQLiteConcurrencyGovernor(",
        "DefaultScenePreparer(",
        "DefaultHostExecutor(",
    )
    offenders: list[str] = []
    for root in ui_roots:
        for file_path in _iter_python_files(root):
            content = file_path.read_text(encoding="utf-8")
            if any(token in content for token in forbidden_tokens):
                offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现 UI 层泄漏的 Host 内部实现细节构造：", *offenders])


@pytest.mark.unit
def test_service_layer_does_not_construct_agent_create_args() -> None:
    """验证 Service 层不再直接构造 AgentCreateArgs。"""

    services_root = _APP_SRC / "services"
    offenders: list[str] = []
    for file_path in _iter_python_files(services_root):
        content = file_path.read_text(encoding="utf-8")
        if "AgentCreateArgs(" in content:
            offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现 Service 内部的 AgentCreateArgs 构造：", *offenders])


def _collect_package_root_imports(
    file_path: Path,
    package_roots: tuple[str, ...],
) -> list[str]:
    """提取文件中对 package root 的直接导入。"""

    source = file_path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    hits: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            if module_name in package_roots:
                hits.append(module_name)
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                if module_name in package_roots:
                    hits.append(module_name)
    return hits


@pytest.mark.unit
@pytest.mark.parametrize(
    "target_dir,forbidden_package_roots",
    [
        (_APP_SRC / "services", ("dayu.prompting",)),
        (_APP_SRC / "host", ("dayu.prompting", "dayu.fins")),
        (_APP_SRC / "fins", ("dayu.fins",)),
    ],
)
def test_internal_modules_do_not_import_package_roots_for_implementation_symbols(
    target_dir: Path,
    forbidden_package_roots: tuple[str, ...],
) -> None:
    """验证内部实现层不再从自己或兄弟包的 package root 获取实现符号。"""

    violations: list[str] = []
    for file_path in _iter_python_files(target_dir):
        if file_path.name == "__init__.py":
            continue
        hits = _collect_package_root_imports(file_path, forbidden_package_roots)
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现内部实现层 package root 导入：", *violations])


@pytest.mark.unit
def test_service_layer_does_not_import_pending_turn_store_module() -> None:
    """验证 Service 层不能直接依赖 Host 内部 pending turn 仓储模块。"""

    services_root = _APP_SRC / "services"
    violations: list[str] = []
    for file_path in _iter_python_files(services_root):
        hits = _collect_exact_imports(file_path, ("dayu.host.pending_turn_store",))
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现 Service 直接依赖 Host pending turn 内部模块：", *violations])


@pytest.mark.unit
def test_service_layer_does_not_import_host_concrete_type() -> None:
    """验证 Service 层只依赖 Host 稳定协议，不直接导入 Host 具体实现。"""

    services_root = _APP_SRC / "services"
    violations: list[str] = []
    for file_path in _iter_python_files(services_root):
        hits = _collect_exact_imports(file_path, ("dayu.host.host",))
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现 Service 对 Host 具体实现的依赖：", *violations])


@pytest.mark.unit
def test_service_layer_does_not_access_host_internal_components() -> None:
    """验证 Service 层不直接读取 Host 内部子组件属性。"""

    services_root = _APP_SRC / "services"
    violations: list[str] = []
    forbidden_attrs = (
        "session_registry",
        "run_registry",
        "concurrency_governor",
        "executor",
        "event_bus",
        "_session_registry",
        "_run_registry",
        "_concurrency_governor",
        "_executor",
        "_event_bus",
        "_pending_turn_store",
    )
    for file_path in _iter_python_files(services_root):
        hits = _collect_forbidden_host_attribute_accesses(file_path, forbidden_attrs)
        for access_path in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {access_path}")
    assert not violations, "\n".join(["发现 Service 直接访问 Host 内部子组件：", *violations])


@pytest.mark.unit
def test_prompting_layer_does_not_depend_on_engine_prompt_module() -> None:
    """验证 prompting 不再依赖已删除的 engine prompt 渲染模块。"""

    prompting_root = _APP_SRC / "prompting"
    violations: list[str] = []
    for file_path in _iter_python_files(prompting_root):
        hits = _collect_forbidden_imports(file_path, ("dayu.engine.prompts",))
        for module_name in hits:
            violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现 prompting 对已删除 engine prompt 模块的依赖：", *violations])


@pytest.mark.unit
def test_engine_layer_does_not_keep_prompt_rendering_module_or_symbols() -> None:
    """验证 Engine 不再拥有 prompt 渲染模块或相关实现符号。"""

    prompts_module = _APP_SRC / "engine" / "prompts.py"
    assert not prompts_module.exists(), "发现遗留的 dayu/engine/prompts.py"

    engine_root = _APP_SRC / "engine"
    forbidden_tokens = (
        "PromptParseError",
        "GuidanceParseError",
        "load_prompt(",
        "parse_when_tool_blocks(",
        "parse_when_tag_blocks(",
    )
    offenders: list[str] = []
    for file_path in _iter_python_files(engine_root):
        content = file_path.read_text(encoding="utf-8")
        if any(token in content for token in forbidden_tokens):
            offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现 Engine 遗留的 prompt 渲染实现：", *offenders])


@pytest.mark.unit
def test_execution_contract_is_only_constructed_in_contract_preparation_module() -> None:
    """验证 ExecutionContract 业务装配只在 contract preparation 中收口构造。"""

    allowed_files = {
        (_APP_SRC / "services" / "contract_preparation.py").resolve(),
        (_APP_SRC / "contracts" / "agent_execution.py").resolve(),
        (_APP_SRC / "contracts" / "agent_execution_serialization.py").resolve(),
    }
    offenders: list[str] = []
    for file_path in _iter_python_files(_APP_SRC):
        content = file_path.read_text(encoding="utf-8")
        if "ExecutionContract(" not in content:
            continue
        if file_path.resolve() in allowed_files:
            continue
        offenders.append(str(file_path.relative_to(_REPO_ROOT)))
    assert offenders == [], "\n".join(["发现散落的 ExecutionContract 构造：", *offenders])


@pytest.mark.unit
def test_fins_pipelines_and_tools_do_not_import_legacy_document_repository_modules() -> None:
    """验证 Fins pipelines/tools 不再依赖旧总仓储模块。"""

    target_roots = (
        _APP_SRC / "fins" / "pipelines",
        _APP_SRC / "fins" / "tools",
    )
    forbidden_prefixes = (
        "dayu.fins.storage.document_repository",
        "dayu.fins.storage.fs_document_repository",
    )
    violations: list[str] = []
    for root in target_roots:
        for file_path in _iter_python_files(root):
            hits = _collect_forbidden_imports(file_path, forbidden_prefixes)
            for module_name in hits:
                violations.append(f"{file_path.relative_to(_REPO_ROOT)}: {module_name}")
    assert not violations, "\n".join(["发现 Fins 对旧总仓储模块的依赖：", *violations])


@pytest.mark.unit
def test_fins_runtime_protocol_does_not_expose_legacy_get_repository_api() -> None:
    """验证 FinsRuntimeProtocol 不再暴露总仓储 getter。"""

    runtime_file = _APP_SRC / "fins" / "service_runtime.py"
    content = runtime_file.read_text(encoding="utf-8")
    assert "def get_repository(" not in content, "发现遗留的 FinsRuntimeProtocol.get_repository API"


@pytest.mark.unit
def test_legacy_document_repository_modules_are_deleted() -> None:
    """验证旧总仓储 public 模块已被彻底删除。"""

    assert not (_APP_SRC / "fins" / "storage" / "document_repository.py").exists()
    assert not (_APP_SRC / "fins" / "storage" / "fs_document_repository.py").exists()
