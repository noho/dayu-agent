"""
通用文件工具模块 - 文件访问工具集

提供文件和目录访问的工具，支持文件列表、内容读取、搜索等功能。
所有工具通过 ToolRegistry 的路径安全检查机制保护，无需手动验证路径。

核心工具:
- list_files: 列出目录中的文件
- get_file_sections: 提取文件的章节结构（支持 Markdown / HTML / Docling JSON）
- search_files: 在目录中搜索包含关键词的文件
- read_file: 读取文件内容（支持行范围）
- read_file_section: 按 section ref 读取章节内容

设计原则:
- 自动安全检查：工具声明 file_path_params，由 ToolRegistry 自动验证路径
- 透明降级：无法处理时自动 fallback，不中断工作流
- 处理器驱动：Markdown/HTML/Docling JSON 文件通过 DocumentProcessor 提供精准结构化能力
- 简单直接：工具"做正确的事"，LLM 无需理解底层机制

入口:
- register_doc_tools(registry)
"""
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..processors._doc_processor_factory import create_doc_file_processor
from ..processors.search_utils import extract_query_anchored_snippets
from ..tool_registry import ToolRegistry
from ..exceptions import ToolArgumentError, FileAccessError
from dayu.log import Log
from ..tool_contracts import ToolTruncateSpec
from .base import tool
from dayu.contracts.tool_configs import DocToolLimits

MODULE = "ENGINE.DOC_TOOLS"

# read_file_section 支持的文件格式说明
_SUPPORTED_FORMATS_DESCRIPTION = "md, markdown, html, htm, *_docling.json"


def register_doc_tools(
    registry: ToolRegistry,
    limits: DocToolLimits | None = None,
    allowed_paths: Optional[list[Path]] = None,
    allow_file_write: bool = False,
    allowed_write_paths: Optional[list[str]] = None,
    timeout_budget: float | None = None,
) -> None:
    """
    注册所有通用文件工具。

    Args:
        registry: ToolRegistry 实例。
        limits: 文档工具限制配置（DocToolLimits），如果为 None 则使用默认值。
        allowed_paths: 允许访问的路径列表（文件或目录）。
            调用 registry.register_allowed_paths() 注册路径白名单。
            不传则不注册新路径（测试时可预先注册路径后直接调用）。
        allow_file_write: 是否允许写文件。当前 doc 工具集均为只读工具，该参数预留给未来写工具。
        allowed_write_paths: 允许写入的路径白名单。当前 doc 工具集均为只读工具，该参数预留给未来写工具。
        timeout_budget: Runner 为单次 tool call 提供的预算秒数；当前 doc 工具预留该参数，
            暂未消费。

    Example:
        >>> registry = ToolRegistry()
        >>> limits = DocToolLimits(list_files_max=100)
        >>> register_doc_tools(registry, limits=limits, allowed_paths=[Path("workspace")])
    """
    del timeout_budget
    del allow_file_write
    del allowed_write_paths

    # 注册路径白名单（安全机制）
    if allowed_paths:
        registry.register_allowed_paths(allowed_paths)

    # 使用传入的配置或默认值
    if limits is None:
        limits = DocToolLimits()

    list_files_max = limits.list_files_max
    get_sections_max = limits.get_sections_max
    search_files_max_results = limits.search_files_max_results
    read_file_max_chars = limits.read_file_max_chars
    read_file_section_max_chars = limits.read_file_section_max_chars

    # 注册 list_files
    name, func, schema = _create_list_files_tool(registry, list_files_max)
    registry.register(name, func, schema)

    # 注册 get_file_sections
    name, func, schema = _create_get_file_sections_tool(registry, get_sections_max)
    registry.register(name, func, schema)

    # 注册 search_files
    name, func, schema = _create_search_files_tool(registry, search_files_max_results)
    registry.register(name, func, schema)

    # 注册 read_file
    name, func, schema = _create_read_file_tool(registry, read_file_max_chars)
    registry.register(name, func, schema)

    # 注册 read_file_section
    name, func, schema = _create_read_file_section_tool(registry, read_file_section_max_chars)
    registry.register(name, func, schema)

    Log.verbose("已注册 5 个通用文件工具", module=MODULE)


def _create_list_files_tool(registry: ToolRegistry, max_files: int):
    """创建 list_files 工具"""
    
    parameters = {
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": "起点目录。先用它列出文件，再从返回的 files[].path 里选具体文件继续读取；不要猜不存在的路径。",
            },
            "pattern": {
                "type": "string",
                "description": "可选文件名通配符，例如 *.json、*.md。只在你明确要收窄文件范围时填写。",
            },
            "recursive": {
                "type": "boolean",
                "description": "是否递归子目录。目录层级不确定时设为 true。",
                "default": False,
            },
            "limit": {
                "type": "integer",
                "description": f"最多返回多少个文件。默认 20，最大 {max_files}。",
                "default": 20,
                "minimum": 1,
                "maximum": max_files,
            },
        },
        "required": ["directory"],
    }
    
    @tool(
        registry,
        name="list_files",
        description=(
            "列出目录中的文件。先用它定位文件，再把返回的 files[].path 交给 get_file_sections、read_file 或 read_file_section。"
        ),
        parameters=parameters,
        tags={"doc"},
        file_path_params=["directory"],
        display_name="列出文件",
    )
    def list_files(
        directory: str,
        pattern: Optional[str] = None,
        recursive: bool = False,
        limit: int = 20  # 默认 20，schema 中定义了 maximum
    ) -> Dict[str, Any]:
        """
        列出目录中的文件
        
        Args:
            directory: 目录路径
            pattern: 文件名 glob 模式
            recursive: 是否递归搜索
            limit: 最大返回数量
            
        Returns:
            Dict 包含:
                - directory: 目录路径
                - files: 文件列表（name, path, size, modified）
                - total: 文件总数
                - filtered: 过滤后的数量
        """
        # 确保 limit 不超过配置的硬性上限
        actual_limit = min(limit, max_files)
        
        dir_path = Path(directory)
        
        # 验证是否为目录
        if not dir_path.is_dir():
            raise FileAccessError(directory, "", "路径不是目录")
        
        # 收集文件
        files = []
        if recursive:
            all_files = dir_path.rglob(pattern if pattern else "*")
        else:
            all_files = dir_path.glob(pattern if pattern else "*")
        
        # 过滤并收集文件信息
        for file_path in all_files:
            if not file_path.is_file():
                continue
            
            try:
                stat = file_path.stat()
                files.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(dir_path)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })
            except OSError as e:
                Log.warn(f"无法读取文件信息: {file_path} - {e}", module=MODULE)
                continue
        
        # 按名称排序
        files.sort(key=lambda x: x["name"])
        
        total = len(files)
        filtered_files = files[:actual_limit] if actual_limit < total else files
        
        return {
            "directory": str(dir_path),
            "files": filtered_files,
            "total": total,
            "returned": len(filtered_files),
        }
    
    return list_files.__tool_name__, list_files, list_files.__tool_schema__


def _create_get_file_sections_tool(registry: ToolRegistry, max_sections: int):
    """创建 get_file_sections 工具"""

    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径。优先使用 list_files 返回的 files[].path；大文件先用本工具定位章节，再读具体章节。",
            },
            "limit": {
                "type": "integer",
                "description": f"最多返回多少个章节。默认 10，最大 {max_sections}。",
                "default": 10,
                "minimum": 1,
                "maximum": max_sections,
            },
        },
        "required": ["file_path"],
    }

    @tool(
        registry,
        name="get_file_sections",
        description=(
            "列出文件的章节结构。先用它定位章节；若返回的 sections[].ref 不为 null，就把 ref 交给 read_file_section。若 ref 为 null，改用 read_file，不要猜 ref。"
        ),
        parameters=parameters,
        tags={"doc"},
        file_path_params=["file_path"],
        display_name="浏览文件结构",
    )
    def get_file_sections(
        file_path: str,
        limit: int = 10  # 默认 10，schema 中定义了 maximum
    ) -> Dict[str, Any]:
        """
        列出文件中的所有 sections

        对 Markdown/HTML/Docling JSON 文件通过 DocumentProcessor 精准解析，
        其他格式降级为旧正则 / 单节 fallback。

        Args:
            file_path: 文件路径（已由 ToolRegistry 解析为绝对路径）
            limit: 最大返回 section 数

        Returns:
            Dict 包含:
                - file_path: 文件路径
                - sections: section 列表
                - total_sections: section 总数
                - returned_sections: 返回的 section 数
                - total_lines: 文件总行数
        """
        actual_limit = min(limit, max_sections)
        path = Path(file_path)

        # 尝试通过处理器解析
        processor = _try_create_processor(path)
        if processor is not None:
            return _sections_via_processor(processor, path, actual_limit)

        # 降级路径：手动读取文件
        lines = _read_file_lines(path)
        if lines is None:
            return _fallback_single_section(file_path, path)

        total_lines = len(lines)

        # 尝试提取 Markdown sections（纯文本 .md 但 processor 创建失败时的保底）
        if path.suffix.lower() in {".md", ".markdown"}:
            sections = _extract_markdown_sections(lines)
            if sections:
                filtered_sections = sections[:actual_limit]
                return {
                    "file_path": str(path),
                    "sections": filtered_sections,
                    "total_sections": len(sections),
                    "returned": len(filtered_sections),
                    "total_lines": total_lines,
                }

        # 最终 fallback: 单个 section 覆盖全文
        return _fallback_single_section(file_path, path, total_lines)

    return get_file_sections.__tool_name__, get_file_sections, get_file_sections.__tool_schema__


def _try_create_processor(path: Path):
    """安全地尝试创建处理器，失败时返回 None。

    Args:
        path: 文件绝对路径。

    Returns:
        DocumentProcessor 实例或 None。
    """
    try:
        return create_doc_file_processor(path)
    except Exception as exc:
        Log.warn(f"创建处理器失败，降级处理: {path} - {exc}", module=MODULE)
        return None


def _sections_via_processor(processor, path: Path, limit: int) -> Dict[str, Any]:
    """通过处理器获取章节列表并转换为工具输出格式。

    Args:
        processor: DocumentProcessor 实例。
        path: 文件路径。
        limit: 最大返回数。

    Returns:
        工具输出字典。
    """
    raw_sections = processor.list_sections()
    total_lines = _count_file_lines(path)

    # 通过 list_tables 构建 section_ref → table_ref 列表映射
    section_table_map: Dict[str, list] = {}
    try:
        for tbl in processor.list_tables():
            sec_ref = tbl.get("section_ref")
            if sec_ref:
                section_table_map.setdefault(sec_ref, []).append(tbl.get("table_ref", ""))
    except Exception:
        pass  # 部分 processor 可能不支持 list_tables

    sections = []
    for s in raw_sections:
        ref = s.get("ref")
        tbl_refs = section_table_map.get(ref, []) if ref else []
        section = {
            "ref": ref,
            "title": s.get("title"),
            "level": s.get("level"),
            "parent_ref": s.get("parent_ref"),
            "table_refs": tbl_refs,
            "table_count": len(tbl_refs),
            "preview": s.get("preview", ""),
        }
        # line_range：从 processor 内部 _SectionBlock 中无法直接获取；
        # 此处通过 section 索引估算（list_sections 返回字典，无 start_line/end_line）。
        # 仅置 None，由 read_file_section 提供精确内容。
        section["line_range"] = None
        section["line_count"] = None
        sections.append(section)

    filtered = sections[:limit]
    return {
        "file_path": str(path),
        "sections": filtered,
        "total_sections": len(sections),
        "returned": len(filtered),
        "total_lines": total_lines,
    }


def _count_file_lines(path: Path) -> int:
    """计算文件总行数。

    Args:
        path: 文件路径。

    Returns:
        行数，失败返回 0。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)
    except (UnicodeDecodeError, OSError):
        return 0


def _read_file_lines(path: Path) -> Optional[List[str]]:
    """读取文件全部行，尝试多编码。

    Args:
        path: 文件路径。

    Returns:
        行列表，或 None（无法解码）。
    """
    for encoding in ("utf-8", "gbk"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.readlines()
        except UnicodeDecodeError:
            continue
        except OSError:
            return None
    return None


def _extract_markdown_sections(lines: List[str]) -> List[Dict[str, Any]]:
    """
    从 Markdown 文件中提取章节结构（降级路径：processor 不可用时使用）

    返回的 section 包含新增字段（ref=None, level, parent_ref=None, table_refs=[], table_count=0），
    以保持与处理器路径一致的 schema 形状。

    Args:
        lines: 文件行列表

    Returns:
        sections 列表，如果提取失败返回 []
    """
    sections = []
    current_section = None

    for line_num, line in enumerate(lines, start=1):
        # 匹配 Markdown 标题（# 开头）
        match = re.match(r'^(#{1,6})\s+(.+)$', line.strip())
        if match:
            # 保存上一个 section
            if current_section:
                current_section["line_range"][1] = line_num - 1
                current_section["line_count"] = (
                    current_section["line_range"][1] - current_section["line_range"][0] + 1
                )
                sections.append(current_section)

            # 开始新 section
            title = match.group(2).strip()
            level = len(match.group(1))
            current_section = {
                "ref": None,
                "title": title,
                "level": level,
                "parent_ref": None,
                "table_refs": [],
                "table_count": 0,
                "line_range": [line_num, line_num],
                "line_count": 1,
                "preview": "",
            }

    # 保存最后一个 section
    if current_section:
        current_section["line_range"][1] = len(lines)
        current_section["line_count"] = (
            current_section["line_range"][1] - current_section["line_range"][0] + 1
        )
        sections.append(current_section)

    # 为每个 section 生成 preview（前 150 字符）
    for section in sections:
        start_line = section["line_range"][0]
        end_line = min(section["line_range"][1], start_line + 10)
        preview_lines = lines[start_line - 1:end_line]
        preview_text = "".join(preview_lines).strip()
        section["preview"] = preview_text[:150]

    return sections


def _fallback_single_section(
    file_path: str,
    path: Path,
    total_lines: int | None = None,
) -> Dict[str, Any]:
    """
    Fallback: 返回单个 section 覆盖整个文件

    Args:
        file_path: 文件路径字符串
        path: Path 对象
        total_lines: 总行数（如果已知）

    Returns:
        包含单个 section 的结果
    """
    if total_lines is None:
        total_lines = _count_file_lines(path)

    section = {
        "ref": None,
        "title": path.name,
        "level": None,
        "parent_ref": None,
        "table_refs": [],
        "table_count": 0,
        "line_range": [1, total_lines] if total_lines > 0 else None,
        "line_count": total_lines,
        "preview": f"整个文件（{path.name}）",
    }

    return {
        "file_path": str(path),
        "sections": [section],
        "total_sections": 1,
        "returned": 1,
        "total_lines": total_lines,
    }


def _create_search_files_tool(registry: ToolRegistry, max_results: int):
    """创建 search_files 工具"""

    parameters = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索词或短语。优先传单个明确概念；避免一次塞入过多无关关键词。",
            },
            "directory": {
                "type": "string",
                "description": "起点目录。先在这个目录里找命中文件，再把匹配结果交给 read_file_section 或 read_file。",
            },
            "include_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": '可选文件扩展名过滤，例如 ["py", "md", "json"]。只在你明确要限制文件类型时填写。',
            },
            "limit": {
                "type": "integer",
                "description": f"最多返回多少条命中。默认 20，最大 {max_results}。",
                "default": 20,
                "minimum": 1,
                "maximum": max_results,
            },
        },
        "required": ["directory", "query"],
    }

    @tool(
        registry,
        name="search_files",
        description=(
            "在目录中按关键词查找命中文件。若命中结果带 ref，优先把 ref 交给 read_file_section；若 ref 为 null，再用 read_file。"
        ),
        parameters=parameters,
        tags={"doc"},
        file_path_params=["directory"],
        display_name="搜索文件",
        summary_params=["query"],
    )
    def search_files(
        directory: str,
        query: str,
        include_types: Optional[List[str]] = None,
        limit: int = 20  # 默认 20，schema 中定义了 maximum
    ) -> Dict[str, Any]:
        """
        在目录中搜索包含关键词的文件

        对 Markdown/HTML/Docling JSON 文件通过 processor.search() 获取
        section-aware 搜索结果（含句子窗口 snippet）；其他文件降级为行扫描 +
        extract_query_anchored_snippets 生成质量 snippet。

        Args:
            directory: 目录路径
            query: 搜索关键词
            include_types: 文件类型过滤
            limit: 最大返回数量

        Returns:
            Dict 包含:
                - query: 搜索关键词
                - directory: 目录路径
                - matches: 匹配列表
                - total_matches: 匹配总数
        """
        actual_limit = min(limit, max_results)
        dir_path = Path(directory)

        if not dir_path.is_dir():
            raise FileAccessError(directory, "", "路径不是目录")

        matches: List[Dict[str, Any]] = []

        for file_path in dir_path.rglob("*"):
            if not file_path.is_file():
                continue

            # 文件类型过滤
            if include_types:
                if file_path.suffix.lstrip(".") not in include_types:
                    continue

            relative_path = str(file_path.relative_to(dir_path))

            # 尝试通过处理器搜索
            processor = _try_create_processor(file_path)
            if processor is not None:
                file_matches = _search_via_processor(processor, relative_path, query)
                matches.extend(file_matches)
                if len(matches) >= actual_limit:
                    break
                continue

            # 降级路径：行扫描 + 高质量 snippet
            file_matches = _search_via_line_scan(file_path, relative_path, query, actual_limit - len(matches))
            matches.extend(file_matches)
            if len(matches) >= actual_limit:
                break

        # 截断到限制
        matches = matches[:actual_limit]

        return {
            "query": query,
            "directory": str(dir_path),
            "matches": matches,
            "total_matches": len(matches),
        }

    return search_files.__tool_name__, search_files, search_files.__tool_schema__


def _search_via_processor(
    processor,
    relative_path: str,
    query: str,
) -> List[Dict[str, Any]]:
    """通过处理器搜索文件内容，返回标准化匹配列表。

    Args:
        processor: DocumentProcessor 实例。
        relative_path: 文件相对路径。
        query: 搜索关键词。

    Returns:
        匹配字典列表。
    """
    try:
        hits = processor.search(query)
    except Exception as exc:
        Log.warn(f"处理器搜索失败: {relative_path} - {exc}", module=MODULE)
        return []

    matches = []
    for hit in hits:
        matches.append({
            "file": relative_path,
            "line_number": None,
            "ref": hit.get("section_ref"),
            "section_title": hit.get("section_title"),
            "snippet": hit.get("snippet", ""),
            "matched_line_content": None,
        })
    return matches


def _search_via_line_scan(
    file_path: Path,
    relative_path: str,
    query: str,
    remaining: int,
) -> List[Dict[str, Any]]:
    """行扫描搜索（降级路径），附带高质量 snippet。

    Args:
        file_path: 文件绝对路径。
        relative_path: 文件相对路径。
        query: 搜索关键词。
        remaining: 剩余可返回的匹配数。

    Returns:
        匹配字典列表。
    """
    query_lower = query.lower()
    matches: List[Dict[str, Any]] = []

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except (UnicodeDecodeError, OSError):
        return []

    # 使用 extract_query_anchored_snippets 生成高质量 snippet
    snippets = extract_query_anchored_snippets(content, query)

    if not snippets:
        return []

    # 将 snippets 关联到行号
    lines = content.split("\n")
    snippet_idx = 0
    for line_num, line in enumerate(lines, start=1):
        if query_lower in line.lower():
            snippet_text = snippets[snippet_idx] if snippet_idx < len(snippets) else line.strip()[:150]
            matches.append({
                "file": relative_path,
                "line_number": line_num,
                "ref": None,
                "section_title": None,
                "snippet": snippet_text,
                "matched_line_content": line.strip(),
            })
            snippet_idx += 1
            if len(matches) >= remaining:
                break

    return matches


def _create_read_file_tool(registry: ToolRegistry, max_chars: int):
    """创建 read_file 工具"""
    
    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径。优先使用 list_files 返回的 files[].path。没有章节 ref、或文件不支持章节读取时用它。",
            },
            "start_line": {
                "type": "integer",
                "description": "起始行号，从 1 开始，包含该行。不填则从第 1 行开始读。",
                "minimum": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "结束行号，从 1 开始，包含该行。不填则读到文件末尾。",
                "minimum": 1,
            },
        },
        "required": ["file_path"],
    }
    
    @tool(
        registry,
        name="read_file",
        description=(
            "按整文件或按行范围读取内容。没有 ref、或文件不支持章节读取时用它。"
        ),
        parameters=parameters,
        tags={"doc"},
        file_path_params=["file_path"],
        display_name="读取文件",
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="text_chars",
            limits={"max_chars": max_chars},
            target_field="content",
        ),
    )
    def read_file(
        file_path: str,
        start_line: Optional[int] = None,
        end_line: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        读取文件内容
        
        Args:
            file_path: 文件路径
            start_line: 起始行号（1-based）
            end_line: 结束行号（inclusive）
            
        Returns:
            Dict 包含:
                - file_path: 文件路径
                - content: 文件内容
                - line_range: 读取的行范围（如果指定）
                - total_lines: 文件总行数
            
        Note:
            截断机制通过 @tool 装饰器声明，由 ToolRegistry 自动处理。
            工具函数返回完整内容，ToolRegistry 根据声明自动截断并添加 truncated/truncation 字段。
        """
        path = Path(file_path)
        
        # 尝试多种编码读取
        encodings = ['utf-8', 'gbk', 'latin1', 'cp1252']
        lines = None
        used_encoding = None
        
        for encoding in encodings:
            try:
                with open(path, 'r', encoding=encoding) as f:
                    lines = f.readlines()
                used_encoding = encoding
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if lines is None:
            raise FileAccessError(
                str(path.parent),
                path.name,
                f"无法读取文件，尝试过的编码: {encodings}"
            )
        
        total_lines = len(lines)
        
        # 处理行范围
        if start_line is None:
            start_line = 1
        if end_line is None:
            end_line = total_lines
        
        # 验证行号
        if start_line < 1:
            raise ToolArgumentError("read_file", "start_line", start_line, "必须 >= 1")
        if end_line < start_line:
            raise ToolArgumentError(
                "read_file",
                "end_line",
                end_line,
                f"必须 >= 起始行号 {start_line}"
            )
        
        # 提取行（转为 0-based 索引）
        start_idx = start_line - 1
        end_idx = min(end_line, total_lines)
        selected_lines = lines[start_idx:end_idx]
        
        content = ''.join(selected_lines)
        
        result = {
            "file_path": str(path),
            "content": content,
            "total_lines": total_lines,
        }
        
        # 如果指定了行范围，返回实际读取的范围
        if start_line != 1 or end_line != total_lines:
            result["line_range"] = [start_line, end_idx]
        
        return result
    
    return read_file.__tool_name__, read_file, read_file.__tool_schema__


def _create_read_file_section_tool(registry: ToolRegistry, max_chars: int):
    """创建 read_file_section 工具"""

    parameters = {
        "type": "object",
        "properties": {
            "file_path": {
                "type": "string",
                "description": "文件路径。优先使用 list_files 返回的 files[].path，或当前已知的允许访问路径。",
            },
            "ref": {
                "type": "string",
                "description": "必须来自 get_file_sections 返回的 sections[].ref。若 get_file_sections 里 ref 为 null，就改用 read_file，不要猜 ref。",
            },
        },
        "required": ["file_path", "ref"],
    }

    @tool(
        registry,
        name="read_file_section",
        description=(
            f"按章节 ref 读取内容。ref 必须来自 get_file_sections；支持的格式有 {_SUPPORTED_FORMATS_DESCRIPTION}。若文件不支持章节读取或 ref 为 null，改用 read_file。"
        ),
        parameters=parameters,
        tags={"doc"},
        file_path_params=["file_path"],
        display_name="读取文件段落",
        truncate=ToolTruncateSpec(
            enabled=True,
            strategy="text_chars",
            limits={"max_chars": max_chars},
            target_field="content",
        ),
    )
    def read_file_section(
        file_path: str,
        ref: str,
    ) -> Dict[str, Any]:
        """
        按 section ref 读取文件章节内容

        通过 DocumentProcessor.read_section(ref) 获取章节详细内容，
        包括正文、表格 ref 列表和子章节导航。

        Args:
            file_path: 文件路径（已由 ToolRegistry 解析为绝对路径）
            ref: 章节 ref（从 get_file_sections 获取）

        Returns:
            Dict 包含:
                - file_path: 文件路径
                - ref: 章节 ref
                - title: 章节标题
                - content: 章节正文（表格位置用 [[t_XXXX]] 占位符标记）
                - tables: 该 section 包含的表格 ref 列表
                - children: 直接子章节导航
                - content_word_count: 正文词数

        Raises:
            ToolArgumentError: ref 无效时抛出
        """
        path = Path(file_path)

        # 创建处理器
        processor = _try_create_processor(path)
        if processor is None:
            raise ToolArgumentError(
                "read_file_section",
                "file_path",
                file_path,
                f"该文件格式不支持 read_file_section。"
                f"支持的格式: {_SUPPORTED_FORMATS_DESCRIPTION}。"
                f"请使用 read_file 工具按行读取。",
            )

        # 读取章节
        try:
            section_content = processor.read_section(ref)
        except KeyError:
            raise ToolArgumentError(
                "read_file_section",
                "ref",
                ref,
                "章节 ref 不存在，请通过 get_file_sections 获取有效的 ref",
            )

        # 构建子章节导航
        children = _get_section_children(processor, ref)

        content = section_content.get("content", "")
        return {
            "file_path": str(path),
            "ref": ref,
            "title": section_content.get("title"),
            "content": content,
            "tables": section_content.get("tables", []),
            "children": children,
            "content_word_count": len(content.split()),
        }

    return read_file_section.__tool_name__, read_file_section, read_file_section.__tool_schema__


def _get_section_children(processor, parent_ref: str) -> List[Dict[str, Any]]:
    """获取指定章节的直接子章节列表。

    遍历 processor.list_sections()，找出 parent_ref 匹配的子节点。

    Args:
        processor: DocumentProcessor 实例。
        parent_ref: 父章节 ref。

    Returns:
        子章节列表，每个元素包含 ref/title/level/preview。
    """
    children = []
    try:
        all_sections = processor.list_sections()
    except Exception:
        return children

    for s in all_sections:
        if s.get("parent_ref") == parent_ref:
            children.append({
                "ref": s.get("ref"),
                "title": s.get("title"),
                "level": s.get("level"),
                "preview": s.get("preview", ""),
            })
    return children
