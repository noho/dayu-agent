"""
截断管理器 — 工具结果截断与游标分页续读

从 ToolRegistry 拆分而来，封装所有与结果截断和分页相关的逻辑：
- 按策略（text_chars / text_lines / list_items / binary_bytes）截断
- 游标存储与过期清理
- fetch_more 续读执行
"""

import base64
import copy
import hashlib
import json
import time
import uuid
from threading import RLock
from typing import Any, Dict, List, Optional

from dayu.contracts.protocols import ToolExecutionContext
from dayu.log import Log
from .tool_contracts import TRUNCATION_STRATEGIES, ToolTruncateSpec
from .tool_result import build_error, build_success

MODULE = "ENGINE.TRUNCATION_MANAGER"

# 游标默认 TTL 300 秒（5 分钟）——与 OpenAI API 请求超时上限对齐，
# 确保游标在同一轮工具调用链内有效，超时后自动清理避免内存泄漏。
_CURSOR_TTL_FALLBACK_SEC = 300.0
_CONTINUATION_ACTION_FETCH_MORE = "fetch_more"
_CONTINUATION_PRIORITY_HIGH = "high"
_SCOPE_MISMATCH_ERROR = {"code": "cursor_scope_mismatch", "message": "cursor scope mismatch"}
TruncationContext = ToolExecutionContext | None


class TruncationManager:
    """工具结果截断与游标分页续读管理器。

    管理截断策略执行和游标生命周期，支持 text_chars / text_lines /
    list_items / binary_bytes 四种截断策略。

    游标存储在内存中，通过 TTL 自动过期。TTL 默认为 300 秒，
    若执行上下文中提供了 timeout 则以 timeout 为准。
    内部通过可重入锁保护游标读写，保证并发工具调用下的游标一致性。
    """

    def __init__(self) -> None:
        self._cursor_store: Dict[str, Dict[str, Any]] = {}
        self._lock = RLock()

    def clear_cursors(self) -> None:
        """清除所有游标，释放关联的数据引用。

        在新 run 开始前调用，避免上一轮残留游标占用内存。
        """
        with self._lock:
            self._cursor_store.clear()

    # ------------------------------------------------------------------
    # 公共入口
    # ------------------------------------------------------------------

    def apply_truncation(
        self,
        name: str,
        arguments: Dict[str, Any],
        value: Any,
        context: TruncationContext,
        truncate_spec: Optional[ToolTruncateSpec],
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """对工具原始返回值按 schema 驱动的截断策略做截断。

        Args:
            name: 工具名称（用于 scope hash 计算）。
            arguments: 工具参数（用于 scope hash 计算）。
            value: 工具原始返回值。
            context: 执行上下文。
            truncate_spec: 截断规格（来自工具 schema）。

        Returns:
            ``(value_or_truncated, truncation_info_or_none)``
        """
        if not truncate_spec or not truncate_spec.enabled:
            return value, None

        strategy = truncate_spec.strategy
        limits = truncate_spec.limits or {}
        if strategy not in TRUNCATION_STRATEGIES:
            return value, None

        limit_key = TRUNCATION_STRATEGIES[strategy]["limit_key"]
        limit = limits.get(limit_key)
        if not isinstance(limit, int) or limit <= 0:
            return value, None

        scope_hash = self._build_scope_hash(name, arguments)

        target_field = truncate_spec.target_field

        if strategy in ("text_chars", "text_lines"):
            text, template, field_path = self._extract_text_target(value, target_field)
            if text is None:
                return value, None
            if strategy == "text_chars":
                output_value, truncation = self._truncate_text_chars(
                    text=text, limit=limit, template=template,
                    field_path=field_path, context=context, scope_hash=scope_hash,
                )
            else:
                output_value, truncation = self._truncate_text_lines(
                    text=text, limit=limit, template=template,
                    field_path=field_path, context=context, scope_hash=scope_hash,
                )
            if truncation:
                return output_value, truncation
            return value, None

        if strategy == "list_items":
            items, template, field_path = self._extract_list_target(value, target_field)
            if items is None:
                return value, None
            output_value, truncation = self._truncate_list_items(
                items=items, limit=limit, template=template,
                field_path=field_path, context=context, scope_hash=scope_hash,
            )
            if truncation:
                # 应用工具级续读行为覆盖（如 list_tables 不鼓励 fetch_more）
                if truncate_spec.continuation_hint:
                    truncation.update(truncate_spec.continuation_hint)
                return output_value, truncation
            return value, None

        if strategy == "binary_bytes":
            data, template, field_path = self._extract_binary_target(value)
            if data is None:
                return value, None
            output_value, truncation = self._truncate_binary_bytes(
                data=data, limit=limit, template=template,
                field_path=field_path, context=context, scope_hash=scope_hash,
            )
            if truncation:
                return output_value, truncation
            return value, None

        return value, None

    def execute_fetch_more(
        self,
        arguments: Dict[str, Any],
        context: TruncationContext,
    ) -> Dict[str, Any]:
        """执行 fetch_more 续读操作。

        Args:
            arguments: 包含 ``cursor``、``scope_token`` 与可选 ``limit`` 的参数。
            context: 执行上下文（用于 scope 校验）。

        Returns:
            标准化工具结果 dict。
        """
        with self._lock:
            cursor = arguments.get("cursor")
            if not isinstance(cursor, str) or not cursor:
                return build_error("invalid_cursor", "cursor is required")
            record = self._cursor_store.get(cursor)
            if not record:
                return build_error("cursor_not_found", "cursor not found")
            now = time.monotonic()
            if record.get("expires_at", 0) <= now:
                self._cursor_store.pop(cursor, None)
                return build_error("cursor_expired", "cursor expired")
            scope_error = self._validate_cursor_context(record, context)
            if scope_error:
                return build_error(
                    str(scope_error.get("code") or "cursor_scope_mismatch"),
                    str(scope_error.get("message") or "cursor scope mismatch"),
                )
            scope_token_error = self._validate_scope_token(record, arguments)
            if scope_token_error:
                return build_error(
                    str(scope_token_error.get("code") or "cursor_scope_mismatch"),
                    str(scope_token_error.get("message") or "cursor scope mismatch"),
                )

            limit = self._resolve_fetch_limit(arguments.get("limit"), record["limit"])
            chunk, chunk_size = self._build_chunk(
                mode=record["mode"], data=record["data"],
                offset=record["offset"], limit=limit,
            )
            output_value = self._apply_chunk_to_template(
                record["template"], record["field_path"], chunk,
            )
            self._postprocess_truncated_value(
                record["tool_name"], record["field_path"],
                output_value, chunk, length_field=record.get("length_field"),
            )
            new_offset = record["offset"] + chunk_size
            if chunk_size <= 0:
                self._cursor_store.pop(cursor, None)
                return build_success(value=output_value)

            has_more = new_offset < record["total"]
            if has_more:
                # single-use 语义：旧 cursor 失效，创建新 cursor 作为下一页凭证
                self._cursor_store.pop(cursor, None)
                new_cursor = self._store_cursor(
                    tool_name=record["tool_name"],
                    scope_hash=record["scope_hash"],
                    reason=record["reason"],
                    unit=record["unit"],
                    limit=record["limit"],
                    total=record["total"],
                    data=record["data"],
                    offset=new_offset,
                    template=record["template"],
                    field_path=record["field_path"],
                    mode=record["mode"],
                    context=ToolExecutionContext(
                        run_id=str(record.get("run_id") or "").strip() or None,
                        iteration_id=str(record.get("iteration_id") or "").strip() or None,
                        tool_call_id=str(record.get("tool_call_id") or "").strip() or None,
                        timeout_seconds=max(
                            record.get("expires_at", 0) - record.get("created_at", 0),
                            _CURSOR_TTL_FALLBACK_SEC,
                        ),
                    ),
                    length_field=record.get("length_field"),
                )
                truncation = self._build_truncation_info(
                    cursor=new_cursor, reason=record["reason"],
                    limit=record["limit"], unit=record["unit"],
                    total=record["total"], has_more=True,
                    scope_token=self._cursor_store[new_cursor].get("scope_token"),
                )
                return build_success(value=output_value, truncation=truncation)

            self._cursor_store.pop(cursor, None)
            return build_success(value=output_value)

    # ------------------------------------------------------------------
    # 目标提取
    # ------------------------------------------------------------------

    def _extract_text_target(
        self,
        value: Any,
        target_field: Optional[str] = None,
    ) -> tuple[Optional[str], Optional[Dict[str, Any]], Optional[List[str]]]:
        """从工具原始返回值中提取文本截断目标。

        Args:
            value: 工具原始返回值。
            target_field: 显式指定的截断目标字段名。为 None 时回退到启发式选择。

        Returns:
            ``(text, template, field_path)``
        """
        if isinstance(value, str):
            return value, None, None

        if isinstance(value, dict):
            field = target_field if target_field and target_field in value else self._select_largest_text_field(value)
            if not field:
                return None, None, None
            text = value.get(field)
            if not isinstance(text, str):
                return None, None, None
            template = copy.deepcopy(value)
            template[field] = None
            return text, template, [field]

        return None, None, None

    def _extract_list_target(
        self,
        value: Any,
        target_field: Optional[str] = None,
    ) -> tuple[Optional[List[Any]], Optional[Dict[str, Any]], Optional[List[str]]]:
        """从工具原始返回值中提取列表截断目标。

        Args:
            value: 工具原始返回值。
            target_field: 显式指定的截断目标字段名。为 None 时回退到启发式选择。
        """
        if isinstance(value, list):
            return value, None, None
        if isinstance(value, dict):
            # 当显式指定 target_field 且命中时，直接使用单层路径
            if target_field and target_field in value and isinstance(value[target_field], list):
                field_path: Optional[List[str]] = [target_field]
            else:
                field_path = self._select_largest_list_path(value)
            if not field_path:
                return None, None, None
            items = self._read_nested_dict_value(value, field_path)
            if not isinstance(items, list):
                return None, None, None
            template = copy.deepcopy(value)
            if not self._write_nested_dict_value(template, field_path, None):
                return None, None, None
            return items, template, field_path
        return None, None, None

    def _extract_binary_target(
        self, value: Any,
    ) -> tuple[Optional[bytes], Optional[Dict[str, Any]], Optional[List[str]]]:
        """从工具原始返回值中提取二进制截断目标。"""
        if isinstance(value, (bytes, bytearray)):
            return bytes(value), None, None
        return None, None, None

    # ------------------------------------------------------------------
    # 字段选择
    # ------------------------------------------------------------------

    def _select_largest_text_field(self, value: Dict[str, Any]) -> Optional[str]:
        """选出字典中最长的字符串字段。"""
        candidates = [(key, val) for key, val in value.items() if isinstance(val, str)]
        if not candidates:
            return None
        return max(candidates, key=lambda item: len(item[1]))[0]

    def _select_largest_list_path(self, value: Dict[str, Any]) -> Optional[List[str]]:
        """选出字典中最大列表字段的路径（支持嵌套）。

        使用迭代 DFS 遍历字典树，找到元素最多的列表所在路径。
        """
        best_path: Optional[List[str]] = None
        best_size = -1
        # 栈元素: (当前节点, 当前路径)
        stack: List[tuple[Any, List[str]]] = [(value, [])]
        while stack:
            node, path = stack.pop()
            if isinstance(node, list):
                if len(node) > best_size:
                    best_size = len(node)
                    best_path = path
                continue
            if not isinstance(node, dict):
                continue
            for key, child in node.items():
                if isinstance(key, str):
                    stack.append((child, path + [key]))
        return best_path if best_path else None

    def _read_nested_dict_value(
        self, value: Dict[str, Any], field_path: List[str]
    ) -> Any:
        """按 key 路径读取嵌套 dict 的值。"""
        target: Any = value
        for key in field_path:
            if not isinstance(target, dict):
                return None
            target = target.get(key)
        return target

    def _write_nested_dict_value(
        self, value: Dict[str, Any], field_path: List[str], data: Any
    ) -> bool:
        """按 key 路径写入嵌套 dict 的值。"""
        if not field_path:
            return False
        target: Any = value
        for key in field_path[:-1]:
            if not isinstance(target, dict):
                return False
            target = target.get(key)
        if not isinstance(target, dict):
            return False
        target[field_path[-1]] = data
        return True

    # ------------------------------------------------------------------
    # 截断实现
    # ------------------------------------------------------------------

    def _truncate_text_chars(
        self,
        *,
        text: str,
        limit: int,
        template: Optional[Dict[str, Any]],
        field_path: Optional[List[str]],
        context: TruncationContext,
        scope_hash: str,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """按字符数截断文本。"""
        total = len(text)
        if total <= limit:
            return text, None
        chunk = text[:limit]
        output_value = self._apply_chunk_to_template(template, field_path, chunk)
        cursor = self._store_cursor(
            tool_name="text", scope_hash=scope_hash,
            reason=TRUNCATION_STRATEGIES["text_chars"]["reason"],
            unit=TRUNCATION_STRATEGIES["text_chars"]["unit"],
            limit=limit, total=total, data=text, offset=len(chunk),
            template=template, field_path=field_path, mode="text", context=context,
        )
        truncation = self._build_truncation_info(
            cursor=cursor, reason=TRUNCATION_STRATEGIES["text_chars"]["reason"],
            limit=limit, unit=TRUNCATION_STRATEGIES["text_chars"]["unit"],
            total=total, has_more=True, scope_token=self._cursor_store.get(cursor, {}).get("scope_token"),
        )
        return output_value, truncation

    def _truncate_text_lines(
        self,
        *,
        text: str,
        limit: int,
        template: Optional[Dict[str, Any]],
        field_path: Optional[List[str]],
        context: TruncationContext,
        scope_hash: str,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """按行数截断文本。"""
        lines = text.splitlines(keepends=True)
        total = len(lines)
        if total <= limit:
            return text, None
        chunk_lines = lines[:limit]
        chunk = "".join(chunk_lines)
        output_value = self._apply_chunk_to_template(template, field_path, chunk)
        cursor = self._store_cursor(
            tool_name="text_lines", scope_hash=scope_hash,
            reason=TRUNCATION_STRATEGIES["text_lines"]["reason"],
            unit=TRUNCATION_STRATEGIES["text_lines"]["unit"],
            limit=limit, total=total, data=lines, offset=len(chunk_lines),
            template=template, field_path=field_path, mode="text_lines", context=context,
        )
        truncation = self._build_truncation_info(
            cursor=cursor, reason=TRUNCATION_STRATEGIES["text_lines"]["reason"],
            limit=limit, unit=TRUNCATION_STRATEGIES["text_lines"]["unit"],
            total=total, has_more=True, scope_token=self._cursor_store.get(cursor, {}).get("scope_token"),
        )
        return output_value, truncation

    def _truncate_list_items(
        self,
        *,
        items: List[Any],
        limit: int,
        template: Optional[Dict[str, Any]],
        field_path: Optional[List[str]],
        context: TruncationContext,
        scope_hash: str,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """按元素数截断列表。"""
        total = len(items)
        if total <= limit:
            return items, None
        chunk = items[:limit]
        output_value = self._apply_chunk_to_template(template, field_path, chunk)
        cursor = self._store_cursor(
            tool_name="list", scope_hash=scope_hash,
            reason=TRUNCATION_STRATEGIES["list_items"]["reason"],
            unit=TRUNCATION_STRATEGIES["list_items"]["unit"],
            limit=limit, total=total, data=items, offset=len(chunk),
            template=template, field_path=field_path, mode="list", context=context,
        )
        truncation = self._build_truncation_info(
            cursor=cursor, reason=TRUNCATION_STRATEGIES["list_items"]["reason"],
            limit=limit, unit=TRUNCATION_STRATEGIES["list_items"]["unit"],
            total=total, has_more=True, scope_token=self._cursor_store.get(cursor, {}).get("scope_token"),
        )
        return output_value, truncation

    def _truncate_binary_bytes(
        self,
        *,
        data: bytes,
        limit: int,
        template: Optional[Dict[str, Any]],
        field_path: Optional[List[str]],
        context: TruncationContext,
        scope_hash: str,
    ) -> tuple[Any, Optional[Dict[str, Any]]]:
        """按字节数截断二进制数据。"""
        total = len(data)
        if total <= limit:
            return data, None
        chunk = data[:limit]
        output_value = self._apply_chunk_to_template(template, field_path, chunk)
        cursor = self._store_cursor(
            tool_name="binary", scope_hash=scope_hash,
            reason=TRUNCATION_STRATEGIES["binary_bytes"]["reason"],
            unit=TRUNCATION_STRATEGIES["binary_bytes"]["unit"],
            limit=limit, total=total, data=data, offset=len(chunk),
            template=template, field_path=field_path, mode="binary", context=context,
        )
        truncation = self._build_truncation_info(
            cursor=cursor, reason=TRUNCATION_STRATEGIES["binary_bytes"]["reason"],
            limit=limit, unit=TRUNCATION_STRATEGIES["binary_bytes"]["unit"],
            total=total, has_more=True, scope_token=self._cursor_store.get(cursor, {}).get("scope_token"),
        )
        return output_value, truncation

    # ------------------------------------------------------------------
    # 模板与后处理
    # ------------------------------------------------------------------

    def _apply_chunk_to_template(
        self,
        template: Optional[Dict[str, Any]],
        field_path: Optional[List[str]],
        chunk: Any,
    ) -> Any:
        """将截断后的数据块填充到模板的对应字段位置。"""
        if template is None or field_path is None:
            return chunk
        value = copy.deepcopy(template)
        target = value
        for key in field_path[:-1]:
            if not isinstance(target, dict):
                return value
            target = target.get(key)
        if isinstance(target, dict):
            target[field_path[-1]] = chunk
        return value

    def _postprocess_truncated_value(
        self,
        tool_name: str,
        field_path: Optional[List[str]],
        value: Any,
        chunk: Any,
        *,
        length_field: Optional[str] = None,
    ) -> None:
        """截断后处理钩子（当前为 no-op）。"""
        return

    # ------------------------------------------------------------------
    # Scope hash 与游标管理
    # ------------------------------------------------------------------

    def _build_scope_hash(self, name: str, arguments: Dict[str, Any]) -> str:
        """为 (工具名, 参数) 生成确定性 SHA-256 哈希。"""
        try:
            payload = {"tool": name, "arguments": arguments}
            encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        except (TypeError, ValueError):
            encoded = f"{name}:{repr(arguments)}"
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _store_cursor(
        self,
        *,
        tool_name: str,
        scope_hash: str,
        reason: str,
        unit: str,
        limit: int,
        total: int,
        data: Any,
        offset: int,
        template: Optional[Dict[str, Any]],
        field_path: Optional[List[str]],
        mode: str,
        context: TruncationContext,
        length_field: Optional[str] = None,
    ) -> str:
        """创建游标并存入游标存储。

        Returns:
            新游标的唯一标识。
        """
        with self._lock:
            now = time.monotonic()
            ttl = _CURSOR_TTL_FALLBACK_SEC
            if context:
                timeout_seconds = context.timeout_seconds
                if isinstance(timeout_seconds, (int, float)) and timeout_seconds > 0:
                    ttl = float(timeout_seconds)
            cursor = uuid.uuid4().hex
            self._cleanup_expired_cursors(now)
            self._cursor_store[cursor] = {
                "tool_name": tool_name,
                "scope_hash": scope_hash,
                "reason": reason,
                "unit": unit,
                "limit": limit,
                "total": total,
                "data": data,
                "offset": offset,
                "template": template,
                "field_path": field_path,
                "mode": mode,
                "length_field": length_field,
                "created_at": now,
                "expires_at": now + ttl,
                "run_id": context.run_id if context else None,
                "iteration_id": context.iteration_id if context else None,
                "tool_call_id": context.tool_call_id if context else None,
            }
            self._cursor_store[cursor]["scope_token"] = self._build_scope_token(
                cursor=cursor,
                record=self._cursor_store[cursor],
            )
            return cursor

    def _cleanup_expired_cursors(self, now: Optional[float] = None) -> None:
        """清理已过期的游标。"""
        with self._lock:
            current = now if now is not None else time.monotonic()
            expired = [
                cid for cid, rec in self._cursor_store.items()
                if rec.get("expires_at", 0) <= current
            ]
            for cid in expired:
                self._cursor_store.pop(cid, None)

    # ------------------------------------------------------------------
    # 构建辅助
    # ------------------------------------------------------------------

    def _build_truncation_info(
        self,
        *,
        cursor: str,
        reason: str,
        limit: int,
        unit: str,
        total: int,
        has_more: bool,
        scope_token: Optional[str],
    ) -> Dict[str, Any]:
        """构建截断信息字典。

        Args:
            cursor: 续读游标。
            reason: 截断原因。
            limit: 截断阈值。
            unit: 截断单位。
            total: 原始总量估计值。
            has_more: 是否仍有剩余内容。
            scope_token: 续读作用域校验令牌。

        Returns:
            截断信息字典，包含兼容字段与续读增强字段。

        Raises:
            无。
        """

        fetch_more_args = self._build_fetch_more_args(cursor, scope_token)
        return {
            "reason": reason,
            "limit": limit,
            "unit": unit,
            "cursor": cursor,
            "has_more": has_more,
            "total_estimate": total,
            "fetch_more_args": fetch_more_args,
            "continuation_required": has_more,
            "continuation_priority": _CONTINUATION_PRIORITY_HIGH if has_more else None,
            "next_action": _CONTINUATION_ACTION_FETCH_MORE if has_more else None,
        }

    def _build_fetch_more_args(self, cursor: str, scope_token: Optional[str]) -> Dict[str, str]:
        """构建 `fetch_more` 推荐参数。

        Args:
            cursor: 截断游标。
            scope_token: 游标对应的作用域校验令牌。

        Returns:
            可直接用于 `fetch_more` 的参数字典。

        Raises:
            无。
        """

        args: Dict[str, str] = {"cursor": cursor}
        if isinstance(scope_token, str) and scope_token:
            args["scope_token"] = scope_token
        return args

    def _build_scope_token(self, *, cursor: str, record: Dict[str, Any]) -> str:
        """为游标构建强作用域校验令牌。

        Args:
            cursor: 游标字符串。
            record: 游标记录。

        Returns:
            作用域令牌（SHA-256 十六进制摘要）。

        Raises:
            RuntimeError: 构建失败时抛出。
        """

        payload = {
            "cursor": cursor,
            "scope_hash": record.get("scope_hash"),
            "run_id": record.get("run_id"),
            "iteration_id": record.get("iteration_id"),
            "tool_call_id": record.get("tool_call_id"),
            "created_at": record.get("created_at"),
        }
        encoded = json.dumps(payload, sort_keys=True, ensure_ascii=True, default=str)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def _build_chunk(
        self,
        *,
        mode: str,
        data: Any,
        offset: int,
        limit: int,
    ) -> tuple[Any, int]:
        """从原始数据中按 mode 取出下一块。"""
        if mode == "text":
            chunk = data[offset:offset + limit]
            return chunk, len(chunk)
        if mode == "text_lines":
            lines_chunk = data[offset:offset + limit]
            return "".join(lines_chunk), len(lines_chunk)
        if mode == "binary":
            chunk = data[offset:offset + limit]
            return chunk, len(chunk)
        # list 模式
        chunk = data[offset:offset + limit]
        return chunk, len(chunk)

    def _resolve_fetch_limit(self, requested: Any, record_limit: int) -> int:
        """确定 fetch_more 实际使用的 limit。"""
        if isinstance(requested, int) and requested > 0:
            return min(requested, record_limit)
        return record_limit

    def _validate_cursor_context(
        self,
        record: Dict[str, Any],
        context: TruncationContext,
    ) -> Optional[Dict[str, Any]]:
        """校验游标上下文是否与当前运行一致。

        设计说明：
        - `fetch_more` 常发生在模型看到截断结果后的下一次 agent iteration。
        - 因此仅校验 `run_id`，允许同一 run 内跨 iteration 续读。
        """
        if not context:
            return None

        expected_run_id = record.get("run_id")
        actual_run_id = context.run_id
        if expected_run_id and actual_run_id and expected_run_id != actual_run_id:
            return {
                "code": "cursor_scope_mismatch",
                "message": "cursor scope mismatch",
            }
        return None

    def _validate_scope_token(
        self,
        record: Dict[str, Any],
        arguments: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """校验 `fetch_more` 的 scope_token 是否与游标记录匹配。

        Args:
            record: 游标记录。
            arguments: `fetch_more` 调用参数。

        Returns:
            校验失败返回错误对象；通过返回 `None`。

        Raises:
            无。
        """

        expected = record.get("scope_token")
        if not isinstance(expected, str) or not expected:
            # 兼容历史游标记录（未携带 scope_token）。
            return None
        actual = arguments.get("scope_token")
        if not isinstance(actual, str) or not actual:
            return dict(_SCOPE_MISMATCH_ERROR)
        if actual != expected:
            return dict(_SCOPE_MISMATCH_ERROR)
        return None
