"""章节级写作合同与裁剪规则解析。

该模块负责从章节模板原文中提取结构化的隐藏规则：
- `CHAPTER_CONTRACT`：章节级写作边界与最小输出合同。
- `ITEM_RULE`：条目级条件写作规则。

所有解析与校验都留在 write pipeline 领域内，避免将章节写作语义泄漏到 engine 包。
`CHAPTER_CONTRACT` 当前固定为五字段最小 schema，其中 `preferred_lens` 使用有序对象列表，
便于按公司级 facet 做裁剪。
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field

import yaml

_HTML_COMMENT_PATTERN = re.compile(r"<!--(.*?)-->", re.DOTALL)
_HEADING_PATTERN = re.compile(r"^(#{2,6})\s+(.*\S)\s*$", re.MULTILINE)
_CHAPTER_CONTRACT_START = "CHAPTER_CONTRACT"
_CHAPTER_CONTRACT_END = "END_CHAPTER_CONTRACT"
_ITEM_RULE_START = "ITEM_RULE"
_ITEM_RULE_END = "END_ITEM_RULE"
_SUPPORTED_ITEM_RULE_MODES = {"conditional", "optional"}
_SUPPORTED_LENS_PRIORITIES = {"core", "supporting"}
_CHAPTER_CONTRACT_KEYS = {
    "narrative_mode",
    "must_answer",
    "must_not_cover",
    "required_output_items",
    "preferred_lens",
}


@dataclass(frozen=True)
class PreferredLens:
    """章节优先认知口径。

    Args:
        lens: 该条认知口径的具体内容。
        priority: 重要性，支持 `core` 与 `supporting`。
        facets_any: 命中任一 facet 即可保留的 facet 条件列表。

    Returns:
        无。

    Raises:
        无。
    """

    lens: str
    priority: str
    facets_any: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, object]:
        """转换为 prompt 可消费字典。

        Args:
            无。

        Returns:
            规则字典。

        Raises:
            无。
        """

        return {
            "lens": self.lens,
            "priority": self.priority,
            "facets_any": list(self.facets_any),
        }


@dataclass(frozen=True)
class ChapterContract:
    """章节级写作合同。

    Args:
        narrative_mode: 本章应采用的叙事组织方式。
        must_answer: 本章必须回答的问题。
        must_not_cover: 本章禁止展开的问题。
        required_output_items: 本章最小输出清单。
        preferred_lens: 本章优先采用的认知口径列表。

    Returns:
        无。

    Raises:
        无。
    """

    narrative_mode: str = ""
    must_answer: list[str] = field(default_factory=list)
    must_not_cover: list[str] = field(default_factory=list)
    required_output_items: list[str] = field(default_factory=list)
    preferred_lens: list[PreferredLens] = field(default_factory=list)

    @classmethod
    def empty(cls) -> "ChapterContract":
        """构造空章节合同。

        Args:
            无。

        Returns:
            空合同对象。

        Raises:
            无。
        """

        return cls()

    def to_prompt_fields(self) -> dict[str, object]:
        """转换为 prompt 输入字段。

        Args:
            无。

        Returns:
            面向 prompt 的字段字典。

        Raises:
            无。
        """

        return {
            "narrative_mode": self.narrative_mode,
            "must_answer": list(self.must_answer),
            "must_not_cover": list(self.must_not_cover),
            "required_output_items": list(self.required_output_items),
            "preferred_lens": [item.to_prompt_dict() for item in self.preferred_lens],
        }


@dataclass(frozen=True)
class ItemRule:
    """条目级条件写作规则。

    Args:
        mode: 规则模式，`conditional` 或 `optional`。
        target_heading: 规则绑定的目标标题。
        item: 需要在目标标题下按条件补充的输出项名称。
        when: 触发条件或适用条件说明。
        facets_any: 命中任一 facet 即可保留的 facet 条件列表。

    Returns:
        无。

    Raises:
        无。
    """

    mode: str
    target_heading: str
    item: str
    when: str
    facets_any: list[str] = field(default_factory=list)

    def to_prompt_dict(self) -> dict[str, str | list[str]]:
        """转换为 prompt 可消费字典。

        Args:
            无。

        Returns:
            规则字典。

        Raises:
            无。
        """

        return {
            "mode": self.mode,
            "target_heading": self.target_heading,
            "item": self.item,
            "when": self.when,
            "facets_any": list(self.facets_any),
        }


def extract_chapter_contract(chapter_content: str, *, chapter_title: str) -> ChapterContract:
    """从章节原文中提取 `CHAPTER_CONTRACT`。

    Args:
        chapter_content: 章节完整原文（保留 HTML 注释）。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        解析后的章节合同；若未声明则返回空合同。

    Raises:
        ValueError: 当合同块重复、缺字段或 YAML 非法时抛出。
    """

    payloads = _extract_named_comment_payloads(
        chapter_content=chapter_content,
        start_marker=_CHAPTER_CONTRACT_START,
        end_marker=_CHAPTER_CONTRACT_END,
        chapter_title=chapter_title,
    )
    if not payloads:
        return ChapterContract.empty()
    if len(payloads) > 1:
        raise ValueError(f"章节 {chapter_title!r} 存在多个 CHAPTER_CONTRACT，无法判定唯一合同")
    raw_data = _load_yaml_payload(payloads[0], block_name=_CHAPTER_CONTRACT_START, chapter_title=chapter_title)
    return _parse_chapter_contract_data(raw_data, chapter_title=chapter_title)


def extract_item_rules(chapter_content: str, *, chapter_title: str) -> list[ItemRule]:
    """从章节原文中提取全部 `ITEM_RULE`。

    Args:
        chapter_content: 章节完整原文（保留 HTML 注释）。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        规则列表；若未声明则返回空列表。

    Raises:
        ValueError: 当规则块非法、找不到绑定标题或 YAML 非法时抛出。
    """

    rules: list[ItemRule] = []
    for match in _HTML_COMMENT_PATTERN.finditer(chapter_content):
        comment_body = match.group(1)
        payload = _extract_named_comment_payload_from_body(
            comment_body=comment_body,
            start_marker=_ITEM_RULE_START,
            end_marker=_ITEM_RULE_END,
            chapter_title=chapter_title,
        )
        if payload is None:
            continue
        raw_data = _load_yaml_payload(payload, block_name=_ITEM_RULE_START, chapter_title=chapter_title)
        target_heading = _find_previous_heading_before_position(chapter_content, end_index=match.start())
        if not target_heading:
            raise ValueError(f"章节 {chapter_title!r} 的 ITEM_RULE 未绑定到前序标题，无法确定作用目标")
        rules.append(_parse_item_rule_data(raw_data, chapter_title=chapter_title, target_heading=target_heading))
    return rules


def _extract_named_comment_payloads(
    *,
    chapter_content: str,
    start_marker: str,
    end_marker: str,
    chapter_title: str,
) -> list[str]:
    """提取指定命名注释块的 YAML payload 列表。

    Args:
        chapter_content: 章节原文。
        start_marker: 起始标记。
        end_marker: 结束标记。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        payload 文本列表。

    Raises:
        ValueError: 当注释块标记不闭合时抛出。
    """

    payloads: list[str] = []
    for match in _HTML_COMMENT_PATTERN.finditer(chapter_content):
        payload = _extract_named_comment_payload_from_body(
            comment_body=match.group(1),
            start_marker=start_marker,
            end_marker=end_marker,
            chapter_title=chapter_title,
        )
        if payload is not None:
            payloads.append(payload)
    return payloads


def _extract_named_comment_payload_from_body(
    *,
    comment_body: str,
    start_marker: str,
    end_marker: str,
    chapter_title: str,
) -> str | None:
    """从单个 HTML 注释正文中提取命名块 payload。

    Args:
        comment_body: 注释正文。
        start_marker: 起始标记。
        end_marker: 结束标记。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        payload 文本；若该注释块不是目标块则返回 ``None``。

    Raises:
        ValueError: 当起止标记不匹配时抛出。
    """

    normalized = textwrap.dedent(comment_body).strip()
    if not normalized:
        return None
    lines = normalized.splitlines()
    if lines[0].strip() != start_marker:
        return None
    if lines[-1].strip() != end_marker:
        raise ValueError(f"章节 {chapter_title!r} 的 {start_marker} 注释缺少结束标记 {end_marker}")
    payload_lines = lines[1:-1]
    return "\n".join(payload_lines).strip()


def _load_yaml_payload(payload: str, *, block_name: str, chapter_title: str) -> dict[str, object]:
    """解析注释块中的 YAML payload。

    Args:
        payload: YAML 文本。
        block_name: 块名称，仅用于错误提示。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        解析后的 YAML 对象。

    Raises:
        ValueError: 当 YAML 非法时抛出。
    """

    if not payload:
        raise ValueError(f"章节 {chapter_title!r} 的 {block_name} 为空，无法生成结构化规则")
    try:
        parsed = yaml.safe_load(payload)
    except yaml.YAMLError as exc:
        raise ValueError(f"章节 {chapter_title!r} 的 {block_name} YAML 非法: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"章节 {chapter_title!r} 的 {block_name} 必须解析为映射")
    return parsed


def _parse_chapter_contract_data(raw_data: dict[str, object], *, chapter_title: str) -> ChapterContract:
    """校验并构造章节合同对象。

    Args:
        raw_data: YAML 解析结果。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        `ChapterContract` 对象。

    Raises:
        ValueError: 当字段缺失、类型错误时抛出。
    """

    _validate_chapter_contract_keys(raw_data, chapter_title=chapter_title)
    narrative_mode = _require_string(
        raw_data,
        key="narrative_mode",
        chapter_title=chapter_title,
        block_name=_CHAPTER_CONTRACT_START,
    )
    must_answer = _require_string_list(raw_data, key="must_answer", chapter_title=chapter_title)
    must_not_cover = _require_string_list(raw_data, key="must_not_cover", chapter_title=chapter_title)
    required_output_items = _require_string_list(
        raw_data,
        key="required_output_items",
        chapter_title=chapter_title,
    )
    preferred_lens = _require_preferred_lens_list(raw_data, key="preferred_lens", chapter_title=chapter_title)
    return ChapterContract(
        narrative_mode=narrative_mode,
        must_answer=must_answer,
        must_not_cover=must_not_cover,
        required_output_items=required_output_items,
        preferred_lens=preferred_lens,
    )


def _validate_chapter_contract_keys(raw_data: dict[str, object], *, chapter_title: str) -> None:
    """校验章节合同字段集合是否为最小稳定 schema。

    Args:
        raw_data: 原始章节合同映射。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        无。

    Raises:
        ValueError: 当存在未声明字段时抛出。
    """

    unexpected_keys = sorted(set(raw_data.keys()) - _CHAPTER_CONTRACT_KEYS)
    if unexpected_keys:
        joined_keys = ", ".join(unexpected_keys)
        raise ValueError(
            f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT 包含未支持字段: {joined_keys}；"
            "当前仅允许五字段最小 schema"
        )


def _parse_item_rule_data(raw_data: dict[str, object], *, chapter_title: str, target_heading: str) -> ItemRule:
    """校验并构造条目级条件写作规则。

    Args:
        raw_data: YAML 解析结果。
        chapter_title: 章节标题，仅用于错误提示。
        target_heading: 规则绑定的目标标题。

    Returns:
        `ItemRule` 对象。

    Raises:
        ValueError: 当字段缺失、值非法或存在未知字段时抛出。
    """

    expected_keys = {"mode", "item", "when", "facets_any"}
    unexpected_keys = sorted(set(raw_data.keys()) - expected_keys)
    if unexpected_keys:
        joined_keys = ", ".join(unexpected_keys)
        raise ValueError(f"章节 {chapter_title!r} 的 ITEM_RULE 包含未支持字段: {joined_keys}")
    mode = _require_string(raw_data, key="mode", chapter_title=chapter_title, block_name=_ITEM_RULE_START)
    item = _require_string(raw_data, key="item", chapter_title=chapter_title, block_name=_ITEM_RULE_START)
    when = _require_string(raw_data, key="when", chapter_title=chapter_title, block_name=_ITEM_RULE_START)
    facets_any = _require_optional_string_list(raw_data, key="facets_any", chapter_title=chapter_title)
    if mode not in _SUPPORTED_ITEM_RULE_MODES:
        raise ValueError(f"章节 {chapter_title!r} 的 ITEM_RULE mode={mode!r} 不受支持")
    return ItemRule(
        mode=mode,
        target_heading=target_heading,
        item=item,
        when=when,
        facets_any=facets_any,
    )


def _require_string_list(raw_data: dict[str, object], *, key: str, chapter_title: str) -> list[str]:
    """读取并校验字符串列表字段。

    Args:
        raw_data: 原始映射。
        key: 字段名。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        字符串列表。

    Raises:
        ValueError: 当字段缺失或类型非法时抛出。
    """

    value = raw_data.get(key)
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key} 必须为字符串列表")
    return [item.strip() for item in value]


def _require_preferred_lens_list(raw_data: dict[str, object], *, key: str, chapter_title: str) -> list[PreferredLens]:
    """读取并校验 `preferred_lens` 对象列表字段。

    Args:
        raw_data: 原始映射。
        key: 字段名。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        `PreferredLens` 列表。

    Raises:
        ValueError: 当字段缺失或类型非法时抛出。
    """

    value = raw_data.get(key)
    if isinstance(value, dict):
        # 为了平滑过渡，仍接受旧映射格式并在解析期转换为新对象列表。
        normalized_from_mapping: list[PreferredLens] = []
        for lens_group, lens_items in value.items():
            if not isinstance(lens_group, str) or not isinstance(lens_items, list):
                raise ValueError(
                    f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key} 必须为对象列表；"
                    "若仍使用旧映射格式，值必须为字符串列表"
                )
            if any(not isinstance(item, str) for item in lens_items):
                raise ValueError(f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key}.{lens_group} 必须为字符串列表")
            facets_any = _legacy_lens_group_to_facets(lens_group.strip(), chapter_title=chapter_title)
            priority = "core" if lens_group.strip() == "default" else "supporting"
            for item in lens_items:
                normalized_from_mapping.append(
                    PreferredLens(
                        lens=item.strip(),
                        priority=priority,
                        facets_any=facets_any,
                    )
                )
        return normalized_from_mapping
    if not isinstance(value, list):
        raise ValueError(f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key} 必须为对象列表")
    normalized: list[PreferredLens] = []
    for index, raw_lens in enumerate(value):
        if not isinstance(raw_lens, dict):
            raise ValueError(
                f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key}[{index}] 必须为映射"
            )
        expected_keys = {"lens", "priority", "facets_any"}
        unexpected_keys = sorted(set(raw_lens.keys()) - expected_keys)
        if unexpected_keys:
            joined_keys = ", ".join(unexpected_keys)
            raise ValueError(
                f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key}[{index}] 包含未支持字段: {joined_keys}"
            )
        lens = _require_string(raw_lens, key="lens", chapter_title=chapter_title, block_name=f"{_CHAPTER_CONTRACT_START}.{key}")
        priority = _require_string(
            raw_lens,
            key="priority",
            chapter_title=chapter_title,
            block_name=f"{_CHAPTER_CONTRACT_START}.{key}",
        )
        if priority not in _SUPPORTED_LENS_PRIORITIES:
            raise ValueError(
                f"章节 {chapter_title!r} 的 CHAPTER_CONTRACT.{key}[{index}].priority={priority!r} 不受支持"
            )
        facets_any = _require_optional_string_list(raw_lens, key="facets_any", chapter_title=chapter_title)
        normalized.append(PreferredLens(lens=lens, priority=priority, facets_any=facets_any))
    return normalized


def _legacy_lens_group_to_facets(group_name: str, *, chapter_title: str) -> list[str]:
    """把旧 `preferred_lens` 分组名映射为 facet 条件。

    Args:
        group_name: 旧分组名。
        chapter_title: 章节标题，仅用于错误提示。

    Returns:
        facet 列表；`default` 返回空列表。

    Raises:
        ValueError: 当旧分组名未知且无法稳定映射时抛出。
    """

    if group_name == "default":
        return []
    mapping = {
        "platform_or_internet_company": ["平台互联网"],
        "consumer_brand_company": ["消费品牌/零售"],
        "pharma_biotech_company": ["生物制药"],
        "industrial_or_manufacturing_company": ["工业制造/关键部件"],
        "software_or_data_company": ["企业软件", "垂直软件/创意软件", "数据基础设施/数据中心"],
        "semiconductor_or_chip_company": ["半导体设计", "半导体设备/制造"],
        "financial_or_market_infrastructure_company": ["支付/金融基础设施", "交易所/市场基础设施"],
        "bank_or_diversified_financial_company": ["支付/金融基础设施"],
        "insurance_or_broker_company": ["保险"],
        "healthcare_service_or_provider_company": ["医疗服务"],
        "energy_or_resource_company": ["能源"],
        "transport_or_asset_network_company": ["REIT/基础设施"],
        "telecom_or_connectivity_company": ["REIT/基础设施"],
        "media_or_content_company": ["广告媒体"],
        "utility_or_reit_or_infrastructure_company": ["REIT/基础设施", "数据基础设施/数据中心"],
    }
    if group_name not in mapping:
        raise ValueError(
            f"章节 {chapter_title!r} 的 preferred_lens 旧分组 {group_name!r} 无法映射为 facet，请改为新对象列表格式"
        )
    return list(mapping[group_name])


def _require_optional_string_list(raw_data: dict[str, object], *, key: str, chapter_title: str) -> list[str]:
    """读取可选字符串列表字段。

    Args:
        raw_data: 原始映射。
        key: 字段名。
        chapter_title: 章节标题，仅用于错误提示。

        Returns:
            标准化后的字符串列表；缺失时返回空列表。

        Raises:
            ValueError: 当字段类型非法时抛出。
    """

    value = raw_data.get(key)
    if value is None:
        return []
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"章节 {chapter_title!r} 的 {key} 必须为字符串列表")
    return [item.strip() for item in value if item.strip()]


def _require_string(
    raw_data: dict[str, object],
    *,
    key: str,
    chapter_title: str,
    block_name: str,
) -> str:
    """读取并校验字符串字段。

    Args:
        raw_data: 原始映射。
        key: 字段名。
        chapter_title: 章节标题，仅用于错误提示。
        block_name: 块名称，仅用于错误提示。

    Returns:
        去首尾空白后的字符串。

    Raises:
        ValueError: 当字段缺失或类型非法时抛出。
    """

    value = raw_data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"章节 {chapter_title!r} 的 {block_name}.{key} 必须为非空字符串")
    return value.strip()


def _find_previous_heading_before_position(chapter_content: str, *, end_index: int) -> str:
    """查找给定位置前的最近一个 Markdown 标题。

    Args:
        chapter_content: 章节完整原文。
        end_index: 截止字符位置（不含）。

    Returns:
        标题文本；若找不到则返回空字符串。

    Raises:
        无。
    """

    prefix_text = chapter_content[:end_index]
    last_heading = ""
    for match in _HEADING_PATTERN.finditer(prefix_text):
        last_heading = match.group(2).strip()
    return last_heading
