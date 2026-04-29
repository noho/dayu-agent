"""6-K 表单公共常量、marker 与报表分类逻辑。

本模块只保留 6-K 专属语义：章节 marker、报表标题分类、行信号阈值等，
供 ``BsSixKFormProcessor``（BeautifulSoup 路线）和将来的
``SixKFormProcessor``（edgartools 路线）共同使用。

HTML 财务表结构化核心已下沉到 ``html_financial_statement_common``，
本模块只负责向共享层注入 6-K 专属参数。

两侧处理器均从本模块 import，互不依赖，保持架构独立。

6-K 是外国私人发行人 (Foreign Private Issuer, FPI) 向 SEC 提交的
当期报告表单。典型 6-K 内容为**季度/年度业绩新闻稿** (Press Release)，
随 Exhibit 99.1 附件提交。

与 10-K/10-Q 不同，6-K 没有法定 Item 结构 (Regulation S-K 不适用)；
章节切分依赖**语义关键词**识别新闻稿的常见结构段落：

    Exhibit → Financial Results → Key Highlights → Recent Development
    → Conference Call → Safe Harbor → About Non-GAAP → About Company
    → Financial Statements (Balance Sheets, Income, Reconciliation) → Notes

SEC 相关规则依据：
- SEC Form 6-K (17 CFR §249.306)
- SEC Release No. 34-97879 (Tailored Shareholder Reports)

设计决策：
- 不使用 "Exhibit 99.X" 作为章节标记 — 它是 EDGAR 附件编号，非内容标题，
  其文本归入 Cover Page。
- 财务报表标题 (Balance Sheets / Income / Reconciliation) 作为独立段，
  方便 LLM 按报表类型精准定位。
- 财务报表标记取末次匹配：叙述段常内联引用 "statement of operations" 等短语，
  实际表格标题出现在文档后半段。取末次匹配避免内联引用切断叙述段。
- 搜索实现 token 级回退：多词查询精确匹配失败时，自动降级为单词 OR 匹配，
  提升搜索召回率。
"""

from __future__ import annotations

import html
import re
from typing import Any, Optional

import pandas as pd

from dayu.engine.processors.text_utils import normalize_whitespace as _normalize_whitespace
from dayu.engine.processors.table_utils import parse_html_table_dataframe

from .financial_base import FinancialStatementResult
from .html_financial_statement_common import (
    _normalize_year_token,
    _resolve_period_end_from_fiscal_period,
    build_html_statement_result_from_tables as _build_shared_statement_result_from_tables,
    select_html_statement_tables_by_row_signals as _select_shared_statement_tables_by_row_signals,
)
from .sec_xbrl_query import build_statement_locator
from .sec_form_section_common import _dedupe_markers
from .sec_report_form_common import _find_table_of_contents_cutoff

# ---------------------------------------------------------------------------
# 6-K 表单类型集合
# ---------------------------------------------------------------------------
_SIX_K_FORMS = frozenset({"6-K"})

# ---------------------------------------------------------------------------
# 6-K 章节标记模式
#
# 设计原则：
#   1. 只保留**强语义边界**——匹配命中即表示"此处开始新话题"。
#   2. 叙述段取第一个匹配；财务报表段取**最后一个匹配**，
#      避免内联引用（如 "included in the statement of operations"）
#      被误判为章节标题。
#   3. 不使用 "Exhibit 99.X"——它是 EDGAR 附件编号，非内容标题。
#   4. 公司名匹配泛化到常见法律后缀 (SEC EDGAR filer naming conventions)。
#   5. 财务报表标题取自 SEC Regulation S-X 常见标准名称。
#
# 元组结构：(title, pattern, use_last_match)
#   use_last_match=False — 取首次匹配（叙述段）
#   use_last_match=True  — 取末次匹配（财务报表段，跳过内联引用）
# ---------------------------------------------------------------------------

_SIX_K_MARKERS: tuple[tuple[str, re.Pattern[str], bool], ...] = (
    # ── 业务叙述段（取首次匹配） ──
    (
        "Financial Results and Business Updates",
        re.compile(r"(?i)\bfinancial\s+results(?:\s+and\s+business\s+updates?)?\b"),
        False,
    ),
    (
        "Key Highlights",
        re.compile(r"(?i)\bkey\s+highlights\b"),
        False,
    ),
    (
        "Recent Development",
        re.compile(r"(?i)\brecent\s+developments?\b"),
        False,
    ),
    (
        "Business Outlook",
        re.compile(r"(?i)\b(?:business|operating)\s+outlook\b"),
        False,
    ),
    # ── 电话会 / 免责 / 说明段（取首次匹配） ──
    (
        "Conference Call",
        re.compile(r"(?i)\bconference\s+call\b"),
        False,
    ),
    (
        "Safe Harbor",
        re.compile(r"(?i)\bsafe\s+harbor\b"),
        False,
    ),
    (
        "About Non-GAAP",
        re.compile(r"(?i)\babout\s+non[- ]?gaap\b"),
        False,
    ),
    # 泛化公司名匹配：覆盖 Limited / Inc / Corp / Holdings / Group / Co / Ltd
    # 负向前瞻排除 "About Non-GAAP..." 避免内联干扰
    (
        "About Company",
        re.compile(
            r"(?i)\babout\s+(?!non[- ]?gaap)"
            r"[A-Za-z0-9\.\-\, ]+?"
            r"(?:limited|inc\.?|corp(?:oration)?\.?|holdings?|group|co\.?|ltd\.?)\b"
        ),
        False,
    ),
    # ── 财务报表段 (SEC Regulation S-X 标准名称) ──
    # use_last_match=True: 新闻稿叙述部分常内联引用 "statement of operations"
    # / "balance sheet" 等短语，实际表格标题出现在文档后半段。
    # 取末次匹配避免内联引用切断叙述段。
    (
        "Balance Sheets",
        re.compile(
            r"(?i)\b(?:consolidated\s+)?(?:condensed\s+)?balance\s+sheets?\b"
        ),
        True,
    ),
    (
        "Statements of Income",
        re.compile(
            r"(?i)\b(?:consolidated\s+)?(?:condensed\s+)?statements?\s+of"
            r"\s+(?:comprehensive\s+)?(?:income|operations)\b"
        ),
        True,
    ),
    (
        "Reconciliation",
        re.compile(
            r"(?i)\breconciliations?\s+of\s+(?:GAAP|non[- ]?GAAP)\b"
        ),
        True,
    ),
    # ── 尾段（取首次匹配） ──
    (
        "Notes",
        re.compile(
            r"(?i)\bnotes?\s+for\s+all\s+the\s+"
            r"(?:condensed\s+)?consolidated\s+financial\s+schedules\s+presented\b"
        ),
        False,
    ),
)

_SIX_K_REPORT_MODE_MIN_HEADINGS = 4
_SIX_K_REPORT_HEADING_MARKERS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "About this report",
        re.compile(r"(?im)^\s*About\s+this\s+report\s*$"),
    ),
    (
        "Overview",
        re.compile(r"(?im)^\s*Overview\s*$"),
    ),
    (
        "Governance",
        re.compile(r"(?im)^\s*Governance\s*$"),
    ),
    (
        "Strategy",
        re.compile(r"(?im)^\s*Strategy\s*$"),
    ),
    (
        "Environment",
        re.compile(r"(?im)^\s*Environment\s*$"),
    ),
    (
        "Social",
        re.compile(r"(?im)^\s*Social\s*$"),
    ),
    (
        "Risk management",
        re.compile(
            r"(?im)^\s*(?:Managing\s+sustainability\s+and\s+climate\s+risks|Risk\s+management)\s*$"
        ),
    ),
    (
        "Appendix",
        re.compile(r"(?im)^\s*Appendix(?:\s+\d+)?\s*$"),
    ),
)

_SUPPORTED_STATEMENT_TYPES = frozenset({
    "income",
    "balance_sheet",
    "cash_flow",
    "equity",
    "comprehensive_income",
})
_SECTION_TITLE_BY_STATEMENT_TYPE: dict[str, str] = {
    "balance_sheet": "Balance Sheets",
    "income": "Statements of Income",
    "reconciliation": "Reconciliation",
}
_PRIMARY_EXTRACTABLE_STATEMENT_TYPES = frozenset({"income", "balance_sheet", "cash_flow"})
_STATEMENT_CLASSIFICATION_CONTEXT_WINDOW = 320
_STATEMENT_CLASSIFICATION_MIN_SCORE = 4
_STATEMENT_CLASSIFICATION_CONTEXT_ONLY_MIN_SCORE = 2
_STATEMENT_CLASSIFICATION_WEIGHTS: dict[str, int] = {
    "caption": 6,
    "headers": 4,
    "context": 1,
}
_STATEMENT_LINE_ITEM_WEIGHT = 2
_ROW_SIGNAL_MIN_HITS_BY_STATEMENT: dict[str, int] = {
    "income": 2,
    "balance_sheet": 2,
    "cash_flow": 1,
}

_NON_STATEMENT_TABLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\btable\s+of\s+contents\b"),
    re.compile(r"(?i)\bexhibit\s+index\b"),
    re.compile(r"(?i)\bindex\s+to\s+exhibits\b"),
    re.compile(r"(?i)\bsignatures?\b"),
    re.compile(r"(?i)\bcontacts?\b"),
)

_STATEMENT_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "balance_sheet": (
        re.compile(r"(?i)\bbalance\s+sheets?\b"),
        re.compile(r"(?i)\bstatement(?:s)?\s+of\s+financial\s+position\b"),
        re.compile(r"(?i)\bstatement(?:s)?\s+of\s+assets?\s+and\s+liabilit(?:y|ies)\b"),
        re.compile(r"(?i)\bfinancial\s+position\b"),
        re.compile(r"(?i)\bassets?\s+and\s+liabilit(?:y|ies)\b"),
        re.compile(r"(?i)\bcapital\s+and\s+liabilit(?:y|ies)\b"),
        re.compile(r"(?i)\b(?:balanc(?:e|o)|situaci[oó]n)\s+financier[ao]\b"),
    ),
    "income": (
        re.compile(r"(?i)\b(?:statement(?:s)?|income)\s+statements?\b"),
        re.compile(r"(?i)\bstatement(?:s)?\s+of\s+(?:income|operations|profit\s+or\s+loss)\b"),
        re.compile(r"(?i)\bstatement(?:s)?\s+of\s+comprehensive\s+income\b"),
        re.compile(r"(?i)\b(?:net\s+)?profit\s+for\s+the\s+period\b"),
        re.compile(
            r"(?i)\b(?:standalone|consolidated|condensed)\s+financial\s+results\b"
        ),
        re.compile(r"(?i)\bprofit\s+and\s+loss\b"),
        re.compile(r"(?i)\bestado(?:s)?\s+de\s+resultados?\b"),
    ),
    "cash_flow": (
        re.compile(r"(?i)\bcash\s+flows?\b"),
        re.compile(r"(?i)\bstatement(?:s)?\s+of\s+cash\s+flows?\b"),
        re.compile(r"(?i)\b(?:fluxo|flujos)\s+de\s+caja\b"),
    ),
    "reconciliation": (
        re.compile(r"(?i)\breconciliation(?:s)?\b"),
        re.compile(r"(?i)\bnon[- ]?gaap\b"),
        re.compile(r"(?i)\bgaap\b"),
    ),
}
_STATEMENT_LINE_ITEM_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "balance_sheet": (
        re.compile(r"(?i)\bcapital\b"),
        re.compile(r"(?i)\breserves?\s+and\s+surplus\b"),
        re.compile(r"(?i)\bdeposits?\b"),
        re.compile(r"(?i)\badvances?\b"),
        re.compile(r"(?i)\bborrowings?\b"),
        re.compile(r"(?i)\bother\s+liabilities\s+and\s+provisions\b"),
        re.compile(r"(?i)\bcash\s+and\s+balances?\s+with\s+reserve\s+bank\b"),
        re.compile(r"(?i)\bbalances?\s+with\s+banks?\s+and\s+money\s+at\s+call\b"),
        re.compile(r"(?i)\btotal\s+assets?\b"),
        re.compile(r"(?i)\btotal\s+(?:current\s+)?liabilit(?:y|ies)\b"),
        re.compile(r"(?i)\b(?:non[- ]?current|current)\s+assets?\b"),
        re.compile(r"(?i)\b(?:shareholders'?|stockholders'?)\s+equity\b"),
        re.compile(r"(?i)\btotal\s+equity\b"),
    ),
    "income": (
        re.compile(r"(?i)\btotal\s+revenue\b"),
        re.compile(r"(?i)\bnet\s+sales\b"),
        re.compile(r"(?i)\brevenue\b"),
        re.compile(r"(?i)\bcost\s+of\s+sales\b"),
        re.compile(r"(?i)\bgross\s+profit\b"),
        re.compile(r"(?i)\bincome\s+from\s+operations\b"),
        re.compile(r"(?i)\boperating\s+(?:income|loss|profit)\b"),
        re.compile(r"(?i)\binterest\s+earned\b"),
        re.compile(r"(?i)\binterest\s+expended\b"),
        re.compile(r"(?i)\bnet\s+interest\s+income\b"),
        re.compile(r"(?i)\btotal\s+income\b"),
        re.compile(r"(?i)\bnet\s+(?:income|loss|profit)\b"),
        re.compile(r"(?i)\bprofit\s+attributable\s+to\b"),
        re.compile(r"(?i)\b(?:profit|loss)\s+for\s+the\s+period\b"),
        re.compile(r"(?i)\bearnings\s+per\s+share\b"),
    ),
    "cash_flow": (
        re.compile(r"(?i)\bcash\s+flows?\s+from\s+operating\b"),
        re.compile(r"(?i)\boperating\s+activities\b"),
        re.compile(r"(?i)\bnet\s+cash\b"),
    ),
    "reconciliation": (),
}
_OCR_STATEMENT_TITLE_PATTERNS: dict[str, tuple[re.Pattern[str], ...]] = {
    "balance_sheet": _STATEMENT_PATTERNS["balance_sheet"],
    "income": _STATEMENT_PATTERNS["income"]
    + (
        re.compile(r"(?i)\bconsolidated\s+results\b"),
        re.compile(r"(?i)\bquarterly\s+results\b"),
        re.compile(r"(?i)\bfinancial\s+highlights\b"),
        re.compile(r"(?i)\b(?:key\s+)?summary\s+financials\b"),
        re.compile(r"(?i)\bgroup\s+results\b"),
    ),
    "cash_flow": _STATEMENT_PATTERNS["cash_flow"],
}
_OCR_INCOME_SUMMARY_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?i)\bprofit\s*&\s*loss\b"),
)
_OCR_DATE_RE = re.compile(
    r"(?i)\b(?:year|years|three\s+months|six\s+months|nine\s+months|period|periods)?\s*"
    r"(?:ended|ending|as\s+at)?\s*"
    r"(January|February|March|April|May|June|July|August|September|October|November|December)"
    r"\s+(\d{1,2}),\s*(\d{4})"
)
_OCR_YEAR_RE = re.compile(r"\b(19\d{2}|20\d{2}|2100)\b")
_OCR_CURRENCY_SCALE_RE = re.compile(
    r"(?i)(?:\b(US\$|HK\$|RMB|CNY|USD|EUR|GBP|JPY|CHF|AUD|CAD|NT\$|S\$)\b|(€))"
    r"(?:\s*[’'`]?\s*0{3,6}|\s+in\s+(thousands|millions|billions))?"
)
_OCR_VALUE_TOKEN_RE = re.compile(r"\((?:\d[\d,]*)(?:\.\d+)?\)|\d[\d,]*(?:\.\d+)?|[–—-]")
_OCR_PERCENT_TOKEN_RE = re.compile(r"%")
_OCR_MONTH_TO_NUMBER: dict[str, int] = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}
_OCR_CURRENCY_CODE_MAP: dict[str, str] = {
    "US$": "USD",
    "HK$": "HKD",
    "€": "EUR",
    "RMB": "CNY",
    "CNY": "CNY",
    "USD": "USD",
    "EUR": "EUR",
    "GBP": "GBP",
    "JPY": "JPY",
    "CHF": "CHF",
    "AUD": "AUD",
    "CAD": "CAD",
    "NT$": "TWD",
    "S$": "SGD",
}
_OCR_SUMMARY_LABEL_LOOKAHEAD_CHARS = 96
_OCR_SUMMARY_BODY_WINDOW_CHARS = 4000
_OCR_SUMMARY_MIN_ABS_VALUE = 50.0
_OCR_SUMMARY_TITLE_LOOKBACK_CHARS = 160
_OCR_SUMMARY_TITLE_LOOKAHEAD_CHARS = 96
_OCR_LINE_ITEM_PATTERNS_BY_STATEMENT: dict[str, tuple[tuple[str, re.Pattern[str]], ...]] = {
    "income": (
        ("Revenue", re.compile(r"(?i)\brevenue\b")),
        ("Total revenue", re.compile(r"(?i)\btotal\s+revenue\b")),
        ("Total revenues", re.compile(r"(?i)\btotal\s+revenues\b")),
        ("Net sales", re.compile(r"(?i)\bnet\s+sales\b")),
        ("Cost of sales", re.compile(r"(?i)\bcost\s+of\s+sales\b")),
        ("Gross profit", re.compile(r"(?i)\bgross\s+profit\b")),
        ("Other income and gains", re.compile(r"(?i)\bother\s+income\s+and\s+gains\b")),
        ("EBITDA", re.compile(r"(?i)\bebitda\b")),
        ("Operating income", re.compile(r"(?i)\boperating\s+income\b")),
        ("Operating expenses", re.compile(r"(?i)\boperating\s+expenses\b")),
        ("Income from operations", re.compile(r"(?i)\bincome\s+from\s+operations\b")),
        ("Selling and distribution expenses", re.compile(r"(?i)\bselling\s+and\s+distribution\s+expenses\b")),
        ("Administrative expenses", re.compile(r"(?i)\badministrative\s+expenses\b")),
        ("Research and development expenses", re.compile(r"(?i)\bresearch\s+and\s+development\s+expenses\b")),
        ("Other expenses", re.compile(r"(?i)\bother\s+expenses\b")),
        ("Finance costs", re.compile(r"(?i)\bfinance\s+costs\b")),
        ("Share of profit or loss of a joint venture", re.compile(r"(?i)\bshare\s+of\s+\(?loss\)?/?profit\s+of\s+a\s+joint\s+venture\b")),
        ("Operating profit or loss before tax", re.compile(r"(?i)\boperating\s+profit\s*/?\s*\(?loss\)?\s+before\s+tax\b")),
        ("Income before tax", re.compile(r"(?i)\bincome\s+before\s+tax\b")),
        ("Profit or loss before tax", re.compile(r"(?i)\b(?:profit|loss)\s+before\s+tax\b")),
        ("Income tax expense or credit", re.compile(r"(?i)\bincome\s+tax\s+\(?expense\)?/?credit\b")),
        ("Net income", re.compile(r"(?i)\bnet\s+income\b")),
        ("Profit or loss for the year", re.compile(r"(?i)\b(?:profit|loss)\s+for\s+the\s+year\b")),
        ("Net profit attributable to shareholders", re.compile(r"(?i)\bnet\s+profit\s*/?\s*\(?loss\)?\s+attributable\s+to\s+shareholders\b")),
        ("Owners of the parent", re.compile(r"(?i)\bowners\s+of\s+the\s+parent\b")),
        ("Non-controlling interests", re.compile(r"(?i)\bnon\s*-\s*controlling\s+interests\b")),
        ("Earnings per share", re.compile(r"(?i)\bearnings\s+per\s+share\b")),
        ("Basic and diluted earnings per share", re.compile(r"(?i)\bbasic\s+and\s+diluted\b.*?\b(?:earnings|loss)\s+per\s+share\b")),
    ),
    "balance_sheet": (
        ("Property, plant and equipment", re.compile(r"(?i)\bproperty,\s*plant\s+and\s+equipment\b")),
        ("Right-of-use assets", re.compile(r"(?i)\bright\s*-\s*of\s*-\s*use\s+assets\b")),
        ("Goodwill", re.compile(r"(?i)\bgoodwill\b")),
        ("Other intangible assets", re.compile(r"(?i)\bother\s+intangible\s+assets\b")),
        ("Investment in a joint venture", re.compile(r"(?i)\binvestment\s+in\s+a\s+joint\s+venture\b")),
        ("Financial assets at fair value through profit or loss", re.compile(r"(?i)\bfinancial\s+assets\s+at\s+fair\s+value\s+through\s+profit\s+or\s+loss\b")),
        ("Deferred tax assets", re.compile(r"(?i)\bdeferred\s+tax\s+assets\b")),
        ("Other non-current assets", re.compile(r"(?i)\bother\s+non\s*-\s*current\s+assets\b")),
        ("Total non-current assets", re.compile(r"(?i)\btotal\s+non\s*-\s*current\s+assets\b")),
        ("Inventories", re.compile(r"(?i)\binventories\b")),
        ("Trade receivables", re.compile(r"(?i)\btrade\s+receivables\b")),
        ("Prepayments, other receivables and other assets", re.compile(r"(?i)\bprepayments,\s*other\s+receivables\s+and\s+other\s+assets\b")),
        ("Cash and bank balances", re.compile(r"(?i)\bcash\s+and\s+bank\s+balances\b")),
        ("Total current assets", re.compile(r"(?i)\btotal\s+current\s+assets\b")),
        ("Trade payables", re.compile(r"(?i)\btrade\s+payables\b")),
        ("Other payables and accruals", re.compile(r"(?i)\bother\s+payables\s+and\s+accruals\b")),
        ("Contract liabilities", re.compile(r"(?i)\bcontract\s+liabilities\b")),
        ("Interest-bearing bank and other borrowings", re.compile(r"(?i)\binterest\s*-\s*bearing\s+bank\s+and\s+other\s+borrowings\b")),
        ("Total current liabilities", re.compile(r"(?i)\btotal\s+current\s+liabilities\b")),
        ("Net current assets", re.compile(r"(?i)\bnet\s+current\s+assets\b")),
        ("Total assets less current liabilities", re.compile(r"(?i)\btotal\s+assets\s+less\s+current\s+liabilities\b")),
        ("Deferred tax liabilities", re.compile(r"(?i)\bdeferred\s+tax\s+liabilities\b")),
        ("Long-term payables", re.compile(r"(?i)\blong\s*-\s*term\s+payables\b")),
        ("Deferred income", re.compile(r"(?i)\bdeferred\s+income\b")),
        ("Other non-current liabilities", re.compile(r"(?i)\bother\s+non\s*-\s*current\s+liabilities\b")),
        ("Total non-current liabilities", re.compile(r"(?i)\btotal\s+non\s*-\s*current\s+liabilities\b")),
        ("Net assets", re.compile(r"(?i)\bnet\s+assets\b")),
        ("Share capital", re.compile(r"(?i)\bshare\s+capital\b")),
        ("Treasury shares", re.compile(r"(?i)\btreasury\s+shares\b")),
        ("Reserves", re.compile(r"(?i)\breserves\b")),
        ("Equity attributable to shareholders", re.compile(r"(?i)\bequity\s+attributable\s+to\s+shareholders\b")),
        ("Non-controlling interests", re.compile(r"(?i)\bnon\s*-\s*controlling\s+interests\b")),
        ("Total equity", re.compile(r"(?i)\btotal\s+equity\b")),
        ("Total liabilities", re.compile(r"(?i)\btotal\s+liabilities\b")),
        ("Total assets", re.compile(r"(?i)\btotal\s+assets\b")),
    ),
    "cash_flow": (
        ("Cash flows from operating activities", re.compile(r"(?i)\bcash\s+flows\s+from\s+operating\s+activities\b")),
        ("Profit or loss before tax", re.compile(r"(?i)\b(?:profit|loss)\s+before\s+tax\b")),
        ("Depreciation of property, plant and equipment", re.compile(r"(?i)\bdepreciation\s+of\s+property,\s*plant\s+and\s+equipment\b")),
        ("Depreciation of right-of-use assets", re.compile(r"(?i)\bdepreciation\s+of\s+right\s*-\s*of\s*-\s*use\s+assets\b")),
        ("Amortization of intangible assets", re.compile(r"(?i)\bamortization\s+of\s+intangible\s+assets\b")),
        ("Equity-settled share-based payments", re.compile(r"(?i)\bequity\s*-\s*settled\s+share\s*-\s*based\s+payments\b")),
        ("Finance costs", re.compile(r"(?i)\bfinance\s+costs\b")),
        ("Cash used in operations", re.compile(r"(?i)\bcash\s+used\s+in\s+operations\b")),
        ("Net cash flows used in operating activities", re.compile(r"(?i)\bnet\s+cash\s+flows?\s+used\s+in\s+operating\s+activities\b")),
        ("Cash flows from investing activities", re.compile(r"(?i)\bcash\s+flows\s+from\s+investing\s+activities\b")),
        ("Net cash flows from investing activities", re.compile(r"(?i)\bnet\s+cash\s+flows?\s+\(?used\s+in\)?/?from\s+investing\s+activities\b")),
        ("Cash flows from financing activities", re.compile(r"(?i)\bcash\s+flows\s+from\s+financing\s+activities\b")),
        ("Net cash flows from financing activities", re.compile(r"(?i)\bnet\s+cash\s+flows?\s+\(?used\s+in\)?/?from\s+financing\s+activities\b")),
    ),
}


def _parse_six_k_table_dataframe(table: Any) -> Optional[pd.DataFrame]:
    """从 6-K BS 表格对象安全提取 DataFrame。

    Args:
        table: 6-K 内部表格对象。

    Returns:
        DataFrame；不可用时返回 ``None``。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    table_tag = getattr(table, "tag", None)
    if table_tag is None:
        return None
    return parse_html_table_dataframe(table_tag)

def _build_six_k_markers(full_text: str) -> list[tuple[int, Optional[str]]]:
    """构建 6-K 新闻稿的章节边界标记。

    扫描预定义的语义关键词模式。叙述段取首次匹配，
    财务报表段取末次匹配（跳过内联引用）。
    结果经 ``_dedupe_markers`` 去重排序后返回。

    Args:
        full_text: 文档全文。

    Returns:
        ``(start_index, title)`` 标记列表。
    """

    report_markers = _build_six_k_report_markers(full_text)
    if len(report_markers) >= _SIX_K_REPORT_MODE_MIN_HEADINGS:
        return _dedupe_markers(report_markers)

    markers: list[tuple[int, Optional[str]]] = []
    for title, pattern, use_last in _SIX_K_MARKERS:
        match = _find_six_k_marker_match(
            full_text=full_text,
            title=title,
            pattern=pattern,
            use_last=use_last,
        )
        if match is None:
            continue
        markers.append((int(match.start()), title))
    return _dedupe_markers(markers)


def _build_six_k_report_markers(full_text: str) -> list[tuple[int, Optional[str]]]:
    """为目录型长篇 6-K 报告构建章节边界。

    这类文档常带 ``Table of Contents``，正文标题在目录和正文中都会重复。
    处理策略：
    1. 用 TOC cutoff 跳过目录区域；
    2. 在正文区按强标题匹配 report headings；
    3. 财务报表类标题仍沿用末次匹配，避免正文内联引用污染。

    Args:
        full_text: 文档全文。

    Returns:
        识别出的 ``(start_index, title)`` 列表；不足以判定 report mode 时返回空列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    toc_start = _find_table_of_contents_cutoff(full_text)
    if toc_start <= 0:
        return []

    markers: list[tuple[int, Optional[str]]] = []
    for title, pattern in _SIX_K_REPORT_HEADING_MARKERS:
        match = _find_first_pattern_after(pattern=pattern, full_text=full_text, start_at=toc_start)
        if match is None:
            continue
        markers.append((int(match.start()), title))

    if len(markers) < _SIX_K_REPORT_MODE_MIN_HEADINGS:
        return []

    for title, pattern, use_last in _SIX_K_MARKERS:
        if not use_last:
            continue
        match = _find_six_k_marker_match(
            full_text=full_text,
            title=title,
            pattern=pattern,
            use_last=True,
            start_at=toc_start,
        )
        if match is None:
            continue
        markers.append((int(match.start()), title))
    return markers


def _find_six_k_marker_match(
    *,
    full_text: str,
    title: str,
    pattern: re.Pattern[str],
    use_last: bool,
    start_at: int = 0,
) -> Optional[re.Match[str]]:
    """按 6-K 规则选取单个 marker 的最佳命中。

    Args:
        full_text: 文档全文。
        title: 章节标题。
        pattern: 标题匹配正则。
        use_last: 是否取末次匹配。
        start_at: 最小搜索起点。

    Returns:
        选中的匹配对象；未命中返回 ``None``。

    Raises:
        RuntimeError: 选取失败时抛出。
    """

    valid_matches: list[re.Match[str]] = []
    for match in pattern.finditer(full_text):
        if int(match.start()) < max(0, int(start_at)):
            continue
        if title == "About Company" and not _looks_like_six_k_about_company_context(
            full_text=full_text,
            position=int(match.start()),
        ):
            continue
        valid_matches.append(match)

    if not valid_matches:
        return None
    if use_last:
        return valid_matches[-1]
    return valid_matches[0]


def _find_first_pattern_after(
    *,
    pattern: re.Pattern[str],
    full_text: str,
    start_at: int,
) -> Optional[re.Match[str]]:
    """查找指定位置之后的首个正则命中。

    Args:
        pattern: 目标正则。
        full_text: 文档全文。
        start_at: 搜索起点。

    Returns:
        首个满足起点约束的匹配对象；未命中返回 ``None``。

    Raises:
        RuntimeError: 搜索失败时抛出。
    """

    for match in pattern.finditer(full_text):
        if int(match.start()) >= max(0, int(start_at)):
            return match
    return None


def _looks_like_six_k_about_company_context(
    *,
    full_text: str,
    position: int,
) -> bool:
    """判断 About Company 命中是否位于章节标题语境。

    目标是保留句首/行首的 ``About Company`` 简介段标题，
    同时排除正文中 ``more information about X`` 这类普通叙述。

    Args:
        full_text: 文档全文。
        position: 命中起点。

    Returns:
        看起来像标题时返回 ``True``，正文叙述返回 ``False``。

    Raises:
        RuntimeError: 判断失败时抛出。
    """

    if position <= 0:
        return True

    line_start = full_text.rfind("\n", 0, position) + 1
    prefix = full_text[line_start:position]
    if not prefix.strip():
        return True

    stripped_prefix = prefix.rstrip()
    if not stripped_prefix:
        return True
    return stripped_prefix[-1] in {".", ":", ";", "-", "–", "—", ")"}


def _classify_statement_type_for_table(
    *,
    caption: Optional[str],
    headers: Optional[list[str]],
    context_before: str,
) -> Optional[str]:
    """判定 6-K 表格属于哪类财务报表。

    Args:
        caption: 表格标题。
        headers: 表头列表。
        context_before: 表格前文。

    Returns:
        标准化报表类型；无法判定时返回 `None`。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    caption_text = _normalize_statement_text(str(caption or ""))
    headers_text = _normalize_statement_text(" ".join(str(item or "") for item in (headers or [])))
    context_text = _normalize_statement_text(
        str(context_before or "")[-_STATEMENT_CLASSIFICATION_CONTEXT_WINDOW:]
    )
    if not caption_text and not headers_text and not context_text:
        return None

    if _looks_like_non_statement_table(caption_text=caption_text, headers_text=headers_text):
        return None

    scores = {
        statement_type: _score_statement_type(
            statement_type=statement_type,
            statement_patterns=patterns,
            caption_text=caption_text,
            headers_text=headers_text,
            context_text=context_text,
        )
        for statement_type, patterns in _STATEMENT_PATTERNS.items()
    }
    if not scores:
        return None

    winner, winner_score = max(scores.items(), key=lambda item: item[1])
    if winner_score < _STATEMENT_CLASSIFICATION_MIN_SCORE:
        if winner_score < _STATEMENT_CLASSIFICATION_CONTEXT_ONLY_MIN_SCORE:
            return None
        if _has_caption_or_header_signal(
            statement_patterns=_STATEMENT_PATTERNS[winner],
            caption_text=caption_text,
            headers_text=headers_text,
        ):
            return winner
        return None

    runner_up_score = max(
        (score for statement_type, score in scores.items() if statement_type != winner),
        default=0,
    )
    if winner_score == runner_up_score:
        return None
    return winner


def _normalize_statement_text(value: str) -> str:
    """标准化报表分类文本。

    Args:
        value: 原始文本。

    Returns:
        统一大小写和空白后的文本。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    return _normalize_whitespace(value).lower()


def _looks_like_non_statement_table(
    *,
    caption_text: str,
    headers_text: str,
) -> bool:
    """判断表格是否属于目录/签名等非财报表格。

    Args:
        caption_text: 规范化 caption 文本。
        headers_text: 规范化 headers 文本。

    Returns:
        若命中噪声表模式且缺乏报表信号则返回 ``True``。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    caption_or_header = f"{caption_text} {headers_text}".strip()
    if not caption_or_header:
        return False

    # caption 命中目录/签名类模式时，优先视为噪声表（除非 caption 自身含报表标题）。
    caption_has_noise = any(pattern.search(caption_text) for pattern in _NON_STATEMENT_TABLE_PATTERNS)
    if caption_has_noise:
        caption_has_statement_signal = any(
            _has_caption_or_header_signal(
                statement_patterns=patterns,
                caption_text=caption_text,
                headers_text="",
            )
            for patterns in _STATEMENT_PATTERNS.values()
        )
        if not caption_has_statement_signal:
            return True

    has_noise_pattern = any(pattern.search(caption_or_header) for pattern in _NON_STATEMENT_TABLE_PATTERNS)
    if not has_noise_pattern:
        return False
    has_statement_signal = any(
        _has_caption_or_header_signal(
            statement_patterns=patterns,
            caption_text=caption_text,
            headers_text=headers_text,
        )
        for patterns in _STATEMENT_PATTERNS.values()
    )
    return not has_statement_signal


def _score_statement_type(
    *,
    statement_type: str,
    statement_patterns: tuple[re.Pattern[str], ...],
    caption_text: str,
    headers_text: str,
    context_text: str,
) -> int:
    """计算某报表类型的匹配分数。

    Args:
        statement_patterns: 该报表类型的关键词正则。
        caption_text: 规范化 caption 文本。
        headers_text: 规范化 headers 文本。
        context_text: 规范化上下文文本。

    Returns:
        匹配分数（整数，越高越可信）。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    caption_hits = _count_pattern_hits(statement_patterns=statement_patterns, text=caption_text)
    header_hits = _count_pattern_hits(statement_patterns=statement_patterns, text=headers_text)
    context_hits = _count_pattern_hits(statement_patterns=statement_patterns, text=context_text)
    line_item_hits = _count_pattern_hits(
        statement_patterns=_STATEMENT_LINE_ITEM_PATTERNS.get(statement_type, ()),
        text=headers_text,
    )
    return (
        caption_hits * _STATEMENT_CLASSIFICATION_WEIGHTS["caption"]
        + header_hits * _STATEMENT_CLASSIFICATION_WEIGHTS["headers"]
        + context_hits * _STATEMENT_CLASSIFICATION_WEIGHTS["context"]
        + line_item_hits * _STATEMENT_LINE_ITEM_WEIGHT
    )


def _count_pattern_hits(
    *,
    statement_patterns: tuple[re.Pattern[str], ...],
    text: str,
) -> int:
    """统计文本命中的模式数。

    Args:
        statement_patterns: 目标模式集合。
        text: 待匹配文本。

    Returns:
        命中的模式数量。

    Raises:
        RuntimeError: 统计失败时抛出。
    """

    if not text:
        return 0
    return sum(1 for pattern in statement_patterns if pattern.search(text))


def _has_caption_or_header_signal(
    *,
    statement_patterns: tuple[re.Pattern[str], ...],
    caption_text: str,
    headers_text: str,
) -> bool:
    """判断 caption/headers 是否包含报表类型信号。

    Args:
        statement_patterns: 目标模式集合。
        caption_text: 规范化 caption 文本。
        headers_text: 规范化 headers 文本。

    Returns:
        任一模式命中 caption 或 headers 时返回 ``True``。

    Raises:
        RuntimeError: 判定失败时抛出。
    """

    return any(
        pattern.search(caption_text) or pattern.search(headers_text)
        for pattern in statement_patterns
    )


def _build_statement_result_from_tables(
    *,
    statement_type: str,
    tables: list[Any],
) -> Optional[FinancialStatementResult]:
    """从多个 6-K 表格构建结构化财务报表结果。

    Args:
        statement_type: 报表类型。
        tables: 候选表格列表。

    Returns:
        结构化财务报表结果；低置信时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """
    return _build_shared_statement_result_from_tables(
        statement_type=statement_type,
        tables=tables,
        parse_table_dataframe=_parse_six_k_table_dataframe,
    )


def extract_statement_result_from_ocr_pages(
    *,
    statement_type: str,
    page_texts: list[str],
) -> Optional[FinancialStatementResult]:
    """从 image+OCR 隐藏文本页回退构建 6-K 财务报表。

    Args:
        statement_type: 目标报表类型。
        page_texts: 候选 OCR 页文本列表。

    Returns:
        结构化财务报表结果；无法稳定解析时返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if statement_type not in _OCR_STATEMENT_TITLE_PATTERNS:
        return None

    parsed_pages: list[dict[str, Any]] = []
    for page_text in page_texts:
        parsed_page = _parse_statement_from_ocr_page(
            statement_type=statement_type,
            page_text=page_text,
        )
        if parsed_page is not None:
            parsed_pages.append(parsed_page)
    if not parsed_pages:
        if statement_type != "income":
            return None
        return _extract_income_summary_result_from_ocr_pages(page_texts)

    grouped_pages: dict[tuple[str, ...], dict[str, Any]] = {}
    for parsed_page in parsed_pages:
        periods = list(parsed_page["periods"])
        period_key = tuple(str(period["period_end"]) for period in periods)
        payload = grouped_pages.setdefault(
            period_key,
            {
                "periods": periods,
                "rows": [],
                "currency": parsed_page.get("currency"),
                "units": parsed_page.get("units"),
                "scale": parsed_page.get("scale"),
            },
        )
        payload["rows"].extend(parsed_page["rows"])
        if payload.get("currency") is None:
            payload["currency"] = parsed_page.get("currency")
        if payload.get("units") is None:
            payload["units"] = parsed_page.get("units")
        if payload.get("scale") is None:
            payload["scale"] = parsed_page.get("scale")

    _, selected_payload = max(
        grouped_pages.items(),
        key=lambda item: (len(item[1]["rows"]), len(item[1]["periods"])),
    )
    rows = _dedupe_ocr_statement_rows(list(selected_payload["rows"]))
    periods = list(selected_payload["periods"])
    if not rows or len(periods) < 2:
        if statement_type != "income":
            return None
        return _extract_income_summary_result_from_ocr_pages(page_texts)

    return {
        "statement_type": statement_type,
        "periods": periods,
        "rows": rows,
        "currency": selected_payload.get("currency"),
        "units": selected_payload.get("units"),
        "scale": selected_payload.get("scale"),
        "data_quality": "extracted",
        "statement_locator": build_statement_locator(
            statement_type=statement_type,
            periods=periods,
            rows=rows,
        ),
    }


def _extract_income_summary_result_from_ocr_pages(
    page_texts: list[str],
) -> Optional[FinancialStatementResult]:
    """从 OCR 页中的 `Profit & Loss` 摘要块回退提取 income 报表。

    当前函数只处理一类高置信版式：页面中存在明确的 `Profit & Loss`
    标题，随后跟随 line item + 当前期间金额 + delta 指标的摘要表。
    这类页面常见于银行/券商 results presentation，虽然不具备完整
    `statement title + 多期间表头` 结构，但仍包含可稳定抽取的当期
    income 核心行项目。

    Args:
        page_texts: OCR 候选页文本列表。

    Returns:
        结构化 income 报表结果；未命中高置信摘要页时返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    best_result: Optional[FinancialStatementResult] = None
    best_score: tuple[int, float] = (0, 0.0)
    for page_text in page_texts:
        parsed_result = _parse_income_summary_from_ocr_page(page_text)
        if parsed_result is None:
            continue
        candidate_score = (
            len(parsed_result["rows"]),
            sum(
                abs(float(value))
                for row in parsed_result["rows"]
                for value in row["values"]
                if isinstance(value, (int, float))
            ),
        )
        if candidate_score > best_score:
            best_result = parsed_result
            best_score = candidate_score
    return best_result


def _parse_income_summary_from_ocr_page(
    page_text: str,
) -> Optional[FinancialStatementResult]:
    """解析单页 OCR 文本中的 `Profit & Loss` 摘要表。

    Args:
        page_text: 原始 OCR 页文本。

    Returns:
        命中时返回单期间 income 报表；否则返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    normalized_text = _normalize_ocr_page_text(page_text)
    for title_pattern in _OCR_INCOME_SUMMARY_TITLE_PATTERNS:
        for title_match in title_pattern.finditer(normalized_text):
            parsed_result = _build_income_summary_result_from_title_match(
                normalized_text=normalized_text,
                title_match=title_match,
            )
            if parsed_result is not None:
                return parsed_result
    return None


def _build_income_summary_result_from_title_match(
    *,
    normalized_text: str,
    title_match: re.Match[str],
) -> Optional[FinancialStatementResult]:
    """根据 `Profit & Loss` 标题命中构建单期间 income 报表。

    Args:
        normalized_text: 规范化后的 OCR 页文本。
        title_match: `Profit & Loss` 标题匹配对象。

    Returns:
        可稳定解析时返回单期间 income 报表；否则返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    window_start = int(title_match.start())
    page_body = normalized_text[window_start:window_start + _OCR_SUMMARY_BODY_WINDOW_CHARS]
    period_summary = _extract_income_summary_period(
        normalized_text=normalized_text,
        title_start=window_start,
        page_body=page_body,
    )
    if period_summary is None:
        return None

    rows = _extract_income_summary_rows(page_body)
    if len(rows) < 2:
        return None

    return {
        "statement_type": "income",
        "periods": [period_summary],
        "rows": rows,
        "currency": _extract_income_summary_currency(page_body),
        "units": None,
        "scale": None,
        "data_quality": "extracted",
        "statement_locator": build_statement_locator(
            statement_type="income",
            periods=[period_summary],
            rows=rows,
        ),
    }


def _extract_income_summary_period(
    *,
    normalized_text: str,
    title_start: int,
    page_body: str,
) -> Optional[dict[str, Any]]:
    """提取 `Profit & Loss` 摘要页当前期间。

    Args:
        normalized_text: 规范化后的完整 OCR 页文本。
        title_start: 标题命中的起始偏移。
        page_body: 从标题开始截取的正文窗口。

    Returns:
        当前期间摘要；无法识别时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    explicit_period = _extract_income_summary_period_from_title_window(
        normalized_text=normalized_text,
        title_start=title_start,
    )
    if explicit_period is not None:
        return explicit_period

    anchor_month_day = _extract_ocr_anchor_month_day(page_body)
    period_entries, _ = _extract_ocr_period_entries_and_header_end(header_text=page_body[:800])
    if not period_entries:
        return None
    period_summaries = _build_ocr_period_summaries(
        period_entries=period_entries,
        anchor_month_day=anchor_month_day,
        page_text=page_body,
    )
    if not period_summaries:
        return None
    return max(period_summaries, key=lambda item: str(item["period_end"]))


def _extract_income_summary_period_from_title_window(
    *,
    normalized_text: str,
    title_start: int,
) -> Optional[dict[str, Any]]:
    """从标题附近窗口提取单期间口径。

    优先读取 `1Q25 Earnings`、`2Q25`、`H1 2025` 等贴近标题的 token，
    避免被正文中的比较期间、页码或脚注年份干扰。

    Args:
        normalized_text: 规范化后的完整 OCR 页文本。
        title_start: 标题命中的起始偏移。

    Returns:
        当前期间摘要；窗口内未命中时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    window = normalized_text[
        max(0, title_start - _OCR_SUMMARY_TITLE_LOOKBACK_CHARS):
        title_start + _OCR_SUMMARY_TITLE_LOOKAHEAD_CHARS
    ]
    period_match = re.search(
        r"(?i)\b(?:(?P<quarter>[1-4])Q|Q(?P<quarter_rev>[1-4])|(?P<half>[12])H|H(?P<half_rev>[12])|FY)"
        r"\s*[-/']?\s*(?P<year>\d{2,4})\b",
        window,
    )
    if period_match is None:
        return None

    fiscal_year = _extract_fiscal_year_from_ocr_period_match(period_match)
    if fiscal_year is None:
        return None
    fiscal_period = _extract_fiscal_period_from_ocr_period_match(period_match)
    period_end = _resolve_ocr_period_end(
        fiscal_year=fiscal_year,
        fiscal_period=fiscal_period,
        anchor_month_day=(12, 31),
    )
    return {
        "period_end": period_end,
        "fiscal_year": fiscal_year,
        "fiscal_period": fiscal_period,
    }


def _extract_income_summary_rows(page_body: str) -> list[dict[str, Any]]:
    """从 `Profit & Loss` 摘要块中提取行项目和值。

    Args:
        page_body: 从标题开始截取的 OCR 文本窗口。

    Returns:
        单期间 income 行项目列表。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    rows: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for label, pattern in _OCR_LINE_ITEM_PATTERNS_BY_STATEMENT["income"]:
        for match in pattern.finditer(page_body):
            if label in seen_labels:
                break
            value = _extract_income_summary_value_after_label(
                page_body=page_body,
                label_end=int(match.end()),
            )
            if value is None:
                continue
            rows.append(
                {
                    "concept": "",
                    "label": label,
                    "values": [value],
                }
            )
            seen_labels.add(label)
            break
    return rows


def _extract_income_summary_value_after_label(
    *,
    page_body: str,
    label_end: int,
) -> Optional[float]:
    """提取摘要页某个 income label 后的当前期间金额。

    Args:
        page_body: 从标题开始截取的 OCR 文本窗口。
        label_end: label 结束偏移。

    Returns:
        当前期间金额；无法稳定识别时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    lookahead = page_body[label_end:label_end + _OCR_SUMMARY_LABEL_LOOKAHEAD_CHARS]
    for match in _OCR_VALUE_TOKEN_RE.finditer(lookahead):
        raw_token = str(match.group(0) or "")
        # 百分比、年份、过小的 delta 数值不视为当前期间金额。
        if _OCR_PERCENT_TOKEN_RE.search(lookahead[:int(match.end())]) is not None:
            continue
        value = _parse_ocr_numeric_token(raw_token)
        if value is None:
            continue
        if 1900 < abs(value) < 2100:
            continue
        if "," not in raw_token and abs(value) < _OCR_SUMMARY_MIN_ABS_VALUE:
            continue
        return value
    return None


def _extract_income_summary_currency(page_body: str) -> Optional[str]:
    """提取摘要页货币代码。

    Args:
        page_body: 从标题开始截取的 OCR 文本窗口。

    Returns:
        标准货币代码；缺失时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    currency_raw, _ = _extract_ocr_currency_and_scale(page_body[:800])
    return _map_ocr_currency_code(currency_raw)


def _parse_statement_from_ocr_page(
    *,
    statement_type: str,
    page_text: str,
) -> Optional[dict[str, Any]]:
    """解析单页 OCR 文本中的财务报表。

    Args:
        statement_type: 目标报表类型。
        page_text: 原始 OCR 页文本。

    Returns:
        单页解析结果；无法稳定解析时返回 `None`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    normalized_text = _normalize_ocr_page_text(page_text)
    title_match = _find_ocr_statement_title_match(
        statement_type=statement_type,
        page_text=normalized_text,
    )
    if title_match is None:
        return None

    page_body = normalized_text[int(title_match.start()):]
    anchor_month_day = _extract_ocr_anchor_month_day(page_body)
    period_entries, header_end = _extract_ocr_period_entries_and_header_end(
        header_text=page_body,
    )
    if len(period_entries) < 2:
        return None

    period_count = len(period_entries)
    values = _extract_ocr_numeric_values(page_body[header_end:])
    if len(values) < period_count * 2:
        return None
    row_count = len(values) // period_count
    if row_count < 2:
        return None

    row_values = values[: row_count * period_count]
    labels = _extract_ocr_line_item_labels(
        statement_type=statement_type,
        label_source=page_body[:header_end],
        row_count=row_count,
    )
    if len(labels) < row_count:
        labels.extend(
            f"line_item_{index:02d}"
            for index in range(len(labels) + 1, row_count + 1)
        )

    periods = _build_ocr_period_summaries(
        period_entries=period_entries,
        anchor_month_day=anchor_month_day,
        page_text=page_body,
    )
    rows = [
        {
            "concept": "",
            "label": labels[index],
            "values": row_values[index * period_count:(index + 1) * period_count],
        }
        for index in range(row_count)
    ]
    currency_raw, scale = _extract_ocr_currency_and_scale(page_body[:header_end])
    return {
        "periods": periods,
        "rows": rows,
        "currency": _map_ocr_currency_code(currency_raw),
        "units": _build_ocr_units_label(currency_raw=currency_raw, scale=scale),
        "scale": scale,
    }


def _normalize_ocr_page_text(page_text: str) -> str:
    """标准化 OCR 页文本。

    Args:
        page_text: 原始 OCR 文本。

    Returns:
        标准化后的页文本。

    Raises:
        RuntimeError: 标准化失败时抛出。
    """

    return _normalize_whitespace(html.unescape(str(page_text or "")))


def _find_ocr_statement_title_match(
    *,
    statement_type: str,
    page_text: str,
) -> Optional[re.Match[str]]:
    """查找 OCR 页中的报表标题命中。

    Args:
        statement_type: 目标报表类型。
        page_text: 标准化页文本。

    Returns:
        标题匹配对象；未命中返回 `None`。

    Raises:
        RuntimeError: 搜索失败时抛出。
    """

    for pattern in _OCR_STATEMENT_TITLE_PATTERNS.get(statement_type, ()):
        match = pattern.search(page_text)
        if match is not None:
            return match
    return None


def _extract_ocr_anchor_month_day(page_text: str) -> tuple[int, int]:
    """提取 OCR 页中报表共同使用的月日锚点。

    Args:
        page_text: 标准化页文本。

    Returns:
        `(month, day)`；无法识别时默认 `(12, 31)`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    match = _OCR_DATE_RE.search(page_text)
    if match is None:
        return (12, 31)
    month_token = str(match.group(1) or "").strip().lower()
    day_value = int(match.group(2) or 31)
    return (_OCR_MONTH_TO_NUMBER.get(month_token, 12), min(day_value, 31))


_OCR_PERIOD_TOKEN_RE = re.compile(
    r"(?i)\b(?:(?P<quarter>[1-4])Q|Q(?P<quarter_rev>[1-4])|(?P<half>[12])H|H(?P<half_rev>[12])|"
    r"(?P<nine>9)M|(?P<twelve>12)M|FY)\s*[-/']?\s*(?P<year>\d{2,4})\b|"
    r"\b(?P<year_first>\d{2,4})\s*[-/']?\s*(?:(?P<year_first_quarter>[1-4])Q|Q(?P<year_first_quarter_rev>[1-4])|"
    r"(?P<year_first_half>[12])H|H(?P<year_first_half_rev>[12])|(?P<year_first_nine>9)M|"
    r"(?P<year_first_twelve>12)M|FY)\b"
)


def _extract_ocr_period_entries_and_header_end(
    *,
    header_text: str,
) -> tuple[list[tuple[int, Optional[str]]], int]:
    """从 OCR 页头部提取期间列与数值区起点。

    Args:
        header_text: 第一组数值出现前的页头文本。

    Returns:
        `(period_entries, header_end)`。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    token_matches = list(_OCR_PERIOD_TOKEN_RE.finditer(header_text))
    grouped_token_matches = _group_close_ocr_matches(token_matches)
    token_group = next(
        (group for group in reversed(grouped_token_matches) if len(group) >= 2),
        [],
    )
    if len(token_group) >= 2:
        period_entries: list[tuple[int, Optional[str]]] = []
        for match in token_group:
            fiscal_period = _extract_fiscal_period_from_ocr_period_match(match)
            fiscal_year = _extract_fiscal_year_from_ocr_period_match(match)
            if fiscal_year is None:
                continue
            period_entry = (fiscal_year, fiscal_period)
            if period_entry not in period_entries:
                period_entries.append(period_entry)
        if len(period_entries) >= 2:
            header_end = _extend_ocr_header_end(
                header_text=header_text,
                match_end=int(token_group[-1].end()),
            )
            return period_entries, header_end

    year_matches = list(_OCR_YEAR_RE.finditer(header_text))
    if len(year_matches) < 2:
        return [], len(header_text)

    grouped_matches = _group_close_ocr_matches(year_matches)
    candidate_group = next(
        (group for group in reversed(grouped_matches) if len(group) >= 2),
        [],
    )
    if len(candidate_group) < 2:
        return [], len(header_text)

    period_entries: list[tuple[int, Optional[str]]] = []
    for match in candidate_group:
        year_value = int(match.group(1))
        period_entry = (year_value, None)
        if period_entry not in period_entries:
            period_entries.append(period_entry)
    if len(period_entries) < 2:
        return [], len(header_text)

    header_end = _extend_ocr_header_end(
        header_text=header_text,
        match_end=int(candidate_group[-1].end()),
    )
    return period_entries, header_end


def _group_close_ocr_matches(
    matches: list[re.Match[str]],
) -> list[list[re.Match[str]]]:
    """按距离把 OCR 页头中的 period match 分组。

    Args:
        matches: 顺序匹配结果。

    Returns:
        分组后的匹配列表。

    Raises:
        RuntimeError: 分组失败时抛出。
    """

    grouped_matches: list[list[re.Match[str]]] = []
    current_group: list[re.Match[str]] = []
    for match in matches:
        if not current_group:
            current_group = [match]
            continue
        previous_match = current_group[-1]
        if int(match.start()) - int(previous_match.end()) <= 16:
            current_group.append(match)
            continue
        grouped_matches.append(current_group)
        current_group = [match]
    if current_group:
        grouped_matches.append(current_group)
    return grouped_matches


def _extend_ocr_header_end(
    *,
    header_text: str,
    match_end: int,
) -> int:
    """把 OCR 页头结束位置延伸到紧邻的 currency/scale 文本之后。

    Args:
        header_text: 页头文本。
        match_end: 当前期间列匹配的结束位置。

    Returns:
        延伸后的页头结束位置。

    Raises:
        RuntimeError: 计算失败时抛出。
    """

    header_end = match_end
    currency_match = _OCR_CURRENCY_SCALE_RE.search(header_text, header_end)
    if currency_match is not None:
        header_end = int(currency_match.end())
        next_currency_match = _OCR_CURRENCY_SCALE_RE.search(header_text, header_end)
        if next_currency_match is not None and int(next_currency_match.start()) - header_end <= 24:
            header_end = int(next_currency_match.end())
    return header_end


def _extract_fiscal_period_from_ocr_period_match(
    match: re.Match[str],
) -> Optional[str]:
    """从 OCR 期间 token match 提取 fiscal period。

    Args:
        match: OCR 期间 token 匹配对象。

    Returns:
        fiscal period；无法识别时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    quarter_raw = match.group("quarter") or match.group("quarter_rev")
    if quarter_raw is not None:
        return f"Q{quarter_raw}"
    quarter_year_first = match.group("year_first_quarter") or match.group("year_first_quarter_rev")
    if quarter_year_first is not None:
        return f"Q{quarter_year_first}"
    half_raw = match.group("half") or match.group("half_rev")
    if half_raw is not None:
        return f"H{half_raw}"
    half_year_first = match.group("year_first_half") or match.group("year_first_half_rev")
    if half_year_first is not None:
        return f"H{half_year_first}"
    if match.group("nine") is not None or match.group("year_first_nine") is not None:
        return "Q3"
    if match.group("twelve") is not None or match.group("year_first_twelve") is not None:
        return "FY"
    raw_text = str(match.group(0) or "").upper()
    if "FY" in raw_text:
        return "FY"
    return None


def _extract_fiscal_year_from_ocr_period_match(
    match: re.Match[str],
) -> Optional[int]:
    """从 OCR 期间 token match 提取 fiscal year。

    Args:
        match: OCR 期间 token 匹配对象。

    Returns:
        fiscal year；无法识别时返回 `None`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    year_raw = match.group("year") or match.group("year_first")
    if year_raw is None:
        return None
    return _normalize_year_token(year_raw)


def _extract_ocr_numeric_values(value_text: str) -> list[Optional[float]]:
    """提取 OCR 页中的数值序列。

    Args:
        value_text: 数值区文本。

    Returns:
        数值列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    values: list[Optional[float]] = []
    for match in _OCR_VALUE_TOKEN_RE.finditer(value_text):
        values.append(_parse_ocr_numeric_token(match.group(0)))
    return values


def _parse_ocr_numeric_token(token: str) -> Optional[float]:
    """把 OCR 数值 token 转成浮点数。

    Args:
        token: 原始数值 token。

    Returns:
        浮点数；横线占位、空 token 或无法解析为浮点时返回 `None`。

    Raises:
        无。
    """

    normalized_token = str(token or "").strip()
    if normalized_token in {"-", "–", "—"}:
        return None
    is_negative = normalized_token.startswith("(") and normalized_token.endswith(")")
    cleaned = normalized_token.strip("()").replace(",", "")
    # OCR 文本中括号负数可能含货币符号（如 "($1,234)"），剥离括号后货币符号仍残留，
    # 必须再次清理，否则 ``float`` 会抛 ``ValueError`` 中断整条 OCR 解析链路。
    cleaned = re.sub(r"[$€£¥₩₹]", "", cleaned).strip()
    if not cleaned:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return -value if is_negative else value


def _extract_ocr_line_item_labels(
    *,
    statement_type: str,
    label_source: str,
    row_count: int,
) -> list[str]:
    """从 OCR 页头文本提取行标签。

    Args:
        statement_type: 目标报表类型。
        label_source: 页头标签文本。
        row_count: 期望行数。

    Returns:
        行标签列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    normalized_source = _normalize_whitespace(label_source)
    normalized_source = re.sub(_OCR_DATE_RE, " ", normalized_source)
    normalized_source = re.sub(r"(?i)\bnotes?\b", " ", normalized_source)
    normalized_source = re.sub(r"\b\d{1,2}\b", " ", normalized_source)
    matches: list[tuple[int, int, str]] = []
    for label, pattern in _OCR_LINE_ITEM_PATTERNS_BY_STATEMENT.get(statement_type, ()):
        for match in pattern.finditer(normalized_source):
            matches.append((int(match.start()), int(match.end()), label))
    matches.sort(key=lambda item: (item[0], -(item[1] - item[0]), item[2]))

    labels: list[str] = []
    consumed_until = -1
    for start, end, label in matches:
        if start < consumed_until:
            continue
        labels.append(label)
        consumed_until = end
        if len(labels) >= row_count:
            break
    return labels


def _build_ocr_period_summaries(
    *,
    period_entries: list[tuple[int, Optional[str]]],
    anchor_month_day: tuple[int, int],
    page_text: str,
) -> list[dict[str, Any]]:
    """构建 OCR 财务报表的期间摘要。

    Args:
        period_entries: `(fiscal_year, fiscal_period)` 列表。
        anchor_month_day: 月日锚点。
        page_text: 页文本，用于推断 fiscal period。

    Returns:
        期间摘要列表。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    month_value, day_value = anchor_month_day
    inferred_fiscal_period = _infer_ocr_fiscal_period(
        page_text=page_text,
        month_value=month_value,
    )
    periods: list[dict[str, Any]] = []
    for fiscal_year, explicit_fiscal_period in period_entries:
        fiscal_period = explicit_fiscal_period or inferred_fiscal_period
        period_end = _resolve_ocr_period_end(
            fiscal_year=fiscal_year,
            fiscal_period=fiscal_period,
            anchor_month_day=anchor_month_day,
        )
        periods.append(
            {
                "period_end": period_end,
                "fiscal_year": fiscal_year,
                "fiscal_period": fiscal_period,
            }
        )
    return periods


def _resolve_ocr_period_end(
    *,
    fiscal_year: int,
    fiscal_period: Optional[str],
    anchor_month_day: tuple[int, int],
) -> str:
    """解析 OCR 期间列的 period_end 文本。

    Args:
        fiscal_year: 财年。
        fiscal_period: 明示或推断出的财期。
        anchor_month_day: 页头月日锚点。

    Returns:
        `YYYY-MM-DD` 形式的 period_end。

    Raises:
        RuntimeError: 解析失败时抛出。
    """

    if fiscal_period is not None:
        resolved = _resolve_period_end_from_fiscal_period(
            fiscal_period=fiscal_period,
            fiscal_year=fiscal_year,
        )
        if resolved is not None:
            return resolved.isoformat()
    month_value, day_value = anchor_month_day
    return f"{fiscal_year:04d}-{month_value:02d}-{day_value:02d}"


def _infer_ocr_fiscal_period(
    *,
    page_text: str,
    month_value: int,
) -> Optional[str]:
    """推断 OCR 财报期间口径。

    Args:
        page_text: 页文本。
        month_value: 期间月份。

    Returns:
        fiscal period 标识；无法判断时返回 `None`。

    Raises:
        RuntimeError: 推断失败时抛出。
    """

    normalized_text = page_text.lower()
    if "three months ended" in normalized_text or "quarter ended" in normalized_text:
        return _month_to_quarter(month_value)
    if "six months ended" in normalized_text or "half-year" in normalized_text or "half year" in normalized_text:
        return "H1" if month_value <= 6 else "H2"
    if "nine months ended" in normalized_text:
        return "9M"
    if "year ended" in normalized_text:
        return "FY"
    if "as at" in normalized_text and month_value in {3, 6, 9, 12}:
        return _month_to_quarter(month_value)
    return "FY" if month_value == 12 else None


def _month_to_quarter(month_value: int) -> str:
    """将月份映射为季度标识。

    Args:
        month_value: 月份。

    Returns:
        季度标识。

    Raises:
        RuntimeError: 映射失败时抛出。
    """

    quarter = max(1, min(4, (month_value - 1) // 3 + 1))
    return f"Q{quarter}"


def _extract_ocr_currency_and_scale(header_text: str) -> tuple[Optional[str], Optional[str]]:
    """从 OCR 页头提取货币与缩放口径。

    Args:
        header_text: 报表页头文本。

    Returns:
        `(currency_raw, scale)`。

    Raises:
        RuntimeError: 提取失败时抛出。
    """

    match = _OCR_CURRENCY_SCALE_RE.search(header_text)
    if match is None:
        return (None, None)
    currency_raw = str(match.group(1) or match.group(2) or "").strip() or None
    matched_text = str(match.group(0) or "").lower()
    if "million" in matched_text:
        scale = "millions"
    elif "billion" in matched_text:
        scale = "billions"
    elif "000" in matched_text:
        scale = "thousands"
    else:
        scale = None
    return (currency_raw, scale)


def _build_ocr_units_label(
    *,
    currency_raw: Optional[str],
    scale: Optional[str],
) -> Optional[str]:
    """构建 OCR 财报 units 文本。

    Args:
        currency_raw: 原始货币文本。
        scale: 缩放口径。

    Returns:
        units 文本；不存在时返回 `None`。

    Raises:
        RuntimeError: 构建失败时抛出。
    """

    if currency_raw is None and scale is None:
        return None
    if currency_raw is None:
        return scale
    if scale is None:
        return currency_raw
    return f"{currency_raw} in {scale}"


def _map_ocr_currency_code(currency_raw: Optional[str]) -> Optional[str]:
    """把 OCR 原始货币映射为标准代码。

    Args:
        currency_raw: 原始货币文本。

    Returns:
        标准货币代码；未知时回退原文本。

    Raises:
        RuntimeError: 映射失败时抛出。
    """

    if currency_raw is None:
        return None
    return _OCR_CURRENCY_CODE_MAP.get(currency_raw, currency_raw)


def _dedupe_ocr_statement_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按标签和值去重 OCR 财务行。

    Args:
        rows: 原始行列表。

    Returns:
        去重后的行列表。

    Raises:
        RuntimeError: 去重失败时抛出。
    """

    deduped_rows: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, tuple[Optional[float], ...]]] = set()
    for row in rows:
        label = str(row.get("label") or "")
        values = tuple(row.get("values") or [])
        dedupe_key = (label, values)
        if dedupe_key in seen_keys:
            continue
        seen_keys.add(dedupe_key)
        deduped_rows.append(row)
    return deduped_rows


def select_statement_tables_by_row_signals(
    *,
    statement_type: str,
    tables: list[Any],
) -> list[Any]:
    """按行标签语义信号回退筛选候选报表表格。

    当 caption/header/context 无法稳定归类时，直接解析表格并基于
    行标签关键词（如 `total assets` / `revenue` / `net income`）
    识别报表类型。

    Args:
        statement_type: 报表类型。
        tables: 原始候选表格列表。

    Returns:
        按置信度排序的候选表格列表。

    Raises:
        RuntimeError: 解析失败时抛出。
    """
    return _select_shared_statement_tables_by_row_signals(
        tables=tables,
        line_item_patterns=_STATEMENT_LINE_ITEM_PATTERNS.get(statement_type, ()),
        min_hits=_ROW_SIGNAL_MIN_HITS_BY_STATEMENT.get(statement_type, 1),
        parse_table_dataframe=_parse_six_k_table_dataframe,
    )
