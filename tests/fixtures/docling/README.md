# Docling 真实集成测试 Fixture

本目录存放问题 2 第一批真实 Docling 集成测试使用的固定样本。

## 文件说明

- `dayu_docling_integration_fixture.pdf`
  - 一页、可搜索文本 PDF
  - 包含标题、章节、小段正文和 1 个稳定表格
  - 用于验证真实 `PDF -> Docling -> Markdown / _docling.json` 链路
- `dayu_docling_integration_fixture.source.html`
  - 生成上述 PDF 的源 HTML
  - 便于后续重新生成或审阅断言依据

## 来源

该样本为项目内自建 fixture，不含第三方受限版权内容。

PDF 由 `playwright` 的 Chromium `page.pdf()` 基于同目录 HTML 生成，避免依赖
外部文档来源或不稳定的截图/OCR 结果。

## 关键断言点

测试围绕以下稳定内容断言：

- 标题：`Dayu Docling Integration Fixture`
- 章节：`Financial Summary`
- 表头：`Metric` / `FY2024` / `FY2025`
- 行值：`Revenue`、`Operating Margin`、`Free Cash Flow`

## 为什么选它

- 文件体积小，可进入常规 CI
- 表格结构简单，Docling 在当前受控依赖环境下可稳定识别
- 能同时覆盖：
  - `DoclingProcessor`
  - `DoclingUploadService`
  - Web 非 HTML Docling 转换路径
