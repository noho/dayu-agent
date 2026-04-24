"""分析报告导出辅助模块。"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path


def generate_report_file(report_path: Path, output_format: str) -> tuple[bytes, str, str]:
    """根据格式生成报告文件内容。"""

    ticker = report_path.stem.replace("_qual_report", "")
    if output_format == "markdown":
        with open(report_path, "rb") as file:
            content = file.read()
        return content, f"{ticker}_qual_report.md", "text/markdown"
    if output_format == "html":
        return convert_to_html(report_path, ticker)
    if output_format == "pdf":
        return convert_to_pdf(report_path, ticker)
    raise ValueError(f"不支持的格式: {output_format}")


def ensure_pandoc_v3_or_newer() -> None:
    """校验 pandoc 版本是否满足 3.0+ 要求。"""

    try:
        version_result = subprocess.run(
            ["pandoc", "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError as exception:
        raise RuntimeError("pandoc 未安装，无法生成 HTML/PDF 格式") from exception
    except subprocess.CalledProcessError as exception:
        raise RuntimeError("执行 pandoc --version 失败，无法识别版本") from exception

    first_line = version_result.stdout.splitlines()[0] if version_result.stdout else ""
    match = re.search(r"pandoc\s+(?P<major>\d+)\.(?P<minor>\d+)\.(?P<patch>\d+)", first_line)
    if match is None:
        raise RuntimeError(f"无法解析 pandoc 版本信息: {first_line}")
    major = int(match.group("major"))
    if major < 3:
        raise RuntimeError(f"当前 pandoc 版本为 {first_line}，请升级到 3.0 及以上。")


def convert_to_html(report_path: Path, ticker: str) -> tuple[bytes, str, str]:
    """将 Markdown 报告转换为 HTML 格式。"""

    render_dir = Path(__file__).parents[4] / "render"
    assets_dir = render_dir
    diagram_filter = assets_dir / "diagram.lua"
    ensure_pandoc_v3_or_newer()

    if not diagram_filter.is_file():
        raise RuntimeError(f"渲染资源缺失: {diagram_filter}")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        html_output = temp_path / f"{ticker}_qual_report.html"
        resource_path = os.pathsep.join([str(report_path.parent.resolve()), str(assets_dir.resolve())])
        command: list[str] = [
            "pandoc",
            str(report_path),
            f"--lua-filter={diagram_filter}",
            f"--resource-path={resource_path}",
            "-f",
            "gfm+hard_line_breaks",
            "-t",
            "html5",
            "-s",
            "--embed-resources",
            f"--css={assets_dir / 'github-markdown.css'}",
            f"--include-before-body={assets_dir / 'before.html'}",
            f"--include-after-body={assets_dir / 'after.html'}",
            "-o",
            str(html_output),
        ]
        try:
            subprocess.run(command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exception:
            raise RuntimeError(f"HTML 转换失败: {exception.stderr}")
        except FileNotFoundError as exception:
            raise RuntimeError("pandoc 未安装，无法生成 HTML 格式") from exception
        with open(html_output, "rb") as file:
            content = file.read()
    return content, f"{ticker}_qual_report.html", "text/html"


def convert_to_pdf(report_path: Path, ticker: str) -> tuple[bytes, str, str]:
    """将 Markdown 报告转换为 PDF 格式。"""

    render_dir = Path(__file__).parents[4] / "render"
    assets_dir = render_dir
    diagram_filter = assets_dir / "diagram.lua"
    ensure_pandoc_v3_or_newer()

    if not diagram_filter.is_file():
        raise RuntimeError(f"渲染资源缺失: {diagram_filter}")

    chrome_bin = os.environ.get("PUPPETEER_EXECUTABLE_PATH", "").strip()
    if not chrome_bin:
        resolved_chrome_bin = shutil.which("google-chrome")
        chrome_bin = resolved_chrome_bin if isinstance(resolved_chrome_bin, str) else ""
    if not chrome_bin:
        mac_chrome = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
        if Path(mac_chrome).is_file():
            chrome_bin = mac_chrome

    if not chrome_bin or not Path(chrome_bin).is_file():
        raise RuntimeError("Chrome 未找到，无法生成 PDF 格式。请设置 PUPPETEER_EXECUTABLE_PATH 或安装 Google Chrome。")

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        html_output = temp_path / f"{ticker}_qual_report.html"
        pdf_output = temp_path / f"{ticker}_qual_report.pdf"
        resource_path = os.pathsep.join([str(report_path.parent.resolve()), str(assets_dir.resolve())])
        pandoc_command: list[str] = [
            "pandoc",
            str(report_path),
            f"--lua-filter={diagram_filter}",
            f"--resource-path={resource_path}",
            "-f",
            "gfm+hard_line_breaks",
            "-t",
            "html5",
            "-s",
            "--embed-resources",
            f"--css={assets_dir / 'github-markdown.css'}",
            f"--include-before-body={assets_dir / 'before.html'}",
            f"--include-after-body={assets_dir / 'after.html'}",
            "-o",
            str(html_output),
        ]
        try:
            subprocess.run(pandoc_command, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as exception:
            raise RuntimeError(f"PDF 中间 HTML 转换失败: {exception.stderr}")
        except FileNotFoundError as exception:
            raise RuntimeError("pandoc 未安装，无法生成 PDF 格式") from exception

        html_uri = html_output.resolve().as_uri()
        chrome_args = [
            chrome_bin,
            "--headless",
            "--disable-gpu",
            "--disable-background-networking",
            "--disable-default-apps",
            "--disable-component-update",
            "--disable-client-side-phishing-detection",
            "--disable-features=TranslateUI",
            "--disable-sync",
            "--disable-extensions",
            "--metrics-recording-only",
            "--password-store=basic",
            "--use-mock-keychain",
            "--no-first-run",
            "--no-default-browser-check",
            "--incognito",
            "--bwsi",
            "--disable-logging",
            "--log-level=3",
            "--disable-popup-blocking",
            "--disable-notifications",
            "--run-all-compositor-stages-before-draw",
            "--virtual-time-budget=10000",
            f"--print-to-pdf={pdf_output}",
            "--print-to-pdf-no-header",
            "--no-pdf-header-footer",
            html_uri,
        ]
        try:
            subprocess.run(chrome_args, check=True, capture_output=True)
        except subprocess.CalledProcessError as exception:
            raise RuntimeError("PDF 生成失败: Chrome 转换出错") from exception

        with open(pdf_output, "rb") as file:
            content = file.read()
    return content, f"{ticker}_qual_report.pdf", "application/pdf"
