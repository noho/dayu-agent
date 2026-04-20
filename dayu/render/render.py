#!/usr/bin/env python3
"""将 Markdown 渲染为 HTML/PDF/Word。

Usage:
    python render.py <input_markdown> [output_docx]
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from dayu.console_output import configure_standard_streams_for_console_output


DEFAULT_CHROME_MAC_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
HTML_EXTENSIONS = {"html", "htm"}
WORD_EXTENSIONS = {"docx", "doc"}
MARKDOWN_INPUT_FORMAT = "gfm+hard_line_breaks"


def _print_help() -> None:
    """输出 `dayu-render` 的帮助信息。

    参数:
        无。

    返回值:
        无。

    异常:
        无。
    """
    print("Usage: dayu-render <input_markdown> [output_path]")
    print()
    print("将 Markdown 渲染为 HTML / PDF / Word。")


def _ensure_dir(path: Path) -> None:
    """确保目录存在。

    参数:
        path: 需要创建或确认存在的目录路径。

    返回值:
        无。

    异常:
        OSError: 目录创建失败时抛出。
    """
    path.mkdir(parents=True, exist_ok=True)


def _normalize_ext(path: Path) -> str:
    """返回输出文件扩展名（小写且不含前导点）。

    参数:
        path: 输出文件路径。

    返回值:
        标准化后的扩展名字符串。

    异常:
        无。
    """
    return path.suffix.lower().lstrip(".")


def _build_resource_path(input_md: Path, assets_dir: Path, output_dir: Path) -> str:
    """构建 pandoc 资源搜索路径。

    参数:
        input_md: 输入 Markdown 文件路径。
        assets_dir: 渲染资源目录（css、模板、lua 过滤器）。
        output_dir: 输出文件目录。

    返回值:
        供 `--resource-path` 使用的路径字符串。

    异常:
        无。
    """
    unique_dirs = []
    for path in (input_md.parent.resolve(), assets_dir.resolve(), output_dir.resolve()):
        if path not in unique_dirs:
            unique_dirs.append(path)
    return os.pathsep.join(str(path) for path in unique_dirs)


def _run_pandoc(args: list[str], *, cwd: Path) -> None:
    """执行 pandoc 命令。

    参数:
        args: 传递给 pandoc 的参数列表（不含可执行文件名）。
        cwd: 运行命令时的工作目录。

    返回值:
        无。

    异常:
        subprocess.CalledProcessError: pandoc 返回非零退出码时抛出。
        FileNotFoundError: 系统中未安装 pandoc 时抛出。
    """
    subprocess.run(["pandoc", *args], check=True, cwd=str(cwd))


def _build_pandoc_base_args(input_md: Path, assets_dir: Path, output_dir: Path) -> list[str]:
    """构建 pandoc 公共参数。

    参数:
        input_md: 输入 Markdown 文件路径。
        assets_dir: 渲染资源目录。
        output_dir: 输出文件所在目录。

    返回值:
        pandoc 公共参数列表。

    异常:
        无。
    """
    resource_path = _build_resource_path(input_md, assets_dir, output_dir)
    return [
        str(input_md),
        f"--lua-filter={assets_dir / 'diagram.lua'}",
        f"--resource-path={resource_path}",
        "-f",
        MARKDOWN_INPUT_FORMAT,
        "-s",
        "--embed-resources",
    ]


def _generate_html(input_md: Path, target_html: Path, assets_dir: Path) -> None:
    """生成 HTML 文件。

    参数:
        input_md: 输入 Markdown 文件路径。
        target_html: 目标 HTML 文件路径。
        assets_dir: 渲染资源目录。

    返回值:
        无。

    异常:
        subprocess.CalledProcessError: pandoc 执行失败时抛出。
        FileNotFoundError: 系统中未安装 pandoc 时抛出。
    """
    _ensure_dir(target_html.parent)
    args = _build_pandoc_base_args(input_md, assets_dir, target_html.parent)
    args.extend(
        [
            "-t",
            "html5",
            f"--css={assets_dir / 'github-markdown.css'}",
            f"--include-before-body={assets_dir / 'before.html'}",
            f"--include-after-body={assets_dir / 'after.html'}",
            "-o",
            str(target_html),
        ]
    )

    _run_pandoc(args, cwd=target_html.parent)


def _generate_word(input_md: Path, target_path: Path, assets_dir: Path) -> None:
    """生成 Word 文件。

    参数:
        input_md: 输入 Markdown 文件路径。
        target_path: 目标 Word 文件路径。
        assets_dir: 渲染资源目录。

    返回值:
        无。

    异常:
        subprocess.CalledProcessError: pandoc 执行失败时抛出。
        FileNotFoundError: 系统中未安装 pandoc 时抛出。
    """
    _ensure_dir(target_path.parent)
    args = _build_pandoc_base_args(input_md, assets_dir, target_path.parent)
    reference_doc = assets_dir / "reference.docx"
    if reference_doc.is_file():
        args.append(f"--reference-doc={reference_doc}")
    args.extend(["-o", str(target_path)])

    _run_pandoc(args, cwd=target_path.parent)


def _generate_generic(input_md: Path, target_path: Path, assets_dir: Path) -> None:
    """生成通用输出格式文件（非 HTML/PDF/Word）。

    参数:
        input_md: 输入 Markdown 文件路径。
        target_path: 目标输出文件路径。
        assets_dir: 渲染资源目录。

    返回值:
        无。

    异常:
        subprocess.CalledProcessError: pandoc 执行失败时抛出。
        FileNotFoundError: 系统中未安装 pandoc 时抛出。
    """
    _ensure_dir(target_path.parent)
    args = _build_pandoc_base_args(input_md, assets_dir, target_path.parent)
    args.extend(["-o", str(target_path)])

    _run_pandoc(args, cwd=target_path.parent)


def _find_chrome_binary() -> str:
    """查找可用的 Chrome 可执行文件路径。

    优先读取 `PUPPETEER_EXECUTABLE_PATH`，其次尝试系统 PATH 中的
    `google-chrome`，最后尝试 macOS 默认安装路径。

    参数:
        无。

    返回值:
        Chrome 可执行文件绝对路径。

    异常:
        SystemExit: 未找到 Chrome 可执行文件时抛出。
    """
    chrome_bin = os.environ.get("PUPPETEER_EXECUTABLE_PATH", "").strip()
    if not chrome_bin:
        if shutil.which("google-chrome"):
            chrome_bin = shutil.which("google-chrome") or ""
        elif Path(DEFAULT_CHROME_MAC_PATH).is_file():
            chrome_bin = DEFAULT_CHROME_MAC_PATH

    if not chrome_bin or not Path(chrome_bin).is_file():
        raise SystemExit(
            "Chrome binary not found. Set PUPPETEER_EXECUTABLE_PATH or install Google Chrome."
        )
    return chrome_bin


def _convert_html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    """使用 Headless Chrome 将 HTML 转换为 PDF。

    参数:
        html_path: 输入 HTML 文件路径。
        pdf_path: 输出 PDF 文件路径。

    返回值:
        无。

    异常:
        SystemExit: 未找到 Chrome 可执行文件时抛出。
        subprocess.CalledProcessError: Chrome 执行失败时抛出。
    """
    chrome_bin = _find_chrome_binary()
    html_uri = html_path.resolve().as_uri()
    chrome_args = [
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
        f"--print-to-pdf={pdf_path}",
        "--print-to-pdf-no-header",
        "--no-pdf-header-footer",
        html_uri,
    ]

    subprocess.run([chrome_bin, *chrome_args], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _open_file(path: Path) -> None:
    """尝试使用系统默认程序打开生成文件。

    参数:
        path: 需要打开的文件路径。

    返回值:
        无。

    异常:
        无。内部会吞掉异常并回退到标准错误输出提示。
    """
    try:
        if sys.platform.startswith("darwin"):
            subprocess.run(["open", str(path)], check=False)
        elif sys.platform.startswith("linux"):
            subprocess.run(["xdg-open", str(path)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        elif sys.platform.startswith("win"):
            os.startfile(str(path))  # type: ignore[attr-defined]
        else:
            print(f"Generated file: {path}", file=sys.stderr)
    except Exception:
        print(f"Generated file: {path}", file=sys.stderr)


def main() -> int:
    """脚本入口函数。

    参数:
        无。命令行参数从 `sys.argv` 读取。

    返回值:
        退出码。`0` 表示成功，`1` 表示参数或输入文件错误。

    异常:
        SystemExit: 依赖检查失败时抛出。
        subprocess.CalledProcessError: 外部命令执行失败时抛出。
        FileNotFoundError: 外部命令不存在时抛出。
    """
    configure_standard_streams_for_console_output()

    if len(sys.argv) >= 2 and sys.argv[1] in {"-h", "--help"}:
        _print_help()
        return 0

    if len(sys.argv) < 2:
        print("Usage: dayu-render <input_markdown> [output_path]", file=sys.stderr)
        return 1

    assets_dir = Path(__file__).resolve().parent
    diagram_filter = assets_dir / "diagram.lua"
    if not diagram_filter.is_file():
        raise SystemExit(f"diagram.lua not found: {diagram_filter}")

    input_md_raw = Path(sys.argv[1])
    input_md = input_md_raw.expanduser().resolve()
    if not input_md.is_file():
        print(f"Input markdown not found: {input_md}", file=sys.stderr)
        return 1

    if len(sys.argv) >= 3:
        output_path_raw = Path(sys.argv[2])
    else:
        output_path_raw = input_md.with_suffix(".docx")

    output_path = output_path_raw.expanduser().resolve()
    _ensure_dir(output_path.parent)

    ext = _normalize_ext(output_path)
    is_html = ext in HTML_EXTENSIONS
    is_pdf = ext == "pdf"
    is_word = ext in WORD_EXTENSIONS

    if is_pdf:
        html_target = output_path.with_suffix(".html")
        _ensure_dir(html_target.parent)

        _generate_html(input_md, html_target, assets_dir)
        _convert_html_to_pdf(html_target, output_path)
    elif is_html:
        _generate_html(input_md, output_path, assets_dir)
    elif is_word:
        _generate_word(input_md, output_path, assets_dir)
    else:
        _generate_generic(input_md, output_path, assets_dir)

    _open_file(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
