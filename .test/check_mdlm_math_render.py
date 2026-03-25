#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import functools
import http.server
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SLIDE_FILE = ROOT / "slides" / "mdllm.md"
CONFIG_FILE = ROOT / "mkslides.yml"
CHROME_CANDIDATES = (
    "google-chrome",
    "google-chrome-stable",
    "chromium",
    "chromium-browser",
)


def _count_math_blocks(markdown: str) -> int:
    sys.path.insert(0, str(ROOT))
    from preprocess_math import _consume_code_span, _consume_fenced_code, _consume_math

    count = 0
    index = 0
    while index < len(markdown):
        fenced = _consume_fenced_code(markdown, index)
        if fenced is not None:
            _, index = fenced
            continue

        if markdown[index] == "`":
            _, index = _consume_code_span(markdown, index)
            continue

        math = _consume_math(markdown, index)
        if math is not None:
            _, index = math
            count += 1
            continue

        index += 1

    return count


def _find_browser() -> str:
    for candidate in CHROME_CANDIDATES:
        path = shutil.which(candidate)
        if path:
            return path
    raise RuntimeError(
        "No Chrome/Chromium binary found. Tried: "
        + ", ".join(CHROME_CANDIDATES)
    )


class _QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format: str, *args: object) -> None:
        return


def _extract_rendered_sections(dom: str) -> str:
    sections = re.findall(
        r"<section\b[^>]*data-markdown-parsed=\"true\"[^>]*>(.*?)</section>",
        dom,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if not sections:
        raise AssertionError("Reveal did not produce any rendered markdown sections.")
    return "\n".join(sections)


def _check_rendered_math(rendered_html: str, expected_math_blocks: int) -> list[str]:
    problems: list[str] = []

    rendered_math_blocks = (
        rendered_html.count("<mjx-container")
        + rendered_html.count('class="katex"')
    )
    if rendered_math_blocks < expected_math_blocks:
        problems.append(
            "The browser rendered fewer math expressions than expected "
            f"({rendered_math_blocks} rendered vs {expected_math_blocks} source blocks)."
        )

    lowered = rendered_html.lower()
    if "<mjx-merror" in lowered:
        problems.append("MathJax emitted an <mjx-merror> node.")
    if "math input error" in lowered:
        problems.append("MathJax reported a math input error in the rendered DOM.")

    raw_tex_patterns = {
        r"\$\$": "raw display-math delimiters `$$...$$` remain in rendered slides",
        r"(?<!\$)\$(?!\$)": "raw inline-math delimiters `$...$` remain in rendered slides",
        r"\\\(": r"raw inline delimiters `\(...\)` remain in rendered slides",
        r"\\\)": r"raw inline delimiters `\(...\)` remain in rendered slides",
        r"\\\[": r"raw display delimiters `\[...\]` remain in rendered slides",
        r"\\\]": r"raw display delimiters `\[...\]` remain in rendered slides",
    }
    for pattern, message in raw_tex_patterns.items():
        if re.search(pattern, rendered_html):
            problems.append(message)

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build slides/mdllm.md with the current mkslides config, serve the "
            "generated HTML from /tmp, and verify math rendering in headless Chrome."
        )
    )
    parser.add_argument(
        "--keep-artifacts",
        action="store_true",
        help="Keep the temporary /tmp build directory instead of deleting it.",
    )
    parser.add_argument(
        "--virtual-time-budget-ms",
        type=int,
        default=20000,
        help="How long Chrome should wait for JS-driven rendering before dumping the DOM.",
    )
    args = parser.parse_args()

    expected_math_blocks = _count_math_blocks(SLIDE_FILE.read_text())
    chrome = _find_browser()

    tmp_root = Path(tempfile.mkdtemp(prefix="mkslides-math-render.", dir="/tmp"))
    site_dir = tmp_root / "site"
    html_path = site_dir / "index.html"
    dom_dump_path = tmp_root / "mdllm.rendered.html"

    server: http.server.ThreadingHTTPServer | None = None
    server_thread: threading.Thread | None = None
    keep_artifacts = args.keep_artifacts

    try:
        build = subprocess.run(
            [
                "uv",
                "run",
                "mkslides",
                "build",
                str(SLIDE_FILE),
                "-f",
                str(CONFIG_FILE),
                "-d",
                str(site_dir),
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        if build.returncode != 0:
            raise RuntimeError(
                "mkslides build failed.\n"
                f"stdout:\n{build.stdout}\n"
                f"stderr:\n{build.stderr}"
            )
        if not html_path.exists():
            fallback_html = site_dir / f"{SLIDE_FILE.stem}.html"
            if fallback_html.exists():
                html_path = fallback_html
            else:
                raise FileNotFoundError(f"Expected built slide HTML at {html_path}")

        handler = functools.partial(_QuietHandler, directory=str(site_dir))
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()
        url = f"http://127.0.0.1:{server.server_port}/{html_path.name}"

        chrome_run = subprocess.run(
            [
                chrome,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                f"--virtual-time-budget={args.virtual_time_budget_ms}",
                "--dump-dom",
                url,
            ],
            text=True,
            capture_output=True,
            check=False,
        )
        if chrome_run.returncode != 0:
            raise RuntimeError(
                "Headless Chrome failed while rendering the generated slide HTML.\n"
                f"stdout:\n{chrome_run.stdout}\n"
                f"stderr:\n{chrome_run.stderr}"
            )

        dom = chrome_run.stdout
        dom_dump_path.write_text(dom)

        rendered_html = _extract_rendered_sections(dom)
        problems = _check_rendered_math(rendered_html, expected_math_blocks)
        if problems:
            keep_artifacts = True
            raise AssertionError(
                "Math render verification failed:\n- "
                + "\n- ".join(problems)
            )

        print(
            "Math render check passed.\n"
            f"Built slide: {html_path}\n"
            f"Rendered DOM snapshot: {dom_dump_path}\n"
            f"Expected math blocks: {expected_math_blocks}"
        )
        return 0

    except Exception as exc:
        keep_artifacts = True
        print(str(exc), file=sys.stderr)
        print(f"Artifacts kept at: {tmp_root}", file=sys.stderr)
        return 1

    finally:
        if server is not None:
            server.shutdown()
            server.server_close()
        if server_thread is not None:
            server_thread.join(timeout=2)
        if not keep_artifacts:
            with contextlib.suppress(FileNotFoundError):
                shutil.rmtree(tmp_root)


if __name__ == "__main__":
    raise SystemExit(main())
