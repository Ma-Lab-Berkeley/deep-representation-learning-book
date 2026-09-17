"""Exercise build.sh failure handling without installing a TeX toolchain."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


PIPELINE = Path(__file__).resolve().parent
if os.name == "nt":
    # Windows' system32/bash.exe starts WSL, not the Git Bash used by these tests.
    BASH = next((str(path) for path in (
        Path(os.environ.get("ProgramFiles", "C:/Program Files")) / "Git/bin/bash.exe",
        Path("C:/Program Files/Git/bin/bash.exe"),
    ) if path.is_file()), None)
else:
    BASH = shutil.which("bash")


def shell_path(path: Path) -> str:
    value = path.resolve().as_posix()
    return f"/{value[0].lower()}{value[2:]}" if os.name == "nt" else value


# Only external build tools are replaced; build.sh and its log validator are real.
FAKE_TOOLS = r"""
export PATH="/usr/bin:/bin:$PATH"
export TMPDIR="$(cd "$TMPDIR" && pwd)"
rsync() { cp -R "${@: -2:1}." "${@: -1}"; }
kpsewhich() { printf '100\n'; }
xelatex() { printf 'out of 100\n' > texput.log; }
fmtutil-user() { return 0; }
latexmk() {
    printf 'latexmk %s\n' "$*" >> "$TEST_TRACE"
    printf '%s\n' "$TEST_PDF_LOG" > book-main.log
    printf '%%PDF-1.7\n' > book-main.pdf
    printf '\\relax\n' > book-main.aux
    return "$TEST_PDF_STATUS"
}
make4ht() {
    printf 'make4ht\n' >> "$TEST_TRACE"
    printf '%s\n' "$TEST_HTML_LOG" > book-main.log
    return "$TEST_HTML_STATUS"
}
book_test_python() {
    printf 'python %s\n' "$*" >> "$TEST_TRACE"
    "$TEST_PYTHON" "$@"
}
uv() {
    printf 'unexpected uv\n' >> "$TEST_TRACE"
    return 97
}
"""


@unittest.skipUnless(BASH, "Bash is required (Git Bash on Windows)")
class BuildScriptTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="book-build-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "repo"
        pipeline = self.root / "website/latex_to_html"
        pipeline.mkdir(parents=True)
        for name in ("build.sh", "validate_build.py"):
            shutil.copy2(PIPELINE / name, pipeline / name)
        (self.root / "book-main.tex").write_text("fixture\n", encoding="utf-8")
        self.tools = self.root / "fake-tools.sh"
        self.tools.write_text(FAKE_TOOLS, encoding="utf-8", newline="\n")
        self.trace = self.root / "trace.txt"
        self.build_tmp = Path(self.temporary.name) / "temporary-builds"
        self.build_tmp.mkdir()
        self.diagnostics = self.root / "diagnostics"

    def run_build(self, *, strict="1", pdf_status=0, html_status=0,
                  pdf_log="PDF final log: clean", html_log="HTML final log: clean"):
        env = os.environ.copy()
        env.update({
            "BASH_ENV": shell_path(self.tools),
            "TMPDIR": shell_path(self.build_tmp),
            "KEEP_BUILD_DIR": "0",
            "BOOK_BUILD_DIAGNOSTICS": "diagnostics",
            "BOOK_PYTHON": "book_test_python",
            "TEST_TRACE": shell_path(self.trace),
            "TEST_PYTHON": shell_path(Path(sys.executable)),
            "TEST_PDF_STATUS": str(pdf_status),
            "TEST_HTML_STATUS": str(html_status),
            "TEST_PDF_LOG": pdf_log,
            "TEST_HTML_LOG": html_log,
        })
        env.pop("BOOK_BUILD_STRICT", None)
        if strict is not None:
            env["BOOK_BUILD_STRICT"] = strict
        result = subprocess.run(
            [BASH, "website/latex_to_html/build.sh", "book-main.tex", "output"],
            cwd=self.root, env=env, text=True, encoding="utf-8", errors="replace",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=30,
        )
        self.assertEqual(list(self.build_tmp.iterdir()), [], result.stdout)
        return result

    def calls(self):
        return self.trace.read_text(encoding="utf-8") if self.trace.exists() else ""

    def diagnostic(self, stage):
        return (self.diagnostics / f"{stage}.log").read_text(encoding="utf-8")

    def test_strict_pdf_failure_stops_before_html_and_preserves_log(self):
        result = self.run_build(pdf_status=23)
        self.assertEqual(result.returncode, 23, result.stdout)
        self.assertIn("-halt-on-error", self.calls())
        self.assertNotIn("make4ht", self.calls())
        self.assertEqual(self.diagnostic("pdf"), "PDF final log: clean\n")

    def test_default_mode_keeps_legacy_continue_on_pdf_failure(self):
        result = self.run_build(strict=None, pdf_status=23, html_status=31)
        self.assertEqual(result.returncode, 31, result.stdout)
        self.assertIn(" -f ", self.calls())
        self.assertIn("make4ht", self.calls())
        self.assertNotIn("python", self.calls())
        self.assertEqual(self.diagnostic("pdf"), "PDF final log: clean\n")

    def test_strict_unresolved_pdf_reference_stops_before_html(self):
        log = "LaTeX Warning: Reference `missing' on page 1 undefined on input line 4."
        result = self.run_build(pdf_log=log)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("undefined reference", result.stdout)
        self.assertNotIn("make4ht", self.calls())
        self.assertEqual(self.diagnostic("pdf"), log + "\n")

    def test_zero_exit_make4ht_with_fatal_log_fails_and_keeps_both_logs(self):
        log = "! Undefined control sequence."
        result = self.run_build(html_log=log)
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("TeX error", result.stdout)
        self.assertIn("make4ht", self.calls())
        self.assertEqual(self.diagnostic("pdf"), "PDF final log: clean\n")
        self.assertEqual(self.diagnostic("html"), log + "\n")

    def test_strict_missing_html_fails_after_clean_html_log(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("make4ht produced no HTML files", result.stdout)
        self.assertNotIn("[Stage 4]", result.stdout)
        self.assertEqual(self.diagnostic("html"), "HTML final log: clean\n")

    def test_explicit_python_is_used_for_pdf_and_html_validation(self):
        result = self.run_build()
        self.assertEqual(result.returncode, 1, result.stdout)
        calls = self.calls().splitlines()
        python_calls = [line for line in calls if line.startswith("python ")]
        self.assertEqual(len(python_calls), 2, result.stdout)
        self.assertIn("--references", python_calls[0])
        self.assertNotIn("--references", python_calls[1])
        self.assertNotIn("unexpected uv", self.calls())

    def test_make4ht_exit_status_survives_cleanup(self):
        result = self.run_build(html_status=29)
        self.assertEqual(result.returncode, 29, result.stdout)
        self.assertEqual(self.diagnostic("pdf"), "PDF final log: clean\n")
        self.assertEqual(self.diagnostic("html"), "HTML final log: clean\n")

    def test_invalid_strict_value_fails_before_build(self):
        result = self.run_build(strict="yes")
        self.assertEqual(result.returncode, 1, result.stdout)
        self.assertIn("BOOK_BUILD_STRICT must be 0 or 1", result.stdout)
        self.assertEqual(self.calls(), "")


if __name__ == "__main__":
    unittest.main()
