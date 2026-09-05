"""Regression coverage for final build validation; no TeX installation needed."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from bs4 import BeautifulSoup

from validate_build import REQUIRED_HTML, main, validate_html, validate_tex_log


class BuildValidationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def write(self, name, text):
        path = self.root / name
        path.write_text(text, encoding="utf-8")
        return path

    def create_html_tree(self):
        for name in REQUIRED_HTML:
            self.write(name, "<html><head><title>Chapter</title></head><body>"
                       '<h2 id="section">Chapter</h2><p>Book content.</p></body></html>')
        self.write("A2.html", '<html><body><h2>Appendix</h2><p>Book content.</p>'
                   '<dl class="thebibliography"><dt>1</dt>'
                   '<dd id="bib-1">A cited publication.</dd></dl></body></html>')
        self.index = {
            "generated": "2026-09-05T00:00:00+00:00",
            "count": 1,
            "entries": [{"page": "Chapter", "href": "Ch1.html#section",
                         "title": "Chapter", "snippet": "Book content."}],
        }
        self.write("search-index.json", json.dumps(self.index))

    def test_missing_and_empty_logs_fail(self):
        self.assertTrue(validate_tex_log(self.root / "missing.log"))
        self.assertTrue(validate_tex_log(self.write("empty.log", " \n")))

    def test_clean_log_and_biber_log_pass(self):
        log = self.write("book.log", "This is XeTeX\nOutput written on book.pdf (3 pages).\n")
        biber = self.write("book.blg", "INFO - Output to book.bbl\n")
        self.assertEqual(validate_tex_log(log, references=True, biber_log=biber), [])

    def test_fatal_tex_errors_fail(self):
        for text in ("! Undefined control sequence.", "Emergency stop.",
                     "Package test Error: Failed.", "Fatal error occurred",
                     "./book.tex:12: Undefined control sequence.",
                     "./book.tex:12: Missing $ inserted.", "No pages of output."):
            with self.subTest(text=text):
                self.assertTrue(validate_tex_log(self.write("book.log", text)))

    def test_unresolved_and_wrapped_reference_warnings_fail(self):
        for text in (
            "LaTeX Warning: Reference `eq:test' on page 1 undefined on input line 50.",
            "LaTeX Warning: Reference `very-long-label' on page 1\n undefined on input line 50.",
            "Package biblatex Warning: Citation 'missing' on page 3 undefined.",
            "Package biblatex Warning: Citation 'missing'\n(biblatex) on page 3 undefined.",
            "LaTeX Warning: There were undefined references.",
            "LaTeX Warning: There were undefined citations.",
            "LaTeX Warning: Label `eq:test' multiply defined.",
            "LaTeX Warning: There were multiply-defined labels.",
            "LaTeX Warning: Label(s) may have changed. Rerun to get cross-references right.",
            "Package biblatex Warning: Please (re)run Biber on the file:\n(biblatex) book",
            "Package rerunfilecheck Warning: File `book.out' has changed.\n"
            "(rerunfilecheck) Rerun to get outlines right.",
        ):
            with self.subTest(text=text):
                path = self.write("book.log", text)
                self.assertTrue(validate_tex_log(path, references=True))
                self.assertEqual(validate_tex_log(path), [])

    def test_explicit_biber_log_must_exist_and_have_no_errors(self):
        log = self.write("book.log", "Output written on book.pdf")
        self.assertTrue(validate_tex_log(log, biber_log=self.root / "missing.blg"))
        for text in ("ERROR - Cannot find book.bcf", "WARN - I didn't find a database entry for 'x'",
                     "WARN - I didn't find database entry for 'x'",
                     "WARN - I didn't find a database\nentry for 'x'", ""):
            with self.subTest(text=text):
                self.assertTrue(validate_tex_log(log, biber_log=self.write("book.blg", text)))

    def test_valid_complete_html_tree_passes(self):
        self.create_html_tree()
        self.assertEqual(validate_html(self.root), [])

    def test_missing_required_html_fails(self):
        self.create_html_tree()
        for name in ("Ch3.html", "Chx1.html", "A2.html"):
            with self.subTest(name=name):
                path = self.root / name
                original = path.read_text(encoding="utf-8")
                path.unlink()
                self.assertTrue(any(name in error for error in validate_html(self.root)))
                self.write(name, original)

    def test_bibliography_can_be_inline_or_in_a_separate_generated_page(self):
        self.create_html_tree()
        bibliography = (self.root / "A2.html").read_text(encoding="utf-8")
        self.write("A2.html", '<html><body><h2>Appendix</h2><p>Content.</p></body></html>')
        self.assertTrue(any("bibliography" in error for error in validate_html(self.root)))
        for name in ("Chx8.html", "bib.html"):
            with self.subTest(name=name):
                self.write(name, bibliography)
                self.assertEqual(validate_html(self.root), [])
                (self.root / name).unlink()

    def test_empty_bibliography_fails(self):
        self.create_html_tree()
        self.write("A2.html", '<html><body><h2>Appendix</h2><p>Content.</p>'
                   '<dl class="thebibliography"><dd> </dd></dl></body></html>')
        self.assertTrue(any("bibliography" in error for error in validate_html(self.root)))

    def test_extra_generated_pages_are_checked_but_static_shell_is_not(self):
        self.create_html_tree()
        self.write("index.html", "<html><body><div id='app'></div></body></html>")
        self.assertEqual(validate_html(self.root), [])
        for name in ("Chx3.html", "Ch10.html", "Ax3.html", "A3.html"):
            with self.subTest(name=name):
                self.write(name, "<html><body><h2>Extra page</h2><p>See ??</p></body></html>")
                self.assertTrue(any(name in error for error in validate_html(self.root)))
                (self.root / name).unlink()

    def test_blank_or_incomplete_html_fails(self):
        self.create_html_tree()
        for text in ("", "not HTML", "<html><body></body></html>",
                     "<html><body><h2>Chapter</h2></body></html>"):
            with self.subTest(text=text):
                self.write("Ch3.html", text)
                self.assertTrue(validate_html(self.root))

    def test_visible_unresolved_references_fail(self):
        self.create_html_tree()
        for text in ("See ??", r"See \ref{x}", r"See \eqref {x}", r"See \cref{x}", r"See \Cref*{x}"):
            with self.subTest(text=text):
                self.write("Ch3.html", f"<html><body><h2>Chapter</h2><p>{text}</p></body></html>")
                self.assertTrue(validate_html(self.root))

    def test_comments_code_scripts_and_legitimate_labels_are_ignored(self):
        self.create_html_tree()
        self.write("Ch3.html", r"""<html><body><h2>Chapter</h2><p>Content \label{x}.</p>
            <!-- ?? \ref{comment} --><script>let value = x ?? y; "\ref{js}"</script>
            <style>/* ?? \ref{style} */</style><pre>?? \ref{pre}</pre><code>?? \ref{code}</code>
            <p hidden>?? \ref{hidden}</p><p aria-hidden="true">??</p></body></html>""")
        self.assertEqual(validate_html(self.root), [])

    def test_invalid_search_indexes_fail(self):
        self.create_html_tree()
        for payload in ([], {}, {**self.index, "entries": []}, {**self.index, "count": 2},
                        {**self.index, "count": True}, {**self.index, "generated": "not a date"},
                        {**self.index, "entries": [{"title": "missing fields"}]},
                        {**self.index, "entries": [{**self.index["entries"][0], "href": "https://example.com/#x"}]}):
            with self.subTest(payload=payload):
                self.write("search-index.json", json.dumps(payload))
                self.assertTrue(validate_html(self.root))
        self.write("search-index.json", "{invalid")
        self.assertTrue(validate_html(self.root))
        (self.root / "search-index.json").unlink()
        self.assertTrue(validate_html(self.root))

    def test_pdf_signature_is_required_when_requested(self):
        self.create_html_tree()
        path = self.root / "book.pdf"
        self.assertTrue(validate_html(self.root, pdf=path))
        for text in ("", "not a PDF", "%PDF-"):
            self.write("book.pdf", text)
            self.assertTrue(validate_html(self.root, pdf=path))
        self.write("book.pdf", "%PDF-1.7\nMock body")
        self.assertEqual(validate_html(self.root, pdf=path), [])

    def test_cli_exit_status(self):
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            self.assertEqual(main(["tex-log", str(self.root / "missing.log")]), 1)
            log = self.write("book.log", "Output written on book.pdf")
            self.assertEqual(main(["tex-log", str(log), "--references"]), 0)
            self.create_html_tree()
            self.assertEqual(main(["html", str(self.root)]), 0)

    def test_ci_manifest_escapes_links_and_preserves_relative_targets(self):
        self.create_html_tree()
        entries = [
            {"page": "Chapter", "href": 'Ch1.html#section?x="a"&y=<b>',
             "title": 'Title <script> & "quoted"', "snippet": "Text"},
            {"page": "Appendix", "href": "A1.html#proof", "title": "Proof", "snippet": ""},
        ]
        self.write("search-index.json", json.dumps({**self.index, "count": 2, "entries": entries}))
        with redirect_stdout(io.StringIO()):
            self.assertEqual(main(["html", str(self.root), "--write-link-manifest"]), 0)
        manifest = (self.root / "_ci-search-links.html").read_text(encoding="utf-8")
        self.assertIn("&quot;a&quot;&amp;y=&lt;b&gt;", manifest)
        soup = BeautifulSoup(manifest, "html.parser")
        self.assertIsNone(soup.script)
        self.assertEqual([link["href"] for link in soup.find_all("a")],
                         [entry["href"] for entry in entries])
        self.assertEqual([link.get_text() for link in soup.find_all("a")],
                         [entry["title"] for entry in entries])
        self.assertEqual(validate_html(self.root), [])

    def test_ci_manifest_is_not_written_when_validation_fails(self):
        self.create_html_tree()
        self.write("Ch3.html", "")
        with redirect_stderr(io.StringIO()):
            self.assertEqual(main(["html", str(self.root), "--write-link-manifest"]), 1)
        self.assertFalse((self.root / "_ci-search-links.html").exists())


if __name__ == "__main__":
    unittest.main()
