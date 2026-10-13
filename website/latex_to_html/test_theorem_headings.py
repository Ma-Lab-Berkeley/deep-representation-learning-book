"""Run with: python -m unittest discover -s website/latex_to_html -v"""

from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from bs4 import BeautifulSoup

from postprocess import tag_theorem_environments


PIPELINE_DIR = Path(__file__).resolve().parent
REPO_ROOT = PIPELINE_DIR.parent.parent


class TheoremHeadingsTest(unittest.TestCase):
    def test_chinese_chapter_three_headings_keep_numbers_and_links(self):
        # Chapter 3's TeX4ht structure, with the names emitted by book.cfg.
        soup = BeautifulSoup("""
            <div class="newtheorem"><p><span class="head">
            <a id="x11-113002r1"></a>
            定理 <span class="rm-lmbx-10x-x-105">3.1 </span>
            (<a href="A2.html#x20-485001r2">B.2</a>的简化版).
            </span>正文</p></div>
            <div class="newtheorem"><p><span class="head">
            <a id="x11-113005r2"></a>
            例 3.2 (向高斯混合添加噪声).
            </span>正文</p></div>
        """, "html.parser")
        headings_before = [str(head) for head in soup.select(".head")]
        tag_theorem_environments(soup)
        divs = soup.select("div.newtheorem")
        self.assertIn("theorem-theorem", divs[0]["class"])
        self.assertIn("theorem-example", divs[1]["class"])
        self.assertEqual(headings_before, [str(head) for head in soup.select(".head")])

    def test_all_chinese_theorem_types(self):
        names = {
            "定理": "theorem", "推论": "corollary", "关键思想": "keyidea",
            "引理": "lemma", "方法": "method", "命题": "proposition",
            "公理": "axiom", "假设": "assumption", "定义": "definition",
            "模型": "model", "记号": "notation", "例": "example",
            "断言": "claim", "猜想": "conjecture", "练习": "exercise",
            "想法": "idea", "事实": "fact", "问题": "problem", "注": "remark",
        }
        for name, env in names.items():
            for separator in ("", " ", "\u00a0"):
                with self.subTest(name=name, separator=separator):
                    soup = BeautifulSoup(
                        f'<div class="newtheorem"><span class="head">'
                        f'{name}{separator}3.1.</span></div>', "html.parser"
                    )
                    tag_theorem_environments(soup)
                    self.assertIn(f"theorem-{env}", soup.div["class"])

    def test_english_headings_are_unchanged(self):
        for name, env in (("Theorem", "theorem"), ("Example", "example"),
                          ("Key Idea", "keyidea")):
            with self.subTest(name=name):
                soup = BeautifulSoup(
                    f'<div class="newtheorem"><span class="head">'
                    f'<b>{name}</b><span>3.1.</span></span></div>', "html.parser"
                )
                before = str(soup.select_one(".head"))
                tag_theorem_environments(soup)
                self.assertIn(f"theorem-{env}", soup.div["class"])
                self.assertEqual(before, str(soup.select_one(".head")))

    def test_does_not_style_outer_layout_containers(self):
        soup = BeautifulSoup(
            '<div id="content"><div class="newtheorem"><p>'
            '<span class="head">定理 3.1.</span>正文</p></div></div>',
            "html.parser",
        )
        tag_theorem_environments(soup)
        self.assertNotIn("theorem-env", soup.find(id="content").get("class", []))
        self.assertIn("theorem-theorem", soup.select_one(".newtheorem")["class"])

    def test_does_not_infer_type_from_bare_number_or_prose(self):
        for heading in ("3.2 (向高斯混合添加噪声)", "例如，下式成立", "注意事项"):
            with self.subTest(heading=heading):
                soup = BeautifulSoup(
                    f'<div><b>{heading}</b></div>', "html.parser"
                )
                tag_theorem_environments(soup)
                self.assertNotIn("theorem-env", soup.div.get("class", []))

    def test_tagging_is_idempotent(self):
        soup = BeautifulSoup(
            '<div class="newtheorem"><span class="head">例 3.2.</span></div>',
            "html.parser",
        )
        tag_theorem_environments(soup)
        first = str(soup)
        tag_theorem_environments(soup)
        self.assertEqual(first, str(soup))


@unittest.skipUnless(shutil.which("make4ht"), "make4ht is required for the TeX smoke test")
class Make4htTheoremHeadingsTest(unittest.TestCase):
    def test_chinese_names_survive_conversion(self):
        with tempfile.TemporaryDirectory(prefix="theorem-headings-") as directory:
            build_dir = Path(directory)
            shutil.copy(PIPELINE_DIR / "tests" / "theorem-headings.tex", build_dir)
            shutil.copy(REPO_ROOT / "math-theorems_zh.sty", build_dir)
            result = subprocess.run(
                ["make4ht", "-x", "-u", "-c", str(PIPELINE_DIR / "book.cfg"),
                 "-e", str(PIPELINE_DIR / "book.mk4"), "theorem-headings.tex",
                 "html,mathjax,2,fn-in"],
                cwd=build_dir, capture_output=True, text=True, encoding="utf-8",
                errors="replace", timeout=180,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            heads = []
            for html_path in sorted(build_dir.glob("*.html")):
                soup = BeautifulSoup(html_path.read_text(encoding="utf-8"), "html.parser")
                heads.extend(soup.select(".newtheorem .head"))
            text = [head.get_text(" ", strip=True) for head in heads]
            for name, number in (("例", "3.1"), ("定理", "3.1"), ("例", "3.2"),
                                 ("定理", "3.2"), ("注", "3.1"), ("练习", "3.1")):
                pattern = re.compile(rf"^{name}\s*{re.escape(number)}(?!\d|\.\d)")
                self.assertEqual(sum(bool(pattern.match(value)) for value in text), 1, text)
            # Type names must remain real text, not CSS-generated content or images.
            self.assertTrue(all(not head.find("img") for head in heads))
            self.assertTrue(any(head.find("a", href=True) for head in heads), text)


if __name__ == "__main__":
    unittest.main()
