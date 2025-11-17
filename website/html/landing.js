/* Extracted from index.html inline script */
(function () {
  // Use global get_text from common.js

  function formatFooter(template, year) {
    return template.replace("{year}", year);
  }

  function getLastUpdatedText() {
    try {
      var ui = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui) || {};
      var raw = document.lastModified || "";
      var d = raw ? new Date(raw) : new Date();
      if (isNaN(d.getTime())) d = new Date();
      var locale = ui.dateLocale || "en-US";
      var formatted = d.toLocaleDateString(locale, {
        year: "numeric",
        month: "long",
        day: "numeric",
      });
      var template = ui.lastUpdatedTemplate || "Last Updated: {date}";
      return String(template).replace("{date}", formatted);
    } catch (e) {
      return "Last Updated: " + new Date().toISOString().slice(0, 10);
    }
  }

  function getVersionText() {
    try {
      var raw =
        (window.get_text &&
          window.get_text("landing.hero.cover.version")) || "";
      if (Array.isArray(raw)) raw = raw.join(" ");
      var str = String(raw || "");
      var first = (str.split(/\r?\n/) || [""])[0] || "";
      return first.trim();
    } catch (e) {
      return "";
    }
  }

  function Main() {
    return React.createElement(
      "main",
      { className: "main" },
      React.createElement(
        "section",
        { className: "hero" },
        React.createElement(
          "div",
          { className: "hero-card" },
          React.createElement(
            "h1",
            { className: "hero-title" },
            (window.get_text_inline &&
              window.get_text_inline("landing.hero.title")) ||
              ""
          ),
          React.createElement(
            "div",
            { className: "hero-authors" },
            (window.get_text_inline &&
              window.get_text_inline("landing.hero.authors")) ||
              ""
          ),
          React.createElement(
            "p",
            { className: "hero-sub" },
            (window.get_text_inline &&
              window.get_text_inline("landing.hero.subtitle")) ||
              ""
          ),
          // React.createElement('div', { className: 'pub-info' },
          //   React.createElement('div', { className: 'pub-info-title' }, 'Publication Information'),
          //   React.createElement('p', null, 'Placeholder: publication details (publisher, edition, ISBN, publication date) will go here.')
          // ),
          React.createElement(
            "div",
            { className: "citation-info" },
            React.createElement(
              "code",
              { style: { whiteSpace: "pre-wrap" } },
              String.raw` @book{ldrdd2025,
  title={Learning Deep Representations of Data Distributions},
  author={Buchanan, Sam and Pai, Druv and Wang, Peng and Ma, Yi},
  month=aug,
  year={2025},
  publisher={Online},
  note={\url{https://ma-lab-berkeley.github.io/deep-representation-learning-book/}.}
}`
            )
          ),
          React.createElement(
            "div",
            { className: "cta-row" },
            React.createElement(
              "a",
              { className: "btn", href: "Chx1.html" },
              (window.get_text_inline &&
                window.get_text_inline("landing.hero.buttons.readHtml")) ||
                ""
            ),
            React.createElement(
              "a",
              { className: "btn", href: window.BOOK_COMPONENTS.bookPdfPath },
              (window.get_text_inline &&
                window.get_text_inline("landing.hero.buttons.readPdf")) ||
                ""
            ),
            React.createElement(
              "a",
              {
                className: "btn secondary",
                href: "https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book",
                target: "_blank",
                rel: "noopener noreferrer",
              },
              (window.get_text_inline &&
                window.get_text_inline("landing.hero.buttons.github")) ||
                ""
            )
          )
        ),
        React.createElement(
          "div",
          { className: "hero-figure" },
          React.createElement(
            "a",
            {
              className: "cover-ph",
              href: "Chx1.html",
              title:
                (window.get_text &&
                  window.get_text("landing.hero.cover.title")) ||
                "",
            },
            React.createElement("img", {
              className: "cover-img",
              src: window.BOOK_COMPONENTS.coverImagePath,
              alt:
                (window.get_text &&
                  window.get_text("landing.hero.cover.alt")) ||
                "",
              loading: "lazy",
            })
          ),
          React.createElement(
            "div",
            { className: "cover-version" },
            (function () {
              var v = getVersionText();
              var u = getLastUpdatedText();
              return v ? v + "\n" + u : u;
            })()
          )
        )
      ),
      React.createElement(
        "section",
        { className: "sections" },
        React.createElement(
          "div",
          { className: "section-card" },
          React.createElement(
            "h3",
            null,
            (window.get_text &&
              window.get_text("landing.sections.about.title")) ||
              ""
          ),
          ((window.get_text_block &&
            window.get_text_block("landing.sections.about.content")) ||
            null)
        ),
        React.createElement(
          "div",
          { className: "section-card" },
          React.createElement(
            "h3",
            null,
            (window.get_text &&
              window.get_text("landing.sections.acknowledgements.title")) ||
              ""
          ),
          ((window.get_text_block &&
            window.get_text_block(
              "landing.sections.acknowledgements.content"
            )) ||
            null)
        )
        // React.createElement('div', { className: 'section-card' },
        //   React.createElement('h3', null, 'Citation'),
        //   React.createElement('p', null, 'Placeholder: citation information and BibTeX entry will be provided here.')
        // )
      ),
      React.createElement(
        "div",
        { className: "footer" },
        formatFooter(
          (window.get_text && window.get_text("ui.footer")) || "",
          new Date().getFullYear()
        )
      )
    );
  }

  function App() {
    return React.createElement(
      "div",
      { className: "app-shell" },
      React.createElement(Main, null)
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(
    React.createElement(App)
  );
  if (window.insertTopBar) {
    try {
      window.insertTopBar(
        Object.assign({}, window.TOPBAR_OPTIONS || {}, { forceReplace: true })
      );
    } catch (e) {}
  }
  if (window.insertSidebar) {
    try {
      window.insertSidebar(".app-shell", window.NAV_LINKS, window.TOC);
    } catch (e) {}
  }
})();
