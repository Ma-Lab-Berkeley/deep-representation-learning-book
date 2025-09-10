(function () {
  // Use global get_text from common.js

  function formatFooter(template, year) {
    return template.replace("{year}", year);
  }

  const AI_ASSISTANTS = [
    {
      name: "BookQA-7B-Instruct",
      affil: "",
      link: "https://huggingface.co/tianzhechu/BookQA-7B-Instruct",
    },
    {
      name: "BookQA-32B-Instruct",
      affil: "",
      link: "https://huggingface.co/tianzhechu/BookQA-32B-Instruct",
    },
  ];

  const AI_TOOLS = [
    {
      name: "GitHub Copilot",
      affil: "Microsoft",
    },
    {
      name: "Cursor AI",
      affil: "Cursor",
    },
    {
      name: "ChatGPT",
      affil: "OpenAI",
    },
  ];

  // Top bar and sidebar are inserted by common.js

  // Badges component removed

  function Card({ name, affil, link }) {
    const nameElement = link
      ? React.createElement(
          "a",
          {
            href: link,
            target: "_blank",
            rel: "noopener noreferrer",
            className: "name-link",
          },
          name
        )
      : React.createElement("div", { className: "name" }, name);

    const hasAffil = !!(affil && String(affil).trim());
    return React.createElement(
      "div",
      { className: "card" },
      nameElement,
      hasAffil ? React.createElement("p", { className: "affil" }, affil) : null
    );
  }

  function Main() {
    return React.createElement(
      "main",
      { className: "page" },
      React.createElement(
        "h1",
        null,
        (window.get_text && window.get_text("aiHelpers.title")) || ""
      ),
      (function () {
        var intro =
          (window.get_text && window.get_text("aiHelpers.intro")) || "";
        return (
          (window.get_text_block && window.get_text_block(intro, "intro")) ||
          null
        );
      })(),
      React.createElement(
        "section",
        { "aria-label": "BookQA Series" },
        React.createElement(
          "h2",
          { style: { margin: "16px 0 8px", fontSize: "18px" } },
          (window.get_text_inline &&
            window.get_text_inline("aiHelpers.sections.customizedChatbots")) ||
            ""
        ),
        React.createElement(
          "div",
          { className: "ai-helpers-grid" },
          AI_ASSISTANTS.map((p) =>
            React.createElement(Card, { key: p.name, ...p })
          )
        ),
        (window.get_text_block &&
          window.get_text_block("aiHelpers.techDetails", "tech-details")) ||
          null
      ),
      React.createElement(
        "div",
        { className: "foot" },
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
      { className: "layout-with-sidebar" },
      React.createElement(Main, null)
    );
  }

  ReactDOM.createRoot(document.getElementById("root")).render(
    React.createElement(App)
  );
})();
