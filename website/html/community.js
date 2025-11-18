(function () {
  // Use global get_text from common.js

  function Section({ idKey }) {
    var title =
      (window.get_text && window.get_text("community.sections." + idKey)) || "";
    var content =
      (window.get_text && window.get_text("community.content." + idKey)) ||
      (window.get_text && window.get_text("community.pending")) ||
      "";
    return React.createElement(
      "section",
      { className: "text-section", "aria-label": title },
      React.createElement("h2", null, title),
      (window.get_text_block && window.get_text_block(content)) || null
    );
  }

  function Main() {
    return React.createElement(
      "main",
      { className: "page" },
      React.createElement(
        "h1",
        null,
        (window.get_text && window.get_text("community.title")) || ""
      ),
      (function () {
        var intro =
          (window.get_text && window.get_text("community.intro")) || "";
        return (
          (window.get_text_block && window.get_text_block(intro, "intro")) ||
          null
        );
      })(),
      React.createElement(Section, { idKey: "translations" }),
      React.createElement(Section, { idKey: "courses" }),
      React.createElement(Section, { idKey: "tutorials" })
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
