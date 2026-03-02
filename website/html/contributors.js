(function() {
    // Use global get_text from common.js

    // Helper to fetch per-person localized description by id
    function getPersonDescription(personId) {
        if (!personId) return "";
        return (
            (window.get_text &&
                window.get_text("contributors.people." + personId + ".desc")) ||
            ""
        );
    }

    // Badges removed. All labeling handled via localized descriptions in common_components.

    // People data per category.
    // Optional fields: url (personal site), id (for localized description lookup) 
    const AUTHORS = [
        {
            id: "sam-buchanan",
            name: "Sam Buchanan",
            url: "https://sdbuchanan.com/",
            affil: "University of California, Berkeley",
        },
        {
            id: "druv-pai",
            name: "Druv Pai",
            url: "https://druvpai.github.io/",
            affil: "University of California, Berkeley",
        },
        {
            id: "peng-wang",
            name: "Peng Wang",
            url: "https://peng8wang.github.io/",
            affil: "University of Macau",
        },
        {
            id: "yi-ma",
            name: "Yi Ma",
            url: "https://people.eecs.berkeley.edu/~yima/",
            affil: "University of Hong Kong",
        },
    ];

    // const EDITORS = [
    //   // Editors list intentionally left empty for now; will be populated soon.
    // ];

    const CONTRIBUTORS = [
        {
            id: "stephen-butterfill",
            name: "Stephen Butterfill",
            url: "https://www.butterfill.com/",
            affil: "University of Warwick",
        },
        {
            id: "jan-cavel",
            name: "Jan Cavel",
            url: "https://piatra.institute/",
            affil: "Piatra Institute",
        },
        {
            id: "tianzhe-chu",
            name: "Tianzhe Chu",
            url: "https://tianzhechu.com/",
            affil: "University of Hong Kong",
        },
        {
            id: "shenghua-gao",
            name: "Shenghua Gao",
            url: "https://datascience.hku.hk/people/shenghua-gao/",
            affil: "ShanghaiTech University",
        },
        {
            id: "bingbing-huang",
            name: "Bingbing Huang",
            url: "https://scholar.google.com/citations?user=5da8iKwAAAAJ&hl=en-US",
            affil: "University of Hong Kong",
        },
        {
            id: "kerui-min",
            name: "Kerui Min",
            url: "https://www.linkedin.com/in/kerui-min-b974b52a/",
            affil: "MetaSOTA",
        },
        {
            id: "kevin-murphy",
            name: "Kevin Murphy",
            url: "https://www.linkedin.com/in/kevin-murphy-20684115/",
            affil: "Google DeepMind",
        },
        {
            id: "qing-qu",
            name: "Qing Qu",
            url: "https://deepthink-umich.github.io/index.html",
            affil: "University of Michigan, Ann Arbor",
        },
        {
            id: "shengbang-tong",
            name: "Shengbang Tong",
            url: "New York University",
            affil: "https://tsb0601.github.io/",
        },
        {
            id: "jeroen-van-goey",
            name: "Jeroen Van Goey",
            url: "https://www.linkedin.com/in/jeroenvangoey/",
            affil: "InstaDeep",
        },
        {
            id: "chengyu-wang",
            name: "Chengyu Wang",
            affil: "",
        },
        {
            id: "ziyang-wu",
            name: "Ziyang Wu",
            url: "https://robinwu218.github.io/",
            affil: "University of California, Berkeley",
        },
        {
            id: "jingfeng-yang",
            name: "Jingfeng Yang",
            url: "https://jingfeng0705.github.io/",
            affil: "University of California, Berkeley",
        },
        {
            id: "daniel-yeh",
            name: "Chun-Hsiao Yeh",
            url: "https://danielchyeh.github.io/",
            affil: "University of California, Berkeley",
        },
        {
            id: "brent-yi",
            name: "Brent Yi",
            url: "https://brentyi.github.io/",
            affil: "University of California, Berkeley",
        },
        {
            id: "yaodong-yu",
            name: "Yaodong Yu",
            url: "https://yaodongyu.github.io/",
            affil: "University of Maryland, College Park",
        },
        {
            id: "zibo-zhao",
            name: "Zibo Zhao",
            url: "https://maikouuu.github.io/",
            affil: "Hunyuan, Tencent",
        },
    ];

    // Top bar and sidebar are inserted by common.js

    function Card({ id, name, url, affil }) {
        var description = getPersonDescription(id);
        return React.createElement(
            "div",
            { className: "card" },
            React.createElement(
                "div",
                { className: "name" },
                url
                    ? React.createElement(
                        "a",
                        { href: url, target: "_blank", rel: "noopener noreferrer" },
                        name
                    )
                    : name
            ),
            React.createElement("p", { className: "affil" }, affil),
            description
                ? (window.get_text_block &&
                    window.get_text_block(description, "desc")) ||
                null
                : null
        );
    }

    function Main() {
        return React.createElement(
            "main",
            { className: "page" },
            React.createElement(
                "h1",
                null,
                (window.get_text_inline &&
                    window.get_text_inline("contributors.title")) ||
                ""
            ),
            (function() {
                var intro =
                    (window.get_text && window.get_text("contributors.intro")) || "";
                return React.createElement(
                    "p",
                    { className: "intro" },
                    (window.get_text_inline && window.get_text_inline(intro)) || ""
                );
            })(),
            React.createElement(
                "section",
                { "aria-label": "Authors", className: "card-grid" },
                React.createElement(
                    "h2",
                    { style: { margin: "16px 0 8px", fontSize: "18px" } },
                    (window.get_text_inline &&
                        window.get_text_inline("contributors.sections.authors")) ||
                    ""
                ),
                AUTHORS.map((p) => React.createElement(Card, { key: p.name, ...p }))
            ),
            // React.createElement('section', { 'aria-label': 'Editors', className: 'card-grid' },
            //   React.createElement('h2', { style: { margin: '16px 0 8px', fontSize: '18px' } }, (window.get_text && window.get_text('contributors.sections.editors')) || ''),
            //   EDITORS.map((p) => React.createElement(Card, { key: p.name, ...p }))
            // ),
            React.createElement(
                "section",
                { "aria-label": "Contributors", className: "card-grid" },
                React.createElement(
                    "h2",
                    { style: { margin: "16px 0 8px", fontSize: "18px" } },
                    (window.get_text_inline &&
                        window.get_text_inline(
                            "contributors.sections.contributors"
                        )) ||
                    ""
                ),
                CONTRIBUTORS.map((p) =>
                    React.createElement(Card, { key: p.name, ...p })
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
    // if (window.insertTopBar) { try { window.insertTopBar(Object.assign({}, window.TOPBAR_OPTIONS || {}, { forceReplace: true })); } catch(e) {} }
    // if (window.insertSidebar) { try { window.insertSidebar('.layout-with-sidebar', window.NAV_LINKS, window.TOC); } catch(e) {} }
})();
