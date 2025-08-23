/* Extracted from index.html inline script */
(function () {

  // Helper function to get localized text with fallback
  function getText(path, fallback) {
    try {
      var keys = path.split('.');
      var obj = window.BOOK_COMPONENTS;
      for (var i = 0; i < keys.length; i++) {
        obj = obj[keys[i]];
        if (!obj) return fallback;
      }
      return obj;
    } catch (e) {
      return fallback;
    }
  }

  function formatFooter(template, year) {
    return template.replace('{year}', year);
  }

  // Helper to convert markdown-style links to React elements
  function parseLinks(text) {
    var parts = [];
    var regex = /\[([^\]]+)\]\(([^)]+)\)/g;
    var lastIndex = 0;
    var match;
    
    while ((match = regex.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parts.push(text.slice(lastIndex, match.index));
      }
      parts.push(React.createElement('a', { href: match[2], target: '_blank', rel: 'noopener noreferrer' }, match[1]));
      lastIndex = regex.lastIndex;
    }
    
    if (lastIndex < text.length) {
      parts.push(text.slice(lastIndex));
    }
    
    return parts.length === 1 && typeof parts[0] === 'string' ? parts[0] : parts;
  }

  function Main() {
    return (
      React.createElement('main', { className: 'main' },
        React.createElement('section', { className: 'hero' },
          React.createElement('div', { className: 'hero-card' },
            React.createElement('h1', { className: 'hero-title' }, getText('landing.hero.title', 'Learning Deep Representations of Data Distributions')),
            React.createElement('div', { className: 'hero-authors' }, getText('landing.hero.authors', 'Sam Buchanan · Druv Pai · Peng Wang · Yi Ma')),
            React.createElement('p', { className: 'hero-sub' }, getText('landing.hero.subtitle', 'A modern fully open-source textbook exploring why and how deep neural networks learn compact and information-dense representations of high-dimensional real-world data.')),
            // React.createElement('div', { className: 'pub-info' },
            //   React.createElement('div', { className: 'pub-info-title' }, 'Publication Information'),
            //   React.createElement('p', null, 'Placeholder: publication details (publisher, edition, ISBN, publication date) will go here.')
            // ),
            React.createElement('div', { className: 'citation-info' },
              React.createElement('code', { style: { 'whiteSpace': 'pre-wrap' } }, String.raw`@book{ldrdd2025,
  title={Learning Deep Representations of Data Distributions},
  author={Buchanan, Sam and Pai, Druv and Wang, Peng and Ma, Yi},
  year={2025},
  publisher={Online}
}`)
            ),
            React.createElement('div', { className: 'cta-row' },
              React.createElement('a', { className: 'btn', href: 'Chx1.html' }, getText('landing.hero.buttons.readHtml', 'Read the Book (HTML)')),
              React.createElement('a', { className: 'btn', href: 'book-main.pdf' }, getText('landing.hero.buttons.readPdf', 'Read the Book (PDF)')),
              React.createElement('a', { className: 'btn secondary', href: 'https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book', target: '_blank', rel: 'noopener noreferrer' }, getText('landing.hero.buttons.github', 'GitHub Repository'))
            )
          ),
          React.createElement('div', { className: 'hero-figure' },
            React.createElement('a', { className: 'cover-ph', href: 'Chx1.html', title: getText('landing.hero.cover.title', 'Read the Book') },
              React.createElement('img', { className: 'cover-img', src: window.BOOK_COMPONENTS.coverImagePath, alt: getText('landing.hero.cover.alt', 'Book cover: Learning Deep Representations of Data Distributions'), loading: 'lazy' })
            ),
            React.createElement('div', { className: 'cover-version' }, getText('landing.hero.cover.version', 'Version 1.0\nReleased August 18, 2025'))
          )
        ),
        React.createElement('section', { className: 'sections' },
          React.createElement('div', { className: 'section-card' },
            React.createElement('h3', null, getText('landing.sections.about.title', 'About this Book')),
            (function() {
              var paragraphs = getText('landing.sections.about.paragraphs', [
                'In the current era of deep learning and especially "generative artificial intelligence", there is significant investment in training very large generative models. Thus far, such models have been "black boxes" that are difficult to understand in the sense that they have opaque internal mechanisms, leading to difficulties in interpretability, reliability, and control. Naturally, this lack of understanding has led to both hype and fear.',
                'This book is an attempt to "open the black box" and understand the mechanisms of large deep networks, through the perspective of representation learning, which is a major factor --- arguably the single most important one --- in the empirical power of deep learning models. A brief outline of this book is as follows. Chapter 1 will summarize the threads that underlie the whole text. Chapters 2, 3, 4, and 5 will explain the design principles of modern neural network architectures through optimization and information theory, reducing the process of architecture development (long having been described as a sort of "alchemy") to undergraduate-level linear algebra and calculus exercises once the underlying principles are introduced. Chapters 6 and 7 will discuss applications of these principles to solve problems in more paradigmatic ways, obtaining new methods and models which are efficient, interpretable, and controllable by design, and yet no less --- sometimes even more --- powerful than the black-box models they resemble. Chapter 8 will discuss potential future directions for deep learning, the role of representation learning, as well as some open problems.',
                'This book is intended for older undergraduate students, or initial graduate students, who have some background in linear algebra, probability, and machine learning. This book should be suitable as a first course in deep learning for mathematically-minded students, but it may help to have some initial surface-level knowledge of deep learning to better appreciate the perspectives and techniques discussed in the book.',
                'Due to the timeliness of the book, and the prevalence that deep learning may have in the coming years, we have decided to make the book completely open-source and welcome contributions from subject matter experts. The source code is available on [GitHub](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book). There are certainly many topics in deep representation learning that we have not covered in this book; if you are an expert and feel something is missing, you can [let us know](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book?tab=readme-ov-file#raising-an-issue) or [contribute it yourself](https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution). We will work to keep a similar standard of quality for new contributions, and recognize contributions in [the contributors page](contributors.html).'
              ]);
              return paragraphs.map(function(p, i) {
                return React.createElement('p', { key: i }, parseLinks(p));
              });
            })()
          ),
          React.createElement('div', { className: 'section-card' },
            React.createElement('h3', null, getText('landing.sections.acknowledgements.title', 'Acknowledgements')),
            (function() {
              var paragraphs = getText('landing.sections.acknowledgements.paragraphs', [
                'This book is primarily based on research results that have been developed within the past eight years. Thanks to generous funding from UC Berkeley (2018) and the University of Hong Kong (2023), Yi Ma was able to embark and focus on this new exciting research direction in the past eight years. Through these years, related to this research direction, Yi Ma and his research team at Berkeley have been supported by the following research grants:',
                'This book would have not been possible without the financial support for these research projects. The authors have drawn tremendous inspiration from research results by colleagues and students who have been involved in these projects.'
              ]);
              var grants = getText('landing.sections.acknowledgements.grants', [
                'The multi-university *THEORINET* project for the Foundations of Deep Learning, jointly funded by the Simons Foundation and the National Science Foundation (DMS grant #2031899)',
                'The *Closed-Loop Data Transcription via Minimaxing Rate Reduction* project funded by the Office of Naval Research (grant N00014-22-1-2102);',
                'The *Principled Approaches to Deep Learning for Low-dimensional Structures* project funded by the National Science Foundation (CISE grant #2402951).'
              ]);
              var elements = [];
              elements.push(React.createElement('p', { key: 'p1' }, paragraphs[0]));
              elements.push(React.createElement('ul', { key: 'grants' },
                grants.map(function(grant, i) {
                  // Convert *text* to <em>text</em>
                  var parts = grant.split('*');
                  var content = [];
                  for (var j = 0; j < parts.length; j++) {
                    if (j % 2 === 0) {
                      content.push(parts[j]);
                    } else {
                      content.push(React.createElement('em', { key: j }, parts[j]));
                    }
                  }
                  return React.createElement('li', { key: i }, content);
                })
              ));
              elements.push(React.createElement('p', { key: 'p2' }, paragraphs[1]));
              return elements;
            })()
          ),
          // React.createElement('div', { className: 'section-card' },
          //   React.createElement('h3', null, 'Citation'),
          //   React.createElement('p', null, 'Placeholder: citation information and BibTeX entry will be provided here.')
          // )
        ),
        React.createElement('div', { className: 'footer' }, formatFooter(getText('landing.footer', '© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. All rights reserved.'), new Date().getFullYear()))
      )
    );
  }

  function App() {
    return (
      React.createElement('div', { className: 'app-shell' },
        React.createElement(Main, null)
      )
    );
  }

  ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
  if (window.insertTopBar) { try { window.insertTopBar(Object.assign({}, window.TOPBAR_OPTIONS || {}, { forceReplace: true })); } catch (e) { } }
  if (window.insertSidebar) { try { window.insertSidebar('.app-shell', window.NAV_LINKS, window.TOC); } catch (e) { } }
})();

