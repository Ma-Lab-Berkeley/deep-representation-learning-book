/* Extracted from index.html inline script */
(function () {

  function Main() {
    return (
      React.createElement('main', { className: 'main' },
        React.createElement('section', { className: 'hero' },
          React.createElement('div', { className: 'hero-card' },
            React.createElement('h1', { className: 'hero-title' }, 'Learning Deep Representations of Data Distributions'),
            React.createElement('div', { className: 'hero-authors' }, 'Sam Buchanan · Druv Pai · Peng Wang · Yi Ma'),
            React.createElement('p', { className: 'hero-sub' }, 'A modern fully open-source textbook exploring why and how deep neural networks learn compact and information-dense representations of high-dimensional real-world data.'),
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
              React.createElement('a', { className: 'btn', href: 'Chx1.html' }, 'Read the Book (HTML)'),
              React.createElement('a', { className: 'btn', href: 'book-main.pdf' }, 'Read the Book (PDF)'),
              React.createElement('a', { className: 'btn secondary', href: 'https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book', target: '_blank', rel: 'noopener noreferrer' }, 'GitHub Repository')
            )
          ),
          React.createElement('div', { className: 'hero-figure' },
            React.createElement('a', { className: 'cover-ph', href: 'Chx1.html', title: 'Read the Book' },
              React.createElement('img', { className: 'cover-img', src: 'book-cover.png', alt: 'Book cover: Learning Deep Representations of Data Distributions', loading: 'lazy' })
            ),
            React.createElement('div', { className: 'cover-version' }, 'Version 1.0\nReleased August 18, 2025')
          )
        ),
        React.createElement('section', { className: 'sections' },
          React.createElement('div', { className: 'section-card' },
            React.createElement('h3', null, 'About this Book'),
            React.createElement('p', null, 'In the current era of deep learning and especially "generative artificial intelligence", there is significant investment in training very large generative models. Thus far, such models have been "black boxes" that are difficult to understand in the sense that they have opaque internal mechanisms, leading to difficulties in interpretability, reliability, and control. Naturally, this lack of understanding has led to both hype and fear.'),
            React.createElement('p', null, 'This book is an attempt to "open the black box" and understand the mechanisms of large deep networks, through the perspective of representation learning, which is a major factor --- arguably the single most important one --- in the empirical power of deep learning models. A brief outline of this book is as follows. Chapter 1 will summarize the threads that underlie the whole text. Chapters 2, 3, 4, and 5 will explain the design principles of modern neural network architectures through optimization and information theory, reducing the process of architecture development (long having been described as a sort of "alchemy") to undergraduate-level linear algebra and calculus exercises once the underlying principles are introduced. Chapters 6 and 7 will discuss applications of these principles to solve problems in more paradigmatic ways, obtaining new methods and models which are efficient, interpretable, and controllable by design, and yet no less --- sometimes even more --- powerful than the black-box models they resemble. Chapter 8 will discuss potential future directions for deep learning, the role of representation learning, as well as some open problems.'),
            React.createElement('p', null, 'This book is intended for older undergraduate students, or initial graduate students, who have some background in linear algebra, probability, and machine learning. This book should be suitable as a first course in deep learning for mathematically-minded students, but it may help to have some initial surface-level knowledge of deep learning to better appreciate the perspectives and techniques discussed in the book.'),
            React.createElement('p', null, 'Due to the timeliness of the book, and the prevalence that deep learning may have in the coming years, we have decided to make the book completely open-source and welcome contributions from subject matter experts. The source code is available on ', React.createElement('a', { href: 'https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book', target: '_blank', rel: 'noopener noreferrer' }, 'GitHub'), '. There are certainly many topics in deep representation learning that we have not covered in this book; if you are an expert and feel something is missing, you can ', React.createElement('a', { href: 'https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book?tab=readme-ov-file#raising-an-issue', target: '_blank', rel: 'noopener noreferrer' }, 'let us know'), ' or ', React.createElement('a', { href: 'https://github.com/Ma-Lab-Berkeley/deep-representation-learning-book#making-a-contribution', target: '_blank', rel: 'noopener noreferrer' }, 'contribute it yourself'), '. We will work to keep a similar standard of quality for new contributions, and recognize contributions in ', React.createElement('a', { href: 'contributors.html', target: '_blank', rel: 'noopener noreferrer' }, 'the contributors page'), '.')
          ),
          React.createElement('div', { className: 'section-card' },
            React.createElement('h3', null, 'Acknowledgements'),
            React.createElement('p', null, 'This book is primarily based on research results that have been developed within the past eight years. Thanks to generous funding from UC Berkeley (2018) and the University of Hong Kong (2023), Yi Ma was able to embark and focus on this new exciting research direction in the past eight years. Through these years, related to this research direction, Yi Ma and his research team at Berkeley have been supported by the following research grants:'),
            React.createElement('ul', null,
              React.createElement('li', null, "The multi-university ", React.createElement('em', null, 'THEORINET'), " project for the Foundations of Deep Learning, jointly funded by the Simons Foundation and the National Science Foundation (DMS grant #2031899)"),
              React.createElement('li', null, "The ", React.createElement('em', null, 'Closed-Loop Data Transcription via Minimaxing Rate Reduction'), " project funded by the Office of Naval Research (grant N00014-22-1-2102);"),
              React.createElement('li', null, "The ", React.createElement('em', null, 'Principled Approaches to Deep Learning for Low-dimensional Structures'), " project funded by the National Science Foundation (CISE grant #2402951)."),
            ),
            React.createElement('p', null, 'This book would have not been possible without the financial support for these research projects. The authors have drawn tremendous inspiration from research results by colleagues and students who have been involved in these projects.'),
          ),
          // React.createElement('div', { className: 'section-card' },
          //   React.createElement('h3', null, 'Citation'),
          //   React.createElement('p', null, 'Placeholder: citation information and BibTeX entry will be provided here.')
          // )
        ),
        React.createElement('div', { className: 'footer' }, '© ', new Date().getFullYear(), ' Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. All rights reserved.')
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


