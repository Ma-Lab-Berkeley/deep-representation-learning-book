(function(){
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

  // Helper function to render text with links and formatting
  function renderTextWithLinks(text) {
    // Replace BookQA Series with bold text
    let parts = text.split('BookQA Series');
    if (parts.length > 1) {
      const elements = [];
      for (let i = 0; i < parts.length; i++) {
        if (i > 0) {
          elements.push(React.createElement('strong', { 
            style: { fontWeight: '700' }
          }, 'BookQA Series'));
        }
        if (parts[i]) {
          elements.push(parts[i]);
        }
      }
      text = elements;
    }
    
    // Replace EntiGraph with a link
    if (Array.isArray(text)) {
      // If text is already an array of elements, process each element
      const processedElements = [];
      for (let element of text) {
        if (typeof element === 'string') {
          const entiParts = element.split('EntiGraph');
          if (entiParts.length > 1) {
            for (let j = 0; j < entiParts.length; j++) {
              if (j > 0) {
                processedElements.push(React.createElement('a', { 
                  href: 'https://arxiv.org/pdf/2409.07431', 
                  target: '_blank', 
                  rel: 'noopener noreferrer',
                  style: { color: 'var(--accent)', textDecoration: 'none' }
                }, 'EntiGraph'));
              }
              if (entiParts[j]) {
                processedElements.push(entiParts[j]);
              }
            }
          } else {
            processedElements.push(element);
          }
        } else {
          processedElements.push(element);
        }
      }
      text = processedElements;
    } else {
      // If text is a string, process it directly
      const entiParts = text.split('EntiGraph');
      if (entiParts.length > 1) {
        const elements = [];
        for (let i = 0; i < entiParts.length; i++) {
          if (i > 0) {
            elements.push(React.createElement('a', { 
              href: 'https://arxiv.org/pdf/2409.07431', 
              target: '_blank', 
              rel: 'noopener noreferrer',
              style: { color: 'var(--accent)', textDecoration: 'none' }
            }, 'EntiGraph'));
          }
          if (entiParts[i]) {
            elements.push(entiParts[i]);
          }
        }
        text = elements;
      }
    }
    
    // Replace Qwen2.5-7B/32B-Instruct with a link
    if (Array.isArray(text)) {
      // If text is already an array of elements, process each element
      const processedElements = [];
      for (let element of text) {
        if (typeof element === 'string') {
          const qwenParts = element.split('Qwen2.5-7B/32B-Instruct');
          if (qwenParts.length > 1) {
            for (let j = 0; j < qwenParts.length; j++) {
              if (j > 0) {
                processedElements.push(React.createElement('a', { 
                  href: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', 
                  target: '_blank', 
                  rel: 'noopener noreferrer',
                  style: { color: 'var(--accent)', textDecoration: 'none' }
                }, 'Qwen2.5-7B/32B-Instruct'));
              }
              if (qwenParts[j]) {
                processedElements.push(qwenParts[j]);
              }
            }
          } else {
            processedElements.push(element);
          }
        } else {
          processedElements.push(element);
        }
      }
      return processedElements;
    } else {
      // If text is a string, process it directly
      const qwenParts = text.split('Qwen2.5-7B/32B-Instruct');
      if (qwenParts.length === 1) {
        return text;
      }
      
      const elements = [];
      for (let i = 0; i < qwenParts.length; i++) {
        if (i > 0) {
          elements.push(React.createElement('a', { 
            href: 'https://huggingface.co/Qwen/Qwen2.5-7B-Instruct', 
            target: '_blank', 
            rel: 'noopener noreferrer',
            style: { color: 'var(--accent)', textDecoration: 'none' }
          }, 'Qwen2.5-7B/32B-Instruct'));
        }
        if (qwenParts[i]) {
          elements.push(qwenParts[i]);
        }
      }
      return elements;
    }
  }

  // Helper to get localized badge text
  function getBadgeText(badgeKey) {
    var badges = getText('aiHelpers.badges', {});
    var mapping = {
      'Customized Chatbot': badges.customizedChatbot || 'Customized Chatbot',
    };
    return mapping[badgeKey] || badgeKey;
  }

  // Helper to map badge arrays to localized text
  function mapBadges(badges) {
    return badges.map(function(badge) { return getBadgeText(badge); });
  }

  const AI_ASSISTANTS = [
    { name: 'BookQA-7B-Instruct', affil: '', badges: mapBadges(['Language Model']), link: 'https://huggingface.co/tianzhechu/BookQA-7B-Instruct' },
    { name: 'BookQA-32B-Instruct', affil: '', badges: mapBadges(['Language Model']), link: 'https://huggingface.co/tianzhechu/BookQA-32B-Instruct' },
  ];

  const AI_TOOLS = [
    { name: 'GitHub Copilot', affil: 'Microsoft', badges: mapBadges(['Code Generation']) },
    { name: 'Cursor AI', affil: 'Cursor', badges: mapBadges(['Code Generation', 'Documentation']) },
    { name: 'ChatGPT', affil: 'OpenAI', badges: mapBadges(['AI Assistant', 'Translation']) },
  ];

  // Top bar and sidebar are inserted by common.js

  function Badges({ items }) {
    if (!items || !items.length) return null;
    return React.createElement('div', { className: 'badges' }, items.map((b, i) => React.createElement('span', { className: 'badge', key: i }, b)));
  }

  function Card({ name, affil, badges, link }) {
    const nameElement = link 
      ? React.createElement('a', { href: link, target: '_blank', rel: 'noopener noreferrer', className: 'name-link' }, name)
      : React.createElement('div', { className: 'name' }, name);
    
    return (
      React.createElement('div', { className: 'card' },
        nameElement,
        React.createElement('p', { className: 'affil' }, affil),
        React.createElement(Badges, { items: badges })
      )
    );
  }

  function Main() {
    return (
      React.createElement('main', { className: 'page' },
        React.createElement('h1', null, getText('aiHelpers.title', 'AI Helpers')),
        React.createElement('p', { className: 'intro' }, getText('aiHelpers.intro', 'Customized chatbots that have helped in the creation of this book.')),
        React.createElement('section', { 'aria-label': 'BookQA Series' },
          React.createElement('h2', { style: { margin: '16px 0 8px', fontSize: '18px' } }, getText('aiHelpers.sections.customizedChatbots', 'BookQA Series')),
          React.createElement('div', { className: 'ai-helpers-grid' },
            AI_ASSISTANTS.map((p) => React.createElement(Card, { key: p.name, ...p }))
          ),
          React.createElement('p', { className: 'tech-details' }, renderTextWithLinks(getText('aiHelpers.techDetails', 'BookQA Series is designed to help readers understand a book’s content. It can answer questions about the material and give clear explanations of the key concepts and theories. To build these models, we first use EntiGraph to generate a rich set of book-related data by linking sampled entities from the text. We then continually pre-train Qwen2.5-7B/32B-Instruct on this data using auto-regressive training. \
            We also incorporate instruction-following data during training such that the model can learn new knowledge from the book without forgetting basic chatting skills.')))
        ),
        // React.createElement('section', { 'aria-label': 'AI Tools', className: 'ai-helpers-grid' },
        //   React.createElement('h2', { style: { margin: '16px 0 8px', fontSize: '18px' } }, getText('aiHelpers.sections.aiTools', 'AI Tools')),
        //   AI_TOOLS.map((p) => React.createElement(Card, { key: p.name, ...p }))
        // ),
        React.createElement('div', { className: 'foot' }, formatFooter(getText('aiHelpers.footer', '© {year} Sam Buchanan, Druv Pai, Peng Wang, and Yi Ma. All rights reserved.'), new Date().getFullYear()))
      )
    );
  }

  function App() {
    return (
      React.createElement('div', { className: 'layout-with-sidebar' },
        React.createElement(Main, null)
      )
    );
  }

  ReactDOM.createRoot(document.getElementById('root')).render(React.createElement(App));
  // if (window.insertTopBar) { try { window.insertTopBar(Object.assign({}, window.TOPBAR_OPTIONS || {}, { forceReplace: true })); } catch(e) {} }
  // if (window.insertSidebar) { try { window.insertSidebar('.layout-with-sidebar', window.NAV_LINKS, window.TOC); } catch(e) {} }
})();
