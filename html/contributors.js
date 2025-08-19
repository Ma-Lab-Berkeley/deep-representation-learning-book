/* Extracted from contributors.html inline script */
(function(){
  const AUTHORS = [
    { name: 'Sam Buchanan', affil: 'Toyota Technological Institute at Chicago', badges: ['Author', 'Lead Editor'] },
    { name: 'Druv Pai', affil: 'University of California, Berkeley', badges: ['Author', 'Lead Editor', 'Website'] },
    { name: 'Peng Wang', affil: 'University of Macau', badges: ['Author', 'Lead Editor', 'Chinese Translation'] },
    { name: 'Yi Ma', affil: 'University of Hong Kong', badges: ['Senior Author', 'Lead Editor', 'Chinese Translation'] },
  ];

  const CONTRIBUTORS = [
    { name: 'Yaodong Yu', affil: 'University of Maryland, College Park', badges: ['Chapter 4'] },
    { name: 'Ziyang Wu', affil: 'University of California, Berkeley', badges: ['Website'] },
    { name: 'Tianzhe Chu', affil: 'University of Hong Kong', badges: ['AI Helper'] },
  ];

  // Top bar and sidebar are inserted by shared-ui.js

  function Badges({ items }) {
    if (!items || !items.length) return null;
    return React.createElement('div', { className: 'badges' }, items.map((b, i) => React.createElement('span', { className: 'badge', key: i }, b)));
  }

  function Card({ name, affil, badges }) {
    return (
      React.createElement('div', { className: 'card' },
        React.createElement('div', { className: 'name' }, name),
        React.createElement('p', { className: 'affil' }, affil),
        React.createElement(Badges, { items: badges })
      )
    );
  }

  function Main() {
    return (
      React.createElement('main', { className: 'page' },
        React.createElement('h1', null, 'Contributors'),
        React.createElement('p', { className: 'intro' }, 'Core authors and contributors of the book.'),
        React.createElement('section', { 'aria-label': 'Core Team', className: 'authors-grid' },
          React.createElement('h2', { style: { margin: '16px 0 8px', fontSize: '18px' } }, 'Core Editorial Team'),
          AUTHORS.map((p) => React.createElement(Card, { key: p.name, ...p }))
        ),
        React.createElement('section', { 'aria-label': 'Contributors', className: 'authors-grid' },
          React.createElement('h2', { style: { margin: '16px 0 8px', fontSize: '18px' } }, 'Contributors'),
          CONTRIBUTORS.map((p) => React.createElement(Card, { key: p.name, ...p }))
        ),
        React.createElement('div', { className: 'foot' }, '© ', new Date().getFullYear(), ' The Authors. All rights reserved.')
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
  if (window.insertTopBar) { try { window.insertTopBar(Object.assign({}, window.TOPBAR_OPTIONS || {}, { forceReplace: true })); } catch(e) {} }
  if (window.insertSidebar) { try { window.insertSidebar('.layout-with-sidebar', window.NAV_LINKS, window.TOC); } catch(e) {} }
})();


