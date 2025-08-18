/* Shared UI inserter: top bar and optional sidebars */
(function(){
  // Ensure Inter font is available on all pages (chapters don't include it by default)
  try {
    var hasInter = Array.prototype.some.call(document.querySelectorAll('link[rel="stylesheet"]'), function(l){
      var href = l.getAttribute('href') || '';
      return href.indexOf('fonts.googleapis.com') !== -1 && href.indexOf('Inter') !== -1;
    });
    if (!hasInter) {
      var gf = document.createElement('link');
      gf.rel = 'stylesheet';
      gf.href = 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap';
      var head = document.head || document.getElementsByTagName('head')[0];
      if (head) head.appendChild(gf);
    }
  } catch (_) {}
  var DEFAULT_NAV_LINKS = [
    { label: 'Contributors', href: 'contributors.html' },
    { label: 'How to Contribute?', href: 'https://github.com/Ma-Lab-Berkeley/ldrdd-book#making-a-contribution', external: true }
  ];
  var DEFAULT_TOC = [
    { label: 'Preface', href: 'Chx1.html' },
    { label: 'Chapter 1', href: 'Ch1.html' },
    { label: 'Chapter 2', href: 'Ch2.html' },
    { label: 'Chapter 3', href: 'Ch3.html' },
    { label: 'Chapter 4', href: 'Ch4.html' },
    { label: 'Chapter 5', href: 'Ch5.html' },
    { label: 'Chapter 6', href: 'Ch6.html' },
    { label: 'Chapter 7', href: 'Ch7.html' },
    { label: 'Chapter 8', href: 'Ch8.html' },
    { label: 'Appendix A', href: 'A1.html' },
    { label: 'Appendix B', href: 'A2.html' },
  ];

  function h(tag, props){
    var SVG_NS = 'http://www.w3.org/2000/svg';
    var svgTags = { svg:1, path:1, defs:1, linearGradient:1, stop:1 }; // minimal set we use
    var isSvg = Object.prototype.hasOwnProperty.call(svgTags, tag);
    var el = isSvg ? document.createElementNS(SVG_NS, tag) : document.createElement(tag);
    if (props) {
      Object.keys(props).forEach(function(k){
        if (k === 'className') el.className = props[k];
        else if (k === 'text') el.textContent = props[k];
        else if (k === 'html') el.innerHTML = props[k];
        else if (isSvg && k === 'stopColor') el.setAttribute('stop-color', props[k]);
        else el.setAttribute(k, props[k]);
      });
    }
    for (var i=2;i<arguments.length;i++){
      var c = arguments[i];
      if (c == null) continue;
      if (Array.isArray(c)) c.forEach(function(n){ if (n) el.appendChild(n); });
      else el.appendChild(c);
    }
    return el;
  }

  function renderTopBar(options){
    var existing = document.getElementById('book-topbar');
    var shouldReplace = !existing || (options && options.forceReplace === true) || (existing && !existing.getAttribute('data-shared-ui'));
    if (existing && shouldReplace) {
      try { existing.remove(); } catch (_) { /* ignore */ }
    } else if (existing && !shouldReplace) {
      return; // keep existing shared topbar
    }
    var title = (options && options.title) || 'Learning Deep Representations of Data Distributions';
    var langLabel = (options && options.langLabel) || 'CN';
    var brandHref = (options && options.brandHref) || 'index.html';

    // Logo SVG
    var logo = h('span', { className: 'logo' },
      h('svg', { xmlns:'http://www.w3.org/2000/svg', width:'18', height:'18', viewBox:'0 0 24 24', fill:'none' },
        h('path', { d:'M3 12c0-1.1.9-2 2-2h6V4c0-1.1.9-2 2-2h0c1.1 0 2 .9 2 2v6h6c1.1 0 2 .9 2 2h0c0 1.1-.9 2-2 2h-6v6c0 1.1-.9 2-2 2h0c-1.1 0-2-.9-2-2v-6H5c-1.1 0-2-.9-2-2Z', fill:'url(#g1)'}),
        h('defs', null,
          h('linearGradient', { id:'g1', x1:'0', y1:'0', x2:'24', y2:'24' },
            h('stop', { offset:'0%', stopColor:'#7aa2ff' }),
            h('stop', { offset:'100%', stopColor:'#8b78ff' })
          )
        )
      )
    );

    // Search shell only; book.js wires events
    var search = h('div', { className:'search' }, h('input', { className:'search-input', type:'search', placeholder:'Search pages…', 'aria-label':'Search' }), h('div', { className:'search-results' }));

    var ghIcon = h('svg', { xmlns:'http://www.w3.org/2000/svg', width:'16', height:'16', viewBox:'0 0 16 16', fill:'currentColor', role:'img', 'aria-label':'GitHub' },
      h('path', { d:'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.2 1.87.86 2.33.66.07-.52.28-.86.51-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z' })
    );

    // Language dropdown
    var langSelect = h('div', { className: 'lang-select' },
      h('button', { className: 'lang-toggle', type:'button', 'aria-haspopup': 'listbox', 'aria-expanded': 'false', title: 'Select language' },
        h('span', { className: 'lang-label', text: langLabel })
      ),
      h('div', { className: 'lang-menu', role: 'listbox' },
        h('button', { className: 'lang-item', role: 'option', 'data-lang': 'en', type:'button', text: 'English' }),
        h('button', { className: 'lang-item', role: 'option', 'data-lang': 'zh', type:'button', text: '中文' })
      )
    );

    var chatToggle = h('button', { className: 'chat-toggle', type: 'button', title: 'Ask AI about this page' },
      h('span', { className: 'chat-toggle-icon', html: '&#128172;' }),
      h('span', { className: 'chat-toggle-label', text: 'Ask AI' })
    );

    var bar = h('div', { className:'book-topbar', id:'book-topbar', 'data-shared-ui':'1' },
      h('a', { className:'brand brand-link', href:brandHref }, logo, h('div', { className:'title', text:title })),
      h('div', { className:'topbar-right' },
        search,
        langSelect,
        chatToggle,
        h('a', { className:'gh-link', href:'https://github.com/Ma-Lab-Berkeley/ldrdd-book', target:'_blank', rel:'noopener noreferrer' }, ghIcon, h('span', { text:'GitHub' }))
      )
    );
    document.body.insertBefore(bar, document.body.firstChild);

    // Wire up search behavior (unified for all pages)
    try {
      var input = bar.querySelector('.search-input');
      var box = bar.querySelector('.search-results');
      if (input && box) {
        var index = null; var items = []; var active = -1; var open = false;
        fetch('search-index.json').then(function(r){return r.ok?r.json():null}).then(function(j){index=j}).catch(function(){});
        function render(){
          box.innerHTML='';
          if(!open || !items.length){ box.style.display='none'; return; }
          items.forEach(function(it, i){
            var div = document.createElement('div'); div.className='search-item'+(i===active?' active':'');
            if(it.kind==='content'){
              var t=document.createElement('span'); t.className='search-item-title'; t.textContent=it.title;
              var s=document.createElement('span'); s.className='search-secondary'; s.textContent=(it.page||'')+' — '+(it.snippet||'');
              div.appendChild(t); div.appendChild(s);
            } else { div.textContent = it.label; }
            div.onmousedown=function(e){e.preventDefault()};
            div.onclick=function(){ if(it.external){ window.open(it.href,'_blank','noopener,noreferrer'); } else { window.location.href = it.href; } };
            box.appendChild(div);
          });
          box.style.display='block';
        }
        input.addEventListener('input', function(){
          var q=(input.value||'').trim().toLowerCase(); active=-1; open=true;
          if(!q){ items=[]; render(); return; }
          var content=[]; if(index&&index.entries){
            for(var i=0;i<index.entries.length && content.length<10;i++){ var e=index.entries[i];
              var t=(e.title||'').toLowerCase(), s=(e.snippet||'').toLowerCase(), p=(e.page||'').toLowerCase();
              if(t.includes(q)||s.includes(q)||p.includes(q)){ content.push(Object.assign({kind:'content'}, e)); }
            }
          }
          items = content; render();
        });
        input.addEventListener('focus', function(){ open=true; render(); });
        input.addEventListener('blur', function(){ setTimeout(function(){ open=false; render(); }, 120); });
        input.addEventListener('keydown', function(e){ if(!items.length) return; if(e.key==='ArrowDown'){ e.preventDefault(); active=(active+1)%items.length; render(); } else if(e.key==='ArrowUp'){ e.preventDefault(); active=(active-1+items.length)%items.length; render(); } else if(e.key==='Enter'){ var target=items[Math.max(0,active)]||items[0]; if(!target) return; if(target.external){ window.open(target.href,'_blank','noopener,noreferrer'); } else { window.location.href = target.href; } } });
      }
    } catch (e) {}

    // Wire chat toggle and ensure chat panel exists
    try {
      function ensureChatPanel(){
        if (document.getElementById('ai-chat-panel')) return;
        // Shell
        var panel = h('div', { id: 'ai-chat-panel', className: 'ai-chat-panel', role: 'dialog', 'aria-modal': 'false', 'aria-labelledby': 'ai-chat-title' },
          h('div', { className: 'ai-chat-header' },
            h('div', { id: 'ai-chat-title', className: 'ai-chat-title', text: 'Ask AI' }),
            h('div', { className: 'ai-chat-actions' },
              h('button', { className: 'ai-chat-clear', type: 'button', title: 'Clear conversation', text: 'Clear' }),
              h('button', { className: 'ai-chat-close', type: 'button', title: 'Close', text: 'Close' })
            )
          ),
          h('div', { className: 'ai-chat-context' },
            h('label', { className: 'ai-chat-ctx-row' },
              h('input', { type: 'checkbox', className: 'ai-chat-include-selection', checked: 'checked' }),
              h('span', { text: 'Include current text selection' })
            ),
            h('div', { className: 'ai-chat-selection-preview' },
              h('div', { className: 'ai-chat-selection-empty', text: 'Select text in the page to include it as context.' }),
              h('div', { className: 'ai-chat-selection-text' })
            )
          ),
          h('div', { className: 'ai-chat-messages', id: 'ai-chat-messages' }),
          h('form', { className: 'ai-chat-compose', id: 'ai-chat-form' },
            h('textarea', { className: 'ai-chat-input', id: 'ai-chat-input', rows: '3', placeholder: 'Ask a question about this page…' }),
            h('div', { className: 'ai-chat-sendrow' },
              h('button', { className: 'ai-chat-send', id: 'ai-chat-send', type: 'submit', text: 'Send' })
            )
          )
        );
        document.body.appendChild(panel);

        // Behavior
        var closeBtn = panel.querySelector('.ai-chat-close');
        if (closeBtn) closeBtn.addEventListener('click', function(){ document.body.classList.remove('ai-chat-open'); });
        var clearBtn = panel.querySelector('.ai-chat-clear');
        if (clearBtn) clearBtn.addEventListener('click', function(){
          var msgs = panel.querySelector('#ai-chat-messages');
          if (msgs) msgs.innerHTML = '';
          chatState.messages = [];
        });
        var form = panel.querySelector('#ai-chat-form');
        if (form) form.addEventListener('submit', function(e){ e.preventDefault(); sendChatMessage(); });
        var ta = panel.querySelector('#ai-chat-input');
        if (ta) ta.addEventListener('keydown', function(e){
          if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            form && form.dispatchEvent(new Event('submit', { cancelable: true }));
          }
        });
      }

      var chatState = { messages: [], currentSelection: '', sending: false };

      function getTrimmedSelection(maxLen){
        maxLen = maxLen || 1200;
        var sel = window.getSelection && window.getSelection();
        if (!sel || sel.isCollapsed) return '';
        // Ignore selections that are inside the chat panel (preview, input, etc.)
        try {
          var panel = document.getElementById('ai-chat-panel');
          if (panel && sel.rangeCount > 0) {
            var a = sel.anchorNode, f = sel.focusNode;
            var inside = (a && panel.contains(a)) || (f && panel.contains(f));
            if (inside) return '';
          }
        } catch (_) {}
        var text = (sel.toString() || '').trim();
        // Normalize whitespace to keep inline math readable
        // 1) Convert non-breaking spaces to regular spaces
        text = text.replace(/\u00A0/g, ' ');
        // 2) Collapse any sequence of whitespace (spaces, tabs, newlines) into a single space
        text = text.replace(/\s+/g, ' ');
        if (!text) return '';
        if (text.length > maxLen) text = text.slice(0, maxLen) + '\u2026';
        return text;
      }

      function updateSelectionPreview(){
        var panel = document.getElementById('ai-chat-panel'); if (!panel) return;
        var txt = chatState.currentSelection || '';
        var empty = panel.querySelector('.ai-chat-selection-empty');
        var box = panel.querySelector('.ai-chat-selection-text');
        if (!box || !empty) return;
        if (txt) {
          empty.style.display = 'none';
          box.textContent = txt;
          box.style.display = 'block';
        } else {
          box.style.display = 'none';
          empty.style.display = 'block';
        }
      }

      function appendMessage(role, content){
        var panel = document.getElementById('ai-chat-panel'); if (!panel) return;
        var list = panel.querySelector('#ai-chat-messages'); if (!list) return;
        var item = document.createElement('div');
        item.className = 'ai-chat-msg ' + (role === 'user' ? 'from-user' : 'from-assistant');
        var bubble = document.createElement('div');
        bubble.className = 'ai-chat-bubble';
        bubble.textContent = content;
        item.appendChild(bubble);
        list.appendChild(item);
        list.scrollTop = list.scrollHeight + 999;
      }

      function appendTypingIndicator(){
        var panel = document.getElementById('ai-chat-panel'); if (!panel) return null;
        var list = panel.querySelector('#ai-chat-messages'); if (!list) return null;
        var item = document.createElement('div');
        item.className = 'ai-chat-msg from-assistant typing';
        var bubble = document.createElement('div');
        bubble.className = 'ai-chat-bubble';
        var typing = document.createElement('div');
        typing.className = 'ai-typing';
        for (var i = 0; i < 3; i++) {
          var dot = document.createElement('span');
          dot.className = 'dot';
          typing.appendChild(dot);
        }
        bubble.appendChild(typing);
        item.appendChild(bubble);
        list.appendChild(item);
        list.scrollTop = list.scrollHeight + 999;
        return item;
      }

      function removeTypingIndicator(el){
        try { if (el && el.parentNode) el.parentNode.removeChild(el); } catch (_) {}
      }

      function setSending(isSending){
        chatState.sending = !!isSending;
        var btn = document.getElementById('ai-chat-send');
        var input = document.getElementById('ai-chat-input');
        if (btn) btn.disabled = !!isSending;
        if (input) input.disabled = !!isSending;
      }

      function buildPayload(userText){
        var includeSel = false;
        var panel = document.getElementById('ai-chat-panel');
        if (panel) {
          var cb = panel.querySelector('.ai-chat-include-selection');
          includeSel = !!(cb && cb.checked && chatState.currentSelection);
        }
        var msgs = [];
        msgs.push({ role: 'system', content: 'You are an AI assistant helping readers of the book Learning Deep Representations of Data Distributions. Answer clearly and concisely. If relevant, point to sections or headings from the current page.' });
        if (includeSel) msgs.push({ role: 'user', content: 'Context from selected text on the page:\n\n' + chatState.currentSelection });
        msgs.push({ role: 'user', content: userText });
        return msgs;
      }

      function getApiConfig(){
        // Users can override via: window.CHAT_API = { endpoint, model }
        // Default to calling a serverless proxy to avoid exposing API keys
        window.CHAT_API = window.CHAT_API || {
          endpoint: '/.netlify/functions/chat',
          model: 'gpt-4o-mini'
        };
        var cfg = (window.CHAT_API && typeof window.CHAT_API === 'object') ? window.CHAT_API : null;
        if (cfg && cfg.endpoint) return cfg;
        // Fallback placeholder (mock)
        return null;
      }

      function requestAssistant(messages){
        var cfg = getApiConfig();
        if (!cfg) {
          return Promise.resolve({ content: 'Mock response: AI chat is not configured. Set window.CHAT_API = { endpoint, apiKey, model } to connect to your backend (OpenAI-style). You asked: ' + (messages[messages.length-1] && messages[messages.length-1].content || '') });
        }
        var endpoint = cfg.endpoint;
        var body = { model: cfg.model || 'gpt-4o-mini', messages: messages, temperature: 0.2, stream: false };
        var headers = { 'Content-Type': 'application/json' };
        return fetch(endpoint, { method: 'POST', headers: headers, body: JSON.stringify(body) })
          .then(function(r){ return r.json(); })
          .then(function(j){
            // Try to read OpenAI-style
            var txt = (j && j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || (j && j.message && j.message.content) || (typeof j === 'string' ? j : JSON.stringify(j));
            return { content: txt || 'No content in response.' };
          })
          .catch(function(e){ return { content: 'Error contacting chat API: ' + (e && e.message ? e.message : String(e)) }; });
      }

      function sendChatMessage(){
        var input = document.getElementById('ai-chat-input'); if (!input) return;
        var text = (input.value || '').trim(); if (!text) return;
        input.value = '';
        appendMessage('user', text);
        chatState.messages.push({ role: 'user', content: text });
        setSending(true);
        var payload = buildPayload(text);
        var typingEl = appendTypingIndicator();
        requestAssistant(payload).then(function(res){
          removeTypingIndicator(typingEl);
          var msg = (res && res.content) || 'No response.';
          appendMessage('assistant', msg);
          chatState.messages.push({ role: 'assistant', content: msg });
        }).finally(function(){ removeTypingIndicator(typingEl); setSending(false); });
      }

      function openChat(){ ensureChatPanel(); document.body.classList.add('ai-chat-open'); }

      chatToggle && chatToggle.addEventListener('click', function(e){ e.preventDefault(); ensureChatPanel(); document.body.classList.toggle('ai-chat-open'); });

      // Track selection changes and update preview (debounced)
      var selTimer = null;
      function refreshSel(){
        var s = getTrimmedSelection();
        if (s) {
          chatState.currentSelection = s;
        }
        updateSelectionPreview();
      }
      ['mouseup','keyup','selectionchange','touchend'].forEach(function(evt){ document.addEventListener(evt, function(){ if (selTimer) clearTimeout(selTimer); selTimer = setTimeout(refreshSel, 120); }, { passive: true }); });

      // Clear stored selection when user clicks in the main text view (left content area)
      function clearStoredSelection(){ chatState.currentSelection = ''; updateSelectionPreview(); }
      function attachClearOnMainClick(){
        try {
          var main = document.querySelector('.ltx_page_main') || document.querySelector('.page');
          if (!main) return;
          // Avoid adding multiple listeners
          if (main.getAttribute('data-chat-clear-listener')) return;
          main.addEventListener('click', function(evt){
            var panel = document.getElementById('ai-chat-panel');
            if (panel && panel.contains(evt.target)) return; // ignore clicks inside chat
            clearStoredSelection();
          });
          main.setAttribute('data-chat-clear-listener', '1');
        } catch (_) {}
      }
      attachClearOnMainClick();
    } catch (e) {}

    // Wire up language dropdown (simple, non-intrusive behavior)
    try {
      var ls = bar.querySelector('.lang-select');
      var toggle = bar.querySelector('.lang-toggle');
      var menu = bar.querySelector('.lang-menu');
      if (ls && toggle && menu) {
        function closeMenu(){ ls.classList.remove('open'); toggle.setAttribute('aria-expanded','false'); }
        function openMenu(){ ls.classList.add('open'); toggle.setAttribute('aria-expanded','true'); }
        toggle.addEventListener('click', function(e){
          e.stopPropagation();
          if (ls.classList.contains('open')) closeMenu(); else openMenu();
        });
        document.addEventListener('click', function(){ closeMenu(); });
        document.addEventListener('keydown', function(e){ if(e.key==='Escape') closeMenu(); });
        // Selection handlers
        menu.querySelectorAll('.lang-item').forEach(function(btn){
          btn.addEventListener('click', function(e){
            var lang = btn.getAttribute('data-lang');
            var label = btn.textContent || btn.innerText || lang;
            var labelEl = ls.querySelector('.lang-label');
            if (labelEl) labelEl.textContent = (/^[A-Za-z]{2}$/.test(lang) ? lang.toUpperCase() : label);
            // Persist user preference for future
            try { window.localStorage && localStorage.setItem('book-lang', lang); } catch(_) {}
            closeMenu();
            // Soft behavior to avoid regressions: English is the current site; Chinese not yet available
            // Currently only English pages are available; both selections keep user on the same page
          });
        });
        // Initialize from stored preference
        try {
          var saved = window.localStorage && localStorage.getItem('book-lang');
          if (saved) {
            var initBtn = menu.querySelector(".lang-item[data-lang='"+saved+"']");
            if (initBtn) { var t = initBtn.textContent || initBtn.innerText || saved; var labelEl2 = ls.querySelector('.lang-label'); if (labelEl2) labelEl2.textContent = (/^[A-Za-z]{2}$/.test(saved) ? saved.toUpperCase() : t); }
          } else {
            // If document language is known, reflect it
            var docLang = (document.documentElement.getAttribute('lang')||'en').slice(0,2).toLowerCase();
            var labelEl3 = ls.querySelector('.lang-label');
            if (labelEl3) labelEl3.textContent = docLang.toUpperCase();
          }
        } catch (_) {}
      }
    } catch (e) {}
  }

  function renderSidebarInto(container, navLinks, toc){
    function linkItem(l){
      var a = h('a', { href:l.href });
      if (l.external) { a.setAttribute('target','_blank'); a.setAttribute('rel','noopener noreferrer'); }
      a.textContent = l.label;
      return h('li', { className:'nav-item' }, a);
    }
    var aside = h('aside', { className:'book-sidebar sidebar', 'data-shared-ui':'1' },
      h('div', { className:'side-section' }, h('div', { className:'side-h', text:'Navigation' }), h('ul', { className:'nav-list' }, (navLinks||DEFAULT_NAV_LINKS).map(linkItem))),
      h('div', { className:'side-section' }, h('div', { className:'side-h', text:'Table of Contents' }), h('ul', { className:'nav-list toc-list' }, (toc||DEFAULT_TOC).map(linkItem)))
    );
    if (container.firstChild) container.insertBefore(aside, container.firstChild); else container.appendChild(aside);
  }

  function maybeInsertSidebar(){
    var didInsert = false;
    // Landing layout
    var shell = document.querySelector('.app-shell');
    if (shell && !shell.querySelector('.book-sidebar')) {
      renderSidebarInto(shell, window.NAV_LINKS || DEFAULT_NAV_LINKS, window.TOC || DEFAULT_TOC);
      didInsert = true;
    }
    // Simple pages with layout-with-sidebar
    var lw = document.querySelector('.layout-with-sidebar');
    if (lw && !lw.querySelector('.book-sidebar')) {
      renderSidebarInto(lw, window.NAV_LINKS || DEFAULT_NAV_LINKS, window.TOC || DEFAULT_TOC);
      didInsert = true;
    }
    // Chapter pages: wrap main content with a sidebar layout if present
    try {
      var pageMain = document.querySelector('.ltx_page_main');
      // Avoid if a shared sidebar already exists or already wrapped
      var alreadyWrapped = pageMain && pageMain.parentElement && pageMain.parentElement.classList.contains('layout-with-sidebar');
      if (!didInsert && pageMain && !alreadyWrapped && !document.querySelector('.book-sidebar')) {
        var wrapper = document.createElement('div');
        wrapper.className = 'layout-with-sidebar';
        wrapper.setAttribute('data-shared-ui', '1');
        // Insert wrapper before pageMain and move pageMain inside (sidebar first so it renders in column 1)
        if (pageMain.parentNode) {
          pageMain.parentNode.insertBefore(wrapper, pageMain);
          // Insert sidebar as the first child so it occupies the first grid column
          renderSidebarInto(wrapper, window.NAV_LINKS || DEFAULT_NAV_LINKS, window.TOC || DEFAULT_TOC);
          wrapper.appendChild(pageMain);
          didInsert = true;
        }
      }
    } catch (_) {}
    return didInsert;
  }

  function ready(fn){ if (document.readyState==='loading') document.addEventListener('DOMContentLoaded', fn); else fn(); }
  ready(function(){
    try { renderTopBar(window.TOPBAR_OPTIONS || {}); } catch(e) {}
    try {
      // Attempt now
      var inserted = maybeInsertSidebar();
      // If containers may be rendered later (e.g., React), observe and try again
      if (!inserted && window.MutationObserver) {
        var obs = new MutationObserver(function(){
          if (maybeInsertSidebar()) {
            // once at least one sidebar is inserted, we can disconnect to avoid extra work
            try { obs.disconnect(); } catch (_) {}
          }
        });
        obs.observe(document.body || document.documentElement, { childList: true, subtree: true });
      }
    } catch(e) {}
  });

  // Expose APIs if custom use is needed
  window.insertTopBar = renderTopBar;
  window.insertSidebar = function(selector, nav, toc){
    var node = document.querySelector(selector);
    if (!node) return;
    if (node.querySelector('.sidebar')) return; // avoid duplicates
    renderSidebarInto(node, nav, toc);
  };
  window.DEFAULT_NAV_LINKS = DEFAULT_NAV_LINKS; window.DEFAULT_TOC = DEFAULT_TOC;
})();


