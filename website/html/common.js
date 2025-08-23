/* Common JS: shared UI (topbar, sidebar, search, chat) + layout variables */
(function(){
  // --- Begin shared-ui.js content ---
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
    function getDefaultNavLinks() {
      return window.BOOK_COMPONENTS ? window.BOOK_COMPONENTS.buildNavLinks() : [
        { label: 'Contributors', href: 'contributors.html' },
        { label: 'How to Contribute?', href: 'https://github.com/Ma-Lab-Berkeley/ldrdd-book#making-a-contribution', external: true }
      ];
    }
    var DEFAULT_NAV_LINKS = getDefaultNavLinks();
    function getDefaultTOC() {
      return window.BOOK_COMPONENTS ? window.BOOK_COMPONENTS.buildTOC() : [
        { label: 'Preface', href: 'Chx1.html' },
        { label: 'Chapter 1', subtitle: 'Introduction', href: 'Ch1.html' },
        { label: 'Chapter 2', subtitle: 'Learning Linear and Independent Structures', href: 'Ch2.html' },
        { label: 'Chapter 3', subtitle: 'Pursuing Low-Dimensional Distributions via Lossy Compression', href: 'Ch3.html' },
        { label: 'Chapter 4', subtitle: 'Deep Representations from Unrolled Optimization', href: 'Ch4.html' },
        { label: 'Chapter 5', subtitle: 'Consistent and Self-Consistent Representations', href: 'Ch5.html' },
        { label: 'Chapter 6', subtitle: 'Inference with Low-Dimensional Distributions', href: 'Ch6.html' },
        { label: 'Chapter 7', subtitle: 'Learning Representations for Real-World Data', href: 'Ch7.html' },
        { label: 'Chapter 8', subtitle: 'Future Study of Intelligence', href: 'Ch8.html' },
        { label: 'Appendix A', subtitle: 'Optimization Methods', href: 'A1.html' },
        { label: 'Appendix B', subtitle: 'Entropy, Diffusion, Denoising, and Lossy Coding', href: 'A2.html' },
      ];
    }
    var DEFAULT_TOC = getDefaultTOC();

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

    // Shared normalization used by search inputs
    function normalizeText(str){
      try {
        var s = (str || '').toString().toLowerCase();
        try { s = s.normalize('NFD').replace(/\p{Diacritic}+/gu, ''); } catch(_) {}
        s = s.replace(/[\-‐‑‒–—―_/\.]+/g, ' ');
        s = s.replace(/\s+/g, ' ').trim();
        return s;
      } catch(_) { return (str||'')+''; }
    }

    function tokenizeQuery(q){ return normalizeText(q).split(' ').filter(Boolean); }

    function editDistance(a, b){
      a = (a||''); b = (b||'');
      var al=a.length, bl=b.length;
      if (al===0) return bl; if (bl===0) return al;
      if (Math.abs(al-bl) > 2) return 3; // fast reject beyond our threshold
      var prev = new Array(bl+1); var curr = new Array(bl+1);
      for (var j=0;j<=bl;j++) prev[j]=j;
      for (var i=1;i<=al;i++){
        curr[0]=i; var ca=a.charCodeAt(i-1);
        for (var j=1;j<=bl;j++){
          var cb=b.charCodeAt(j-1);
          var cost = (ca===cb)?0:1;
          var ins = curr[j-1]+1, del = prev[j]+1, sub = prev[j-1]+cost;
          curr[j] = ins<del?(ins<sub?ins:sub):(del<sub?del:sub);
        }
        var tmp=prev; prev=curr; curr=tmp;
      }
      return prev[bl];
    }

    function fieldMatchScore(fieldNorm, token){
      if (!fieldNorm || !token) return 0;
      if (fieldNorm.indexOf(token) !== -1) return 2;
      // fuzzy: any word within edit distance <= threshold
      var words = fieldNorm.split(' ');
      var thresh = token.length >= 5 ? 2 : 1;
      for (var i=0;i<words.length;i++){
        var w = words[i]; if (!w) continue;
        if (Math.abs(w.length - token.length) > thresh) continue;
        if (editDistance(w, token) <= thresh) return 1;
      }
      return 0;
    }

    function computeEntryScore(entry, tokens){
      var t = normalizeText(entry.title||'');
      var s = normalizeText(entry.snippet||'');
      var p = normalizeText(entry.page||'');
      var score = 0;
      for (var k=0;k<tokens.length;k++){
        var tok = tokens[k]; var hit = false;
        var r = fieldMatchScore(t, tok); if (r===2){ score+=8; hit=true; } else if (r===1){ score+=5; hit=true; }
        r = fieldMatchScore(s, tok); if (r===2){ score+=3; hit=true; } else if (r===1){ score+=2; hit=true; }
        r = fieldMatchScore(p, tok); if (r===2){ score+=1; hit=true; } else if (r===1){ score+=0.5; hit=true; }
        if (!hit) score -= 5;
      }
      // Small boost for top-level entries (chapter/appx title entries have empty snippet)
      if (!entry.snippet) score += 1;
      return score;
    }

    function escapeHtml(s){ return (s||'').replace(/[&<>"']/g, function(c){ return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;','\'':'&#39;'}[c]); }); }
    function buildTokenRegex(token){
      var esc = token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
      esc = esc.replace(/\s+/g, '[\\s\-‐‑‒–—―_/\\.]+' );
      return new RegExp('('+esc+')', 'gi');
    }
    function highlightText(text, tokens){
      var out = escapeHtml(text||'');
      for (var i=0;i<tokens.length;i++){
        var tok = tokens[i]; if (!tok) continue;
        var re = buildTokenRegex(tok);
        out = out.replace(re, '<mark>$1</mark>');
      }
      return out;
    }

    // --- Lunr search integration ---
    var __SEARCH_DATA = null;            // Raw JSON from search-index.json { entries: [...] }
    var __LUNR_READY = false;            // Whether lunr library is loaded
    var __LUNR_LOADING = false;          // Prevent duplicate script injections
    var __LUNR_INDEX = null;             // Built lunr index
    var __LUNR_REF_TO_ENTRY = null;      // Map ref -> entry
    var __SEARCH_INIT_PROMISE = null;    // In-flight promise for ensuring index

    function ensureLunrLoaded(){
      if (window.lunr) { __LUNR_READY = true; return Promise.resolve(); }
      if (__LUNR_LOADING) {
        return new Promise(function(resolve){
          var check = function(){ if (window.lunr) { __LUNR_READY = true; resolve(); } else setTimeout(check, 20); };
          check();
        });
      }
      __LUNR_LOADING = true;
      return new Promise(function(resolve){
        try {
          var head = document.head || document.getElementsByTagName('head')[0];
          var s = document.createElement('script'); s.id='lunr-js'; s.async = true; s.defer = true;
          s.src = 'https://cdn.jsdelivr.net/npm/lunr@2.3.9/lunr.min.js';
          s.onload = function(){ __LUNR_READY = true; resolve(); };
          s.onerror = function(){ resolve(); };
          head && head.appendChild(s);
        } catch (_) { resolve(); }
      });
    }

    function ensureSearchData(){
      if (__SEARCH_DATA && __SEARCH_DATA.entries) return Promise.resolve(__SEARCH_DATA);
      return fetch('search-index.json').then(function(r){ return r && r.ok ? r.json() : null; }).then(function(j){ __SEARCH_DATA = j || { entries: [] }; return __SEARCH_DATA; }).catch(function(){ __SEARCH_DATA = { entries: [] }; return __SEARCH_DATA; });
    }

    function buildLunrIndex(){
      if (!__LUNR_READY || !window.lunr) return null;
      if (!__SEARCH_DATA || !__SEARCH_DATA.entries) return null;
      try {
        var entries = __SEARCH_DATA.entries || [];
        var refToEntry = Object.create(null);
        var idx = window.lunr(function(){
          this.ref('id');
          this.field('title', { boost: 8 });
          this.field('snippet', { boost: 3 });
          this.field('page', { boost: 1 });
          // Keep metadata for potential future snippet highlighting by position
          this.metadataWhitelist = ['position'];
          for (var i=0;i<entries.length;i++){
            var e = entries[i] || {};
            // Use numeric id for compact ref
            this.add({ id: String(i), title: (e.title||''), snippet: (e.snippet||''), page: (e.page||'') });
            refToEntry[String(i)] = e;
          }
        });
        __LUNR_INDEX = idx; __LUNR_REF_TO_ENTRY = refToEntry;
        return { idx: idx, map: refToEntry };
      } catch (_) { return null; }
    }

    function ensureLunrIndex(){
      if (__LUNR_INDEX && __LUNR_REF_TO_ENTRY) return Promise.resolve({ idx: __LUNR_INDEX, map: __LUNR_REF_TO_ENTRY });
      if (__SEARCH_INIT_PROMISE) return __SEARCH_INIT_PROMISE;
      __SEARCH_INIT_PROMISE = Promise.resolve().then(function(){ return ensureLunrLoaded(); }).then(function(){ return ensureSearchData(); }).then(function(){ return buildLunrIndex(); }).then(function(built){ __SEARCH_INIT_PROMISE = null; return built; }).catch(function(){ __SEARCH_INIT_PROMISE = null; return null; });
      return __SEARCH_INIT_PROMISE;
    }

    function lunrSearchEntries(qRaw, limit){
      limit = limit || 30;
      if (!__LUNR_INDEX || !window.lunr) return [];
      var tokens = tokenizeQuery(qRaw);
      if (!tokens.length) return [];
      try {
        var res = __LUNR_INDEX.query(function(q){
          for (var i=0;i<tokens.length;i++){
            var t = tokens[i]; if (!t) continue;
            var optsTrailing = { fields: ['title'], boost: 8, wildcard: window.lunr.Query.wildcard.TRAILING };
            var optsTrailing2 = { fields: ['snippet'], boost: 3, wildcard: window.lunr.Query.wildcard.TRAILING };
            var optsTrailing3 = { fields: ['page'], boost: 1, wildcard: window.lunr.Query.wildcard.TRAILING };
            q.term(t, optsTrailing);
            q.term(t, optsTrailing2);
            q.term(t, optsTrailing3);
            if (t.length >= 5) {
              q.term(t, { fields: ['title','snippet','page'], boost: 1, editDistance: 1 });
            }
          }
        });
        var out = [];
        for (var j=0;j<res.length && out.length<limit;j++){
          var r = res[j]; var e = __LUNR_REF_TO_ENTRY && __LUNR_REF_TO_ENTRY[r.ref];
          if (!e) continue;
          out.push(Object.assign({ kind: 'content', _score: r.score }, e));
        }
        return out;
      } catch (_) { return []; }
    }

    function renderTopBar(options){
      var existing = document.getElementById('book-topbar');
      var shouldReplace = !existing || (options && options.forceReplace === true) || (existing && !existing.getAttribute('data-shared-ui'));
      if (existing && shouldReplace) {
        try { existing.remove(); } catch (_) { /* ignore */ }
      } else if (existing && !shouldReplace) {
        return; // keep existing shared topbar
      }
      var title = (options && options.title) || (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.bookTitle) || 'Learning Deep Representations of Data Distributions';
      var langLabel = (options && options.langLabel) || (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.langLabel) || 'CN';
      var brandHref = (options && options.brandHref) || (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.brandHref) || 'index.html';

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
      var searchPlaceholder = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.searchPlaceholder) || 'Search pages…';
      var search = h('div', { className:'search' }, h('input', { className:'search-input', type:'search', placeholder:searchPlaceholder, 'aria-label':'Search' }), h('div', { className:'search-results' }));

      var ghIcon = h('svg', { xmlns:'http://www.w3.org/2000/svg', width:'16', height:'16', viewBox:'0 0 16 16', fill:'currentColor', role:'img', 'aria-label':'GitHub' },
        h('path', { d:'M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.2 1.87.86 2.33.66.07-.52.28-.86.51-1.06-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27.68 0 1.36.09 2 .27 1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.013 8.013 0 0 0 16 8c0-4.42-3.58-8-8-8z' })
      );

      // Language dropdown
      var langSelect = h('div', { className: 'lang-select' },
        h('button', { className: 'lang-toggle', type:'button', 'aria-haspopup': 'listbox', 'aria-expanded': 'false', title: 'Select language' },
          h('span', { className: 'lang-label', text: langLabel })
        ),
        h('div', { className: 'lang-menu', role: 'listbox' },
          h('button', { className: 'lang-item', role: 'option', 'data-lang': 'en', type:'button', text: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.languages.en) || 'English' }),
          h('button', { className: 'lang-item', role: 'option', 'data-lang': 'zh', type:'button', text: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.languages.zh) || '中文' })
        )
      );

      var chatTitle = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.askAITitle) || 'Ask AI about this page';
      var chatLabel = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.chatWithAI) || 'Chat with AI';
      var chatToggle = h('button', { className: 'chat-toggle', type: 'button', title: chatTitle },
        h('span', { className: 'chat-toggle-icon', html: '&#128172;' }),
        h('span', { className: 'chat-toggle-label', text: chatLabel })
      );

      // Hamburger menu icon
      var hamburgerIcon = h('svg', { xmlns:'http://www.w3.org/2000/svg', width:'16', height:'16', viewBox:'0 0 16 16', fill:'currentColor', role:'img', 'aria-label':'Menu' },
        h('path', { d:'M2 3h12a1 1 0 0 1 0 2H2a1 1 0 0 1 0-2zm0 4h12a1 1 0 0 1 0 2H2a1 1 0 0 1 0-2zm0 4h12a1 1 0 0 1 0 2H2a1 1 0 0 1 0-2z' })
      );

      var menuLabel = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.menu) || 'Menu';
      var hamburgerToggle = h('button', { className: 'hamburger-toggle', type: 'button', title: 'Toggle navigation menu' },
        hamburgerIcon,
        h('span', { className: 'hamburger-label', text: menuLabel })
      );

      var bar = h('div', { className:'book-topbar', id:'book-topbar', 'data-shared-ui':'1' },
        h('a', { className:'brand brand-link', href:brandHref }, logo, h('div', { className:'title', text:title })),
        h('div', { className:'topbar-right' },
          search,
          langSelect,
          chatToggle,
          h('a', { className:'gh-link', href:'https://github.com/Ma-Lab-Berkeley/ldrdd-book', target:'_blank', rel:'noopener noreferrer' }, ghIcon, h('span', { text: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.github) || 'GitHub' })),
          hamburgerToggle
        )
      );
      document.body.insertBefore(bar, document.body.firstChild);

      // Wire language selector dropdown + navigation
      try {
        function getCurrentLangFromPath(){
          try { var p = (window.location && window.location.pathname) || ''; return /\/zh(?:\/|$)/i.test(p) ? 'zh' : 'en'; } catch (_) { return 'en'; }
        }
        function toEnglishPath(path){
          try {
            var p = (path || '/');
            var parts = p.split('/');
            // Remove ONLY the standalone 'zh' language segment, preserving base paths
            var outParts = [];
            for (var i = 0; i < parts.length; i++) {
              var seg = parts[i];
              if (seg === 'zh') continue;
              outParts.push(seg);
            }
            var out = outParts.join('/');
            out = out.replace(/\/{2,}/g, '/');
            if (out.length > 1 && out.endsWith('/')) out = out.slice(0, -1);
            if (!out) out = '/';
            if (out[0] !== '/') out = '/' + out;
            return out;
          } catch(_) { return path || '/'; }
        }
        function toChinesePath(path){
          try {
            var p = (path||'/'); if (/\/zh(?:\/|$)/i.test(p)) return p; // already zh
            // Insert '/zh' right before the last segment (filename or trailing slash)
            var parts = p.split('/'); if (parts.length === 0) return '/zh/';
            var last = parts.pop(); // may be '' if p ends with '/'
            // Ensure leading slash
            var base = parts.join('/'); if (!base) base = '';
            var out;
            if (base === '' && last === '') { out = '/zh/'; }
            else if (last === '') { out = base + '/zh/'; }
            else { out = base + '/zh/' + last; }
            out = out.replace(/\/{2,}/g, '/');
            if (out[0] !== '/') out = '/' + out;
            return out;
          } catch(_) { return '/zh/'; }
        }
        function buildLangUrl(target){
          try {
            var loc = window.location || { pathname: '/' };
            var path = loc.pathname || '/';
            return target === 'zh' ? toChinesePath(path) : toEnglishPath(path);
          } catch(_) { return target === 'zh' ? '/zh/' : '/'; }
        }

        var langSelect = bar.querySelector('.lang-select');
        var langBtn = bar.querySelector('.lang-toggle');
        var langMenu = bar.querySelector('.lang-menu');
        var langLabelEl = bar.querySelector('.lang-label');
        if (langLabelEl) { try { var curr = getCurrentLangFromPath(); langLabelEl.textContent = curr === 'zh' ? '中文' : 'EN'; } catch(_) {} }
        function positionLangMenu(){
          try {
            if (!langBtn || !langMenu || !langSelect || !langSelect.classList.contains('open')) return;
            var r = langBtn.getBoundingClientRect();
            var menuW = Math.max(160, Math.min(260, r.width));
            langMenu.style.position = 'fixed';
            langMenu.style.top = Math.round(r.bottom + 6) + 'px';
            langMenu.style.left = Math.round(r.right - menuW) + 'px';
            langMenu.style.minWidth = menuW + 'px';
            langMenu.style.zIndex = '1200';
          } catch(_) {}
        }

        if (langBtn && langSelect) {
          langBtn.addEventListener('click', function(e){ e.preventDefault(); langSelect.classList.toggle('open'); positionLangMenu(); });
          document.addEventListener('click', function(evt){ try { if (!langSelect.contains(evt.target)) langSelect.classList.remove('open'); } catch(_){} }, { passive: true });
          document.addEventListener('keydown', function(e){ if (e.key === 'Escape') { try { langSelect.classList.remove('open'); } catch(_){} } }, { passive: true });
          window.addEventListener('resize', function(){ positionLangMenu(); }, { passive: true });
          window.addEventListener('scroll', function(){ positionLangMenu(); }, { passive: true });
          try {
            var items = langSelect.querySelectorAll('.lang-item');
            for (var i=0;i<items.length;i++){
              (function(btn){ btn.addEventListener('click', function(){
                try {
                  var target = btn.getAttribute('data-lang') || '';
                  var current = getCurrentLangFromPath();
                  if (target && target !== current) { var url = buildLangUrl(target); if (url) { window.location.href = url; } }
                } catch(_) {}
              }); })(items[i]);
            }
          } catch(_) {}
        }
      } catch (_) {}

      // Wire up search behavior (unified for all pages) - now using Lunr
      try {
        var input = bar.querySelector('.search-input');
        var box = bar.querySelector('.search-results');
        if (input && box) {
          var items = []; var active = -1; var open = false; var lastTokens = [];
          function positionBox(){
            try {
              if (!open) return;
              var r = input.getBoundingClientRect();
              box.style.position = 'fixed';
              box.style.left = Math.round(r.left) + 'px';
              box.style.top = Math.round(r.bottom + 4) + 'px';
              box.style.width = Math.round(r.width) + 'px';
              box.style.maxHeight = Math.round(Math.max(220, Math.min(520, window.innerHeight * 0.6))) + 'px';
              box.style.overflowY = 'auto';
              box.style.overflowX = 'hidden';
              box.style.zIndex = '1100';
            } catch (_) {}
          }
          function render(){
            box.innerHTML='';
            if(!open || !items.length){ box.style.display='none'; return; }
            items.forEach(function(it, i){
              var div = document.createElement('div'); div.className='search-item'+(i===active?' active':'');
              if(it.kind==='content'){
                var t=document.createElement('span'); t.className='search-item-title'; t.innerHTML=highlightText(it.title, lastTokens);
                var s=document.createElement('span'); s.className='search-secondary'; s.innerHTML=highlightText((it.page||'')+' — '+(it.snippet||''), lastTokens);
                div.appendChild(t); div.appendChild(s);
              } else { div.textContent = it.label; }
              div.onmousedown=function(e){e.preventDefault()};
              div.onclick=function(){ if(it.external){ window.open(it.href,'_blank','noopener,noreferrer'); } else { window.location.href = it.href; } };
              box.appendChild(div);
            });
            box.style.display='block';
            positionBox();
          }
          input.addEventListener('input', function(){
            var qRaw=(input.value||'');
            var q=normalizeText(qRaw); active=-1; open=true; lastTokens = tokenizeQuery(qRaw);
            if(!q){ items=[]; render(); return; }
            ensureLunrIndex().then(function(built){
              var results = lunrSearchEntries(qRaw, 30);
              // Fallback to legacy scoring if Lunr unavailable
              if ((!results || !results.length) && __SEARCH_DATA && __SEARCH_DATA.entries) {
                var scored=[]; for(var i=0;i<__SEARCH_DATA.entries.length;i++){ var e=__SEARCH_DATA.entries[i]; var sc = computeEntryScore(e, lastTokens); if (sc>0) scored.push(Object.assign({kind:'content', _score: sc}, e)); }
                scored.sort(function(a,b){ return b._score - a._score; });
                items = scored.slice(0, 30);
              } else {
                items = results;
              }
              render();
            });
          });
          input.addEventListener('focus', function(){ open=true; render(); });
          input.addEventListener('blur', function(){ setTimeout(function(){ open=false; render(); }, 120); });
          window.addEventListener('resize', function(){ positionBox(); }, { passive: true });
          window.addEventListener('scroll', function(){ positionBox(); }, { passive: true });
          input.addEventListener('keydown', function(e){ if(!items.length) return; if(e.key==='ArrowDown'){ e.preventDefault(); active=(active+1)%items.length; render(); } else if(e.key==='ArrowUp'){ e.preventDefault(); active=(active-1+items.length)%items.length; render(); } else if(e.key==='Enter'){ var target=items[Math.max(0,active)]||items[0]; if(!target) return; if(target.external){ window.open(target.href,'_blank','noopener,noreferrer'); } else { window.location.href = target.href; } } });

          // Mobile sidebar search is handled in renderSidebarInto()
        }
      } catch (e) {}

      // Wire chat toggle and ensure chat panel exists
      try {
        function ensureChatPanel(){
          if (document.getElementById('ai-chat-panel')) return;
          // Shell
          var chatTitleText = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.title) || 'Ask AI';
          var clearText = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.clear) || 'Clear';
          var closeText = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.close) || 'Close';
          var panel = h('div', { id: 'ai-chat-panel', className: 'ai-chat-panel', role: 'dialog', 'aria-modal': 'false', 'aria-labelledby': 'ai-chat-title' },
            h('div', { className: 'ai-chat-header' },
              h('div', { id: 'ai-chat-title', className: 'ai-chat-title', text: chatTitleText }),
              h('div', { className: 'ai-chat-actions' },
                h('button', { className: 'ai-chat-clear', type: 'button', title: 'Clear conversation', text: clearText }),
                h('button', { className: 'ai-chat-close', type: 'button', title: 'Close', text: closeText })
              )
            ),
            h('div', { className: 'ai-chat-context' },
              h('label', { className: 'ai-chat-ctx-row' },
                h('input', { type: 'checkbox', className: 'ai-chat-include-selection', checked: 'checked' }),
                h('span', { text: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.includeSelection) || 'Include current text selection' })
              ),
              h('div', { className: 'ai-chat-selection-preview' },
                h('div', { className: 'ai-chat-selection-empty', text: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.selectionEmpty) || 'Select text in the page to include it as context.' }),
                h('div', { className: 'ai-chat-selection-text' })
              )
            ),
            h('div', { className: 'ai-chat-messages', id: 'ai-chat-messages' }),
            h('form', { className: 'ai-chat-compose', id: 'ai-chat-form' },
              h('textarea', { className: 'ai-chat-input', id: 'ai-chat-input', rows: '3', placeholder: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.placeholder) || 'Ask a question about this page…\n\nYou can also ask about specific content by appending:\n@chapter (e.g., "@3"), @chapter.section (e.g., "@3.1"), @chapter.section.subsection (e.g., "@3.1.2")\n@appendix (e.g., "@A"), @appendix.section (e.g., "@A.1"), @appendix.section.subsection (e.g., "@A.1.2")' }),
              h('div', { className: 'ai-chat-sendrow' },
                h('button', { className: 'ai-chat-send', id: 'ai-chat-send', type: 'submit', text: (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.send) || 'Send' })
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
            document.body.classList.remove('ai-chat-wide');
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

        // Open/close panel when chat button is clicked
        try {
          var barEl = document.getElementById('book-topbar');
          var chatBtn = barEl && barEl.querySelector ? barEl.querySelector('.chat-toggle') : null;
          if (chatBtn) {
            chatBtn.addEventListener('click', function(){
              var isOpen = document.body.classList.contains('ai-chat-open');
              if (isOpen) {
                document.body.classList.remove('ai-chat-open');
                document.body.classList.remove('ai-chat-wide');
              } else {
                ensureChatPanel();
                document.body.classList.add('ai-chat-open');
                try { var ta = document.getElementById('ai-chat-input'); if (ta) ta.focus(); } catch(_) {}
                setTimeout(checkChatOverflow, 80);
              }
            });
          }
        } catch(_) {}

        // Lazy-load KaTeX and render math inside an element if LaTeX delimiters are present
        function containsLatex(text){
          try { if (!text) return false; return /(\$\$[^]*?\$\$|\$[^$]+\$|\\\([^]*?\\\)|\\\[[^]*?\\\])/.test(text); } catch (_) { return false; }
        }
        function ensureKatex(callback){
          try {
            if (window.renderMathInElement) { callback && callback(); return; }
            var head = document.head || document.getElementsByTagName('head')[0];
            if (!head) { callback && callback(); return; }
            // Prevent double-injection
            if (!document.getElementById('katex-css')){
              var link = document.createElement('link'); link.id = 'katex-css'; link.rel = 'stylesheet'; link.href = 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css'; head.appendChild(link);
            }
            function loadScript(id, src, onload){ if (document.getElementById(id)) { onload && onload(); return; } var s = document.createElement('script'); s.id = id; s.src = src; s.async = true; s.onload = onload; head.appendChild(s); }
            loadScript('katex-js', 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js', function(){ loadScript('katex-auto-render-js', 'https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js', function(){ callback && callback(); }); });
          } catch (_) { callback && callback(); }
        }
        function renderMathInBubble(el){ try { if (!el) return; var hasMath = containsLatex(el.textContent || ''); if (!hasMath) { setTimeout(checkChatOverflow, 30); return; } ensureKatex(function(){ try { if (!window.renderMathInElement) { setTimeout(checkChatOverflow, 30); return; } window.renderMathInElement(el, { delimiters: [ { left: '$$', right: '$$', display: true }, { left: '\\[', right: '\\]', display: true }, { left: '$', right: '$', display: false }, { left: '\\(', right: '\\)', display: false } ], throwOnError: false, strict: 'ignore', ignoredTags: ['script','noscript','style','textarea','pre','code'] }); setTimeout(checkChatOverflow, 60); } catch (_) { setTimeout(checkChatOverflow, 30); } }); } catch (_) { setTimeout(checkChatOverflow, 30); } }

        function getTrimmedSelection(maxLen){ maxLen = maxLen || 1200; var sel = window.getSelection && window.getSelection(); if (!sel || sel.isCollapsed) return ''; try { var panel = document.getElementById('ai-chat-panel'); if (panel && sel.rangeCount > 0) { var a = sel.anchorNode, f = sel.focusNode; var inside = (a && panel.contains(a)) || (f && panel.contains(f)); if (inside) return ''; } } catch (_) {} var text = (sel.toString() || '').trim(); text = text.replace(/\u00A0/g, ' '); text = text.replace(/\s+/g, ' '); if (!text) return ''; if (text.length > maxLen) text = text.slice(0, maxLen) + '\u2026'; return text; }

        function updateSelectionPreview(){ var panel = document.getElementById('ai-chat-panel'); if (!panel) return; var txt = chatState.currentSelection || ''; var empty = panel.querySelector('.ai-chat-selection-empty'); var box = panel.querySelector('.ai-chat-selection-text'); if (!box || !empty) return; if (txt) { empty.style.display = 'none'; box.textContent = txt; box.style.display = 'block'; } else { box.style.display = 'none'; empty.style.display = 'block'; } }
        function checkChatOverflow(){ try { var panel = document.getElementById('ai-chat-panel'); if (!panel) return; var list = panel.querySelector('#ai-chat-messages'); if (!list) return; var overflow = false; var bubbles = list.querySelectorAll('.ai-chat-bubble'); for (var i = 0; i < bubbles.length; i++) { var b = bubbles[i]; if (b.scrollWidth > b.clientWidth + 4) { overflow = true; break; } var kd = b.querySelector('.katex-display'); if (kd && kd.scrollWidth > kd.clientWidth + 4) { overflow = true; break; } } if (overflow) { document.body.classList.add('ai-chat-wide'); } else { document.body.classList.remove('ai-chat-wide'); } } catch (_) {} }

        function appendMessage(role, content){ var panel = document.getElementById('ai-chat-panel'); if (!panel) return; var list = panel.querySelector('#ai-chat-messages'); if (!list) return; var item = document.createElement('div'); item.className = 'ai-chat-msg ' + (role === 'user' ? 'from-user' : 'from-assistant'); var bubble = document.createElement('div'); bubble.className = 'ai-chat-bubble'; bubble.textContent = content; item.appendChild(bubble); list.appendChild(item); renderMathInBubble(bubble); setTimeout(checkChatOverflow, 50); list.scrollTop = list.scrollHeight + 999; }
        function appendTypingIndicator(){ var panel = document.getElementById('ai-chat-panel'); if (!panel) return null; var list = panel.querySelector('#ai-chat-messages'); if (!list) return null; var item = document.createElement('div'); item.className = 'ai-chat-msg from-assistant typing'; var bubble = document.createElement('div'); bubble.className = 'ai-chat-bubble'; var typing = document.createElement('div'); typing.className = 'ai-typing'; for (var i = 0; i < 3; i++) { var dot = document.createElement('span'); dot.className = 'dot'; typing.appendChild(dot); } bubble.appendChild(typing); item.appendChild(bubble); list.appendChild(item); list.scrollTop = list.scrollHeight + 999; return item; }
        function removeTypingIndicator(el){ try { if (el && el.parentNode) el.parentNode.removeChild(el); checkChatOverflow(); } catch (_) {} }
        function setSending(isSending){ var btn = document.getElementById('ai-chat-send'); var input = document.getElementById('ai-chat-input'); if (btn) btn.disabled = !!isSending; if (input) input.disabled = !!isSending; setTimeout(checkChatOverflow, 60); }

        function getApiConfig(){ window.CHAT_API = window.CHAT_API || { endpoint: 'https://deep-representation-learning-book-proxy.tianzhechu.workers.dev/api/chat' }; var cfg = (window.CHAT_API && typeof window.CHAT_API === 'object') ? window.CHAT_API : null; if (cfg && cfg.endpoint) return cfg; return null; }
        function requestAssistant(messages){ var cfg = getApiConfig(); if (!cfg) { return Promise.resolve({ content: 'Mock response: AI chat is not configured. Set window.CHAT_API = { endpoint, apiKey, model } to connect to your backend (OpenAI-style).' }); } var endpoint = cfg.endpoint; var body = { model: cfg.model || 'bookqa-7b', messages: messages, temperature: 0.2, stream: false }; var headers = { 'Content-Type': 'application/json' }; if (cfg.apiKey) headers['Authorization'] = 'Bearer ' + cfg.apiKey; return fetch(endpoint, { method: 'POST', headers: headers, body: JSON.stringify(body) }).then(function(r){ return r.json(); }).then(function(j){ var txt = (j && j.choices && j.choices[0] && j.choices[0].message && j.choices[0].message.content) || (j && j.message && j.message.content) || (typeof j === 'string' ? j : JSON.stringify(j)); return { content: txt || 'No content in response.' }; }).catch(function(e){ return { content: 'Error contacting chat API: ' + (e && e.message ? e.message : String(e)) }; }); }
        // --- Chapter, section, subsection, and appendix mention handling (e.g., "@3", "@3.6", "@3.1.2", "@A", "@A.1", "@A.1.2") ---
        function getCurrentChapterFromPath(){ try { var m = (window.location && window.location.pathname || '').match(/\bCh(\d+)\.html$/i); return m ? parseInt(m[1], 10) : null; } catch (_) { return null; } }
        function parseSectionMentions(text){ var out = []; try { if (!text) return out; var seen = Object.create(null); var m; 
          // Match @appendix.section.subsection format (e.g., @A.1.2) - most specific first
          var appendixSubsectionRe = /@([A-Z])\.(\d+)\.(\d+)/g; while ((m = appendixSubsectionRe.exec(text))){ var app = m[1]; var sec = parseInt(m[2],10); var sub = parseInt(m[3],10); if (!isFinite(sec) || !isFinite(sub)) continue; var key = app+":"+sec+":"+sub; if (seen[key]) continue; seen[key] = 1; out.push({ appendix: app, section: sec, subsection: sub }); }
          // Match @chapter.section.subsection format (e.g., @3.1.2) - most specific first
          var subsectionRe = /@(\d+)\.(\d+)\.(\d+)/g; while ((m = subsectionRe.exec(text))){ var ch = parseInt(m[1],10); var sec = parseInt(m[2],10); var sub = parseInt(m[3],10); if (!isFinite(ch) || !isFinite(sec) || !isFinite(sub)) continue; var key = ch+":"+sec+":"+sub; if (seen[key]) continue; seen[key] = 1; out.push({ chapter: ch, section: sec, subsection: sub }); }
          // Match @appendix.section format (e.g., @A.1) - but only if not already part of @appendix.section.subsection
          var appendixSectionRe = /@([A-Z])\.(\d+)(?!\.)/g; while ((m = appendixSectionRe.exec(text))){ var app = m[1]; var sec = parseInt(m[2],10); if (!isFinite(sec)) continue; var key = app+":"+sec; if (seen[key]) continue; seen[key] = 1; out.push({ appendix: app, section: sec, subsection: null }); }
          // Match @chapter.section format (e.g., @3.6) - but only if not already part of @chapter.section.subsection
          var sectionRe = /@(\d+)\.(\d+)(?!\.)/g; while ((m = sectionRe.exec(text))){ var ch = parseInt(m[1],10); var sec = parseInt(m[2],10); if (!isFinite(ch) || !isFinite(sec)) continue; var key = ch+":"+sec; if (seen[key]) continue; seen[key] = 1; out.push({ chapter: ch, section: sec, subsection: null }); }
          // Match @appendix format (e.g., @A) - but only if not already part of @appendix.section
          var appendixRe = /@([A-Z])(?!\.)/g; while ((m = appendixRe.exec(text))){ var app = m[1]; var key = app+":appendix"; if (seen[key]) continue; seen[key] = 1; out.push({ appendix: app, section: null, subsection: null }); }
          // Match @chapter format (e.g., @3) - but only if not already part of @chapter.section
          var chapterRe = /@(\d+)(?!\.)/g; while ((m = chapterRe.exec(text))){ var ch = parseInt(m[1],10); if (!isFinite(ch)) continue; var key = ch+":chapter"; if (seen[key]) continue; seen[key] = 1; out.push({ chapter: ch, section: null, subsection: null }); }
        } catch (_) {} return out; }
        var __APPENDIX_MAPS_CACHE = null;
        function buildAppendixMaps(){ if (__APPENDIX_MAPS_CACHE) return __APPENDIX_MAPS_CACHE; var numToLetter = Object.create(null); var letterToNum = Object.create(null); try { var toc = window.TOC || DEFAULT_TOC; for (var i=0; i<toc.length; i++) { var item = toc[i]; if (!item.label || !item.href) continue; var m = item.label.match(/^Appendix\s+([A-Z])/i); var h = item.href.match(/^A(\d+)\.html$/i); if (m && h) { var letter = m[1].toUpperCase(); var num = parseInt(h[1], 10); if (isFinite(num)) { numToLetter[num] = letter; letterToNum[letter] = num; } } } } catch (_) {} __APPENDIX_MAPS_CACHE = { numToLetter: numToLetter, letterToNum: letterToNum }; return __APPENDIX_MAPS_CACHE; }
        function getCurrentAppendixFromPath(){ try { var m = (window.location && window.location.pathname || '').match(/\bA(\d+)\.html$/i); if (!m) return null; var num = parseInt(m[1], 10); var maps = buildAppendixMaps(); return maps.numToLetter[num] || null; } catch (_) { return null; } }
        function fetchDocument(mention){ return new Promise(function(resolve){ try { if (mention.chapter !== undefined) { var current = getCurrentChapterFromPath(); if (current && current === mention.chapter) { resolve(document); return; } var url = 'Ch' + String(mention.chapter) + '.html'; } else if (mention.appendix !== undefined) { var currentApp = getCurrentAppendixFromPath(); if (currentApp && currentApp === mention.appendix) { resolve(document); return; } var maps = buildAppendixMaps(); var num = maps.letterToNum[mention.appendix]; if (!num) { resolve(null); return; } var url = 'A' + String(num) + '.html'; } else { resolve(null); return; } fetch(url, { method: 'GET' }).then(function(r){ if (!r || !r.ok) { resolve(null); return; } return r.text(); }).then(function(html){ if (!html) { resolve(null); return; } try { var parser = new DOMParser(); var doc = parser.parseFromString(html, 'text/html'); resolve(doc || null); } catch (_) { resolve(null); } }).catch(function(){ resolve(null); }); } catch (_) { resolve(null); } }); }
        function extractSectionTextFromDoc(doc, sectionNumber, maxLen){ try { if (!doc) return ''; var secId = 'S' + String(sectionNumber); var el = doc.querySelector('section.ltx_section#' + secId); if (!el) return ''; var text = (el.innerText || el.textContent || '').replace(/\u00A0/g, ' '); text = text.replace(/\s+/g, ' ').trim(); if (!text) return ''; var MAX = typeof maxLen === 'number' ? maxLen : 6000; if (text.length > MAX) text = text.slice(0, MAX) + '\u2026'; return text; } catch (_) { return ''; } }
        function extractChapterTextFromDoc(doc, maxLen){ try { if (!doc) return ''; var main = doc.querySelector('.ltx_page_main') || doc.querySelector('main') || doc.body; if (!main) return ''; var text = (main.innerText || main.textContent || '').replace(/\u00A0/g, ' '); text = text.replace(/\s+/g, ' ').trim(); if (!text) return ''; var MAX = typeof maxLen === 'number' ? maxLen : 8000; if (text.length > MAX) text = text.slice(0, MAX) + '\u2026'; return text; } catch (_) { return ''; } }
        function extractSubsectionTextFromDoc(doc, sectionNumber, subsectionNumber, maxLen){ try { if (!doc) return ''; var subId = 'S' + String(sectionNumber) + '.SS' + String(subsectionNumber); var el = doc.querySelector('section.ltx_subsection#' + subId); if (!el) return ''; var text = (el.innerText || el.textContent || '').replace(/\u00A0/g, ' '); text = text.replace(/\s+/g, ' ').trim(); if (!text) return ''; var MAX = typeof maxLen === 'number' ? maxLen : 4000; if (text.length > MAX) text = text.slice(0, MAX) + '\u2026'; return text; } catch (_) { return ''; } }
        function buildPayloadAsync(userText){ return new Promise(function(resolve){ try {
            var includeSel = false; var panel = document.getElementById('ai-chat-panel'); if (panel) { var cb = panel.querySelector('.ai-chat-include-selection'); includeSel = !!(cb && cb.checked && chatState.currentSelection); }
            var systemPrompt = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.chat.systemPrompt) || 'You are an AI assistant helping readers of the book Learning Deep Representations of Data Distributions. Answer clearly and concisely. If relevant, point to sections or headings from the current page.';
            var msgs = []; msgs.push({ role: 'system', content: systemPrompt }); if (includeSel) msgs.push({ role: 'user', content: 'Context from selected text on the page:\n\n' + chatState.currentSelection });
            var mentions = parseSectionMentions(userText);
            if (!mentions.length) {
              msgs.push({ role: 'user', content: userText }); resolve(msgs); return;
            }
            var pending = mentions.length; if (!pending) { msgs.push({ role: 'user', content: userText }); resolve(msgs); return; }
            mentions.forEach(function(mn){ fetchDocument(mn).then(function(doc){ var txt = ''; var contextLabel = ''; var identifier = mn.chapter !== undefined ? mn.chapter : mn.appendix; var type = mn.chapter !== undefined ? 'chapter' : 'appendix'; if (mn.subsection !== null && mn.subsection !== undefined) { txt = extractSubsectionTextFromDoc(doc, mn.section, mn.subsection, 4000); contextLabel = 'Context from ' + type + ' ' + identifier + ' subsection ' + identifier + '.' + mn.section + '.' + mn.subsection + ':'; } else if (mn.section !== null && mn.section !== undefined) { txt = extractSectionTextFromDoc(doc, mn.section, 6000); contextLabel = 'Context from ' + type + ' ' + identifier + ' section ' + identifier + '.' + mn.section + ':'; } else { txt = extractChapterTextFromDoc(doc, 8000); contextLabel = 'Context from ' + type + ' ' + identifier + ':'; } if (txt) { msgs.push({ role: 'user', content: contextLabel + '\n\n' + txt }); } }).finally(function(){ pending--; if (pending === 0) { msgs.push({ role: 'user', content: userText }); resolve(msgs); } }); });
          } catch (_) { resolve([{ role: 'system', content: 'You are an AI assistant helping readers of the book Learning Deep Representations of Data Distributions. Answer clearly and concisely. If relevant, point to sections or headings from the current page.' }, { role: 'user', content: userText }]); }
        }); }
        function sendChatMessage(){ var input = document.getElementById('ai-chat-input'); if (!input) return; var text = (input.value || '').trim(); if (!text) return; input.value = ''; appendMessage('user', text); setSending(true); var typingEl = appendTypingIndicator(); buildPayloadAsync(text).then(function(payload){ return requestAssistant(payload); }).then(function(res){ removeTypingIndicator(typingEl); var msg = (res && res.content) || 'No response.'; appendMessage('assistant', msg); }).finally(function(){ removeTypingIndicator(typingEl); setSending(false); }); }

        function clearStoredSelection(){ chatState.currentSelection = ''; updateSelectionPreview(); }
        function attachClearOnMainClick(){ try { var main = document.querySelector('.ltx_page_main') || document.querySelector('.page'); if (!main) return; if (main.getAttribute('data-chat-clear-listener')) return; main.addEventListener('click', function(evt){ var panel = document.getElementById('ai-chat-panel'); if (panel && panel.contains(evt.target)) return; clearStoredSelection(); }); main.setAttribute('data-chat-clear-listener', '1'); } catch (_) {} }

        // Wire up hamburger toggle
        var bar = document.getElementById('book-topbar');
        var hamburgerBtn = bar && bar.querySelector ? bar.querySelector('.hamburger-toggle') : null;
        if (hamburgerBtn) { hamburgerBtn.addEventListener('click', function(e){ e.preventDefault(); document.body.classList.toggle('mobile-nav-open'); }); }

        // Track selection changes and update preview (debounced)
        var selTimer = null; function refreshSel(){ var s = getTrimmedSelection(); if (s) { chatState.currentSelection = s; } updateSelectionPreview(); }
        ['mouseup','keyup','selectionchange','touchend'].forEach(function(evt){ document.addEventListener(evt, function(){ if (selTimer) clearTimeout(selTimer); selTimer = setTimeout(function(){ refreshSel(); checkChatOverflow(); }, 120); }, { passive: true }); });
        window.addEventListener('resize', function(){ if (selTimer) clearTimeout(selTimer); selTimer = setTimeout(checkChatOverflow, 120); }, { passive: true });
        attachClearOnMainClick();
      } catch (e) {}
    }

    function renderSidebarInto(container, navLinks, toc){
      function linkItem(l){
        var a = h('a', { href:l.href });
        if (l.external) { a.setAttribute('target','_blank'); a.setAttribute('rel','noopener noreferrer'); }
        a.appendChild(h('span', { className: 'nav-label', text: l.label }));
        if (l.subtitle) {
          a.appendChild(h('span', { className: 'nav-subtitle', text: l.subtitle }));
        }
        return h('li', { className:'nav-item' }, a);
      }
      var searchHeaderText = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.sidebar.search) || 'Search';
      var mobileSearchPlaceholder = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.ui.searchPlaceholder) || 'Search pages…';
      var mobileSearch = h('div', { className: 'mobile-search-section side-section' }, h('div', { className: 'side-h', text: searchHeaderText }), h('div', { className: 'mobile-search-container' }, h('input', { className: 'mobile-search-input', type: 'search', placeholder: mobileSearchPlaceholder, 'aria-label': 'Search' }), h('div', { className: 'mobile-search-results' })) );
      var navigationText = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.sidebar.navigation) || 'Navigation';
      var tocText = (window.BOOK_COMPONENTS && window.BOOK_COMPONENTS.sidebar.tableOfContents) || 'Table of Contents';
      var defaultNavLinks = navLinks || (window.BOOK_COMPONENTS ? getDefaultNavLinks() : DEFAULT_NAV_LINKS);
      var defaultTOC = toc || (window.BOOK_COMPONENTS ? getDefaultTOC() : DEFAULT_TOC);
      var aside = h('aside', { className:'book-sidebar sidebar', 'data-shared-ui':'1' }, mobileSearch, h('div', { className:'side-section' }, h('div', { className:'side-h', text: navigationText }), h('ul', { className:'nav-list' }, defaultNavLinks.map(linkItem))), h('div', { className:'side-section' }, h('div', { className:'side-h', text: tocText }), h('ul', { className:'nav-list toc-list' }, defaultTOC.map(linkItem))));
      if (container.firstChild) container.insertBefore(aside, container.firstChild); else container.appendChild(aside);

      // Wire up mobile sidebar search (independent from topbar search)
      try {
        var mInput = aside.querySelector('.mobile-search-input');
        var mBox = aside.querySelector('.mobile-search-results');
        if (mInput && mBox) {
          var mItems = []; var mActive = -1; var mOpen = false; var mLastTokens = [];
          function mRender(){
            mBox.innerHTML='';
            if(!mOpen || !mItems.length){ mBox.style.display='none'; return; }
            mItems.forEach(function(it, i){
              var div = document.createElement('div'); div.className='search-item'+(i===mActive?' active':'');
              if(it.kind==='content'){
                var t=document.createElement('span'); t.className='search-item-title'; t.innerHTML = highlightText(it.title, mLastTokens);
                var s=document.createElement('span'); s.className='search-secondary'; s.innerHTML = highlightText((it.page||'')+' — '+(it.snippet||''), mLastTokens);
                div.appendChild(t); div.appendChild(s);
              } else { div.textContent = it.label; }
              div.onmousedown=function(e){e.preventDefault()};
              div.onclick=function(){ if(it.external){ window.open(it.href,'_blank','noopener,noreferrer'); } else { window.location.href = it.href; } };
              mBox.appendChild(div);
            });
            mBox.style.display='block';
            try { mBox.style.maxHeight = '55vh'; mBox.style.overflowY = 'auto'; mBox.style.overflowX = 'hidden'; } catch(_) {}
          }
          mInput.addEventListener('input', function(){
            var q=normalizeText(mInput.value||''); mActive=-1; mOpen=true; mLastTokens = tokenizeQuery(mInput.value||'');
            if(!q){ mItems=[]; mRender(); return; }
            ensureLunrIndex().then(function(){
              var results = lunrSearchEntries(mInput.value||'', 30);
              if ((!results || !results.length) && __SEARCH_DATA && __SEARCH_DATA.entries) {
                var scored=[]; for(var i=0;i<__SEARCH_DATA.entries.length;i++){ var e=__SEARCH_DATA.entries[i]; var sc = computeEntryScore(e, mLastTokens); if (sc>0) scored.push(Object.assign({kind:'content', _score: sc}, e)); }
                scored.sort(function(a,b){ return b._score - a._score; });
                mItems = scored.slice(0, 30);
              } else {
                mItems = results;
              }
              mRender();
            });
          });
          mInput.addEventListener('focus', function(){ mOpen=true; mRender(); });
          mInput.addEventListener('blur', function(){ setTimeout(function(){ mOpen=false; mRender(); }, 120); });
          mInput.addEventListener('keydown', function(e){ if(!mItems.length) return; if(e.key==='ArrowDown'){ e.preventDefault(); mActive=(mActive+1)%mItems.length; mRender(); } else if(e.key==='ArrowUp'){ e.preventDefault(); mActive=(mActive-1+mItems.length)%mItems.length; mRender(); } else if(e.key==='Enter'){ var target=mItems[Math.max(0,mActive)]||mItems[0]; if(!target) return; if(target.external){ window.open(target.href,'_blank','noopener,noreferrer'); } else { window.location.href = target.href; } } });
          document.addEventListener('click', function(evt){ if (!aside.contains(evt.target)) { mOpen=false; mRender(); } }, { passive: true });
        }
      } catch (_) {}
    }

    function maybeInsertSidebar(){
      var didInsert = false;
      // Landing layout
      var shell = document.querySelector('.app-shell');
      if (shell && !shell.querySelector('.book-sidebar')) { renderSidebarInto(shell, window.NAV_LINKS || (window.BOOK_COMPONENTS ? getDefaultNavLinks() : DEFAULT_NAV_LINKS), window.TOC || (window.BOOK_COMPONENTS ? getDefaultTOC() : DEFAULT_TOC)); didInsert = true; }
      // Simple pages with layout-with-sidebar
      var lw = document.querySelector('.layout-with-sidebar');
      if (lw && !lw.querySelector('.book-sidebar')) { renderSidebarInto(lw, window.NAV_LINKS || (window.BOOK_COMPONENTS ? getDefaultNavLinks() : DEFAULT_NAV_LINKS), window.TOC || (window.BOOK_COMPONENTS ? getDefaultTOC() : DEFAULT_TOC)); didInsert = true; }
      // Chapter pages: wrap main content with a sidebar layout if present
      try {
        var pageMain = document.querySelector('.ltx_page_main');
        var alreadyWrapped = pageMain && pageMain.parentElement && pageMain.parentElement.classList.contains('layout-with-sidebar');
        if (!didInsert && pageMain && !alreadyWrapped && !document.querySelector('.book-sidebar')) {
          var wrapper = document.createElement('div');
          wrapper.className = 'layout-with-sidebar';
          wrapper.setAttribute('data-shared-ui', '1');
          if (pageMain.parentNode) {
            pageMain.parentNode.insertBefore(wrapper, pageMain);
            renderSidebarInto(wrapper, window.NAV_LINKS || (window.BOOK_COMPONENTS ? getDefaultNavLinks() : DEFAULT_NAV_LINKS), window.TOC || (window.BOOK_COMPONENTS ? getDefaultTOC() : DEFAULT_TOC));
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
        var inserted = maybeInsertSidebar();
        if (!inserted && window.MutationObserver) {
          var obs = new MutationObserver(function(){ if (maybeInsertSidebar()) { try { obs.disconnect(); } catch (_) {} } });
          obs.observe(document.body || document.documentElement, { childList: true, subtree: true });
        }
      } catch(e) {}
    });

    // Expose APIs
    window.insertTopBar = renderTopBar;
    window.insertSidebar = function(selector, nav, toc){ var node = document.querySelector(selector); if (!node) return; if (node.querySelector('.sidebar')) return; renderSidebarInto(node, nav, toc); };
    window.DEFAULT_NAV_LINKS = DEFAULT_NAV_LINKS; window.DEFAULT_TOC = DEFAULT_TOC;
    window.getDefaultNavLinks = getDefaultNavLinks; window.getDefaultTOC = getDefaultTOC;
  })();
  // --- End shared-ui.js content ---

  // --- Begin layout variable synchronization (from book.js) ---
  (function(){
    function ready(fn){ if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn); else fn(); }
    function setVars(){
      try {
        var tb = document.getElementById('book-topbar');
        var nb = document.querySelector('nav.ltx_page_navbar');
        var ch = document.querySelector('header.ltx_page_header');
        document.documentElement.style.setProperty('--book-topbar-h', (tb?tb.offsetHeight:64) + 'px');
        document.documentElement.style.setProperty('--navbar-h', (nb?nb.offsetHeight:0) + 'px');
        document.documentElement.style.setProperty('--header-h', (ch?ch.offsetHeight:56) + 'px');
      } catch (e) {}
    }
    ready(function(){ setVars(); window.addEventListener('resize', function(){ setVars(); }, {passive:true}); });
  })();
  // --- End layout variable synchronization ---
})();


