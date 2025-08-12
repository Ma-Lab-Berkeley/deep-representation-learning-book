(function () {
  function ready(fn) {
    if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn);
    else fn();
  }

  ready(function () {
    // Compute and expose layout variables for fixed top bars/headers
    try {
      var bookTopbar = document.getElementById('book-topbar');
      var navBar = document.querySelector('nav.ltx_page_navbar');
      var chapterHeader = document.querySelector('header.ltx_page_header');
      var tbH = bookTopbar ? bookTopbar.offsetHeight : 64;
      var nbH = navBar ? navBar.offsetHeight : 0;
      var chH = chapterHeader ? chapterHeader.offsetHeight : 56;
      document.documentElement.style.setProperty('--book-topbar-h', tbH + 'px');
      document.documentElement.style.setProperty('--navbar-h', nbH + 'px');
      document.documentElement.style.setProperty('--header-h', chH + 'px');
      window.addEventListener('resize', function(){
        var tb2 = document.getElementById('book-topbar');
        var nb2 = document.querySelector('nav.ltx_page_navbar');
        var ch2 = document.querySelector('header.ltx_page_header');
        document.documentElement.style.setProperty('--book-topbar-h', (tb2?tb2.offsetHeight:64) + 'px');
        document.documentElement.style.setProperty('--navbar-h', (nb2?nb2.offsetHeight:0) + 'px');
        document.documentElement.style.setProperty('--header-h', (ch2?ch2.offsetHeight:56) + 'px');
      }, {passive:true});
    } catch (e) {}

    // Topbar search is wired in shared-ui.js when the bar is rendered
    // Preserve server-side header links (no JS override)

    // Contributor metadata badges: transform data-badges into styled pills
    try {
      document.querySelectorAll('.card[data-badges]')
        .forEach(function(card){
          var raw = (card.getAttribute('data-badges') || '').trim();
          if (!raw) return;
          var parts = raw.split(',').map(function(s){ return s.trim(); }).filter(Boolean);
          if (!parts.length) return;
          var container = document.createElement('div');
          container.className = 'badges';
          parts.forEach(function(label){
            var b = document.createElement('span');
            b.className = 'badge';
            b.textContent = label;
            container.appendChild(b);
          });
          // Insert after affiliation if present, else append
          var aff = card.querySelector('.affil');
          if (aff && aff.parentNode === card) {
            aff.insertAdjacentElement('afterend', container);
          } else {
            card.appendChild(container);
          }
        });
    } catch (e) {}

    // 1) Add left-positioned link anchors on hover for section/subsection titles in main text
    const headings = document.querySelectorAll('.ltx_title_section, .ltx_title_subsection');
    headings.forEach((h) => {
      const containerSection = h.closest('section[id]');
      const targetId = h.id || (containerSection && containerSection.id);
      if (!targetId) return;
      if (!h.querySelector('.heading-anchor')) {
        const a = document.createElement('a');
        a.href = `#${targetId}`;
        a.className = 'heading-anchor';
        a.setAttribute('aria-label', 'Copy link to this section');
        a.textContent = '🔗';
        // Insert at beginning so it can be positioned to the left via CSS
        h.insertBefore(a, h.firstChild);
      }
    });

    // 1a) Add left-positioned link anchors for theorem-like environments
    const thmTitles = document.querySelectorAll('.ltx_title_theorem');
    thmTitles.forEach((t) => {
      // The target id is the surrounding theorem container's id
      const thmBox = t.closest('.ltx_theorem[id]');
      const targetId = (t.id) || (thmBox && thmBox.id);
      if (!targetId) return;
      if (!t.querySelector('.heading-anchor')) {
        const a = document.createElement('a');
        a.href = `#${targetId}`;
        a.className = 'heading-anchor';
        a.setAttribute('aria-label', 'Copy link to this environment');
        a.textContent = '🔗';
        t.insertBefore(a, t.firstChild);
      }
    });

    // 1b) Add anchor links for equations (outside, to the right of the box)
    // Remove any previous inline anchors for compatibility with older builds
    document.querySelectorAll('.heading-anchor-inline').forEach((n) => n.remove());
    document.querySelectorAll('.ltx_equation[id], .ltx_equationgroup[id]').forEach((eq) => {
      if (eq.querySelector('.heading-anchor-outside')) return;
      const a = document.createElement('a');
      a.href = `#${eq.id}`;
      a.className = 'heading-anchor-outside';
      a.setAttribute('aria-label', 'Copy link to this equation');
      a.textContent = '🔗';
      eq.appendChild(a);
    });

    // 1c) Add anchor links for figures, tables, and algorithms (top-right corner of the container)
    document.querySelectorAll('.ltx_figure[id], .ltx_table[id], .ltx_float.ltx_float_algorithm[id]').forEach((box) => {
      if (box.querySelector('.heading-anchor')) return;
      const a = document.createElement('a');
      a.href = `#${box.id}`;
      a.className = 'heading-anchor';
      a.setAttribute('aria-label', 'Copy link to this figure/table/algorithm');
      a.textContent = '🔗';
      box.insertBefore(a, box.firstChild);
    });

    // 1d) Fallback: plain tabular without wrapping figure (link to nearest labeled container or self)
    document.querySelectorAll('.ltx_tabular').forEach((tbl, idx) => {
      if (tbl.querySelector('.heading-anchor')) return;
      // Find nearest labeled container
      const container = tbl.closest('.ltx_table[id], .ltx_figure[id]');
      let targetId = container && container.id;
      if (!targetId) {
        if (!tbl.id) tbl.id = `tabular-${idx + 1}`;
        targetId = tbl.id;
      }
      const a = document.createElement('a');
      a.href = `#${targetId}`;
      a.className = 'heading-anchor';
      a.setAttribute('aria-label', 'Copy link to this table');
      a.textContent = '🔗';
      tbl.insertBefore(a, tbl.firstChild);
    });

    // 1.5) Mini ToC now generated statically during post-processing; no dynamic generation here

    // 2) Remove per-equation copy link toolbars if any were injected previously
    document.querySelectorAll('.eq-toolbar').forEach((n) => n.remove());

    // 3) Footnote hover popovers (robust to nested markup)
    let currentPop;
    function removePop() {
      if (currentPop) {
        currentPop.remove();
        currentPop = null;
      }
    }
    document.addEventListener('click', (e) => {
      if (!e.target.closest('.ltx_note')) removePop();
    });

    document.querySelectorAll('.ltx_note.ltx_role_footnote').forEach((fn) => {
      const mark = fn.querySelector('.ltx_note_mark');
      // Content may be inside .ltx_note_outer > .ltx_note_content
      const content = fn.querySelector('.ltx_note_content') || fn.querySelector('.ltx_note_outer .ltx_note_content');
      if (!mark || !content) return;
      const trigger = mark.closest('a, sup, span') || mark.parentElement;
      if (!trigger) return;
      trigger.style.cursor = 'help';
      trigger.addEventListener('mouseenter', () => {
        removePop();
        const pop = document.createElement('div');
        pop.className = 'footnote-pop';
        // Clone and sanitize footnote content
        const clone = content.cloneNode(true);
        clone.querySelectorAll('.ltx_note_mark, .ltx_tag.ltx_tag_note').forEach((el) => el.remove());
        let html = clone.innerHTML.replace(/^\s*<sup[^>]*>\s*\d+\s*<\/sup>\s*/i, '');
        html = html.replace(/^\s*\d+\s*/, '');
        pop.innerHTML = html;
        document.body.appendChild(pop);
        const rect = trigger.getBoundingClientRect();
        pop.style.top = Math.round(window.scrollY + rect.bottom + 6) + 'px';
        pop.style.left = Math.round(Math.min(window.scrollX + rect.left, window.scrollX + window.innerWidth - (pop.offsetWidth + 12))) + 'px';
        currentPop = pop;
      });
      trigger.addEventListener('mouseleave', () => setTimeout(removePop, 120));
    });

    // 4) Smooth scroll for internal refs
    document.querySelectorAll("a[href^='#']").forEach((a) => {
      a.addEventListener('click', (e) => {
        const href = a.getAttribute('href');
        const el = href && document.querySelector(href);
        if (el) {
          e.preventDefault();
          history.pushState(null, '', href);
          el.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      });
    });

    // 5) Repair tcolorbox-like environments emitted as SVG boxes with foreignObject
    //    We detect an SVG picture that contains a single foreignObject with an inner
    //    span.ltx_minipage (the usual LaTeXML output for tcolorbox). We then unwrap
    //    the text and replace the whole SVG with a div.tcbox so it looks consistent.
    try {
      document.querySelectorAll('svg.ltx_picture').forEach((svg) => {
        const minipage = svg.querySelector('foreignobject span.ltx_inline-block.ltx_minipage');
        if (!minipage) return;
        const raw = (minipage.innerText || minipage.textContent || '').trim();
        if (!raw) return;
        const text = raw.replace(/\s+/g, ' ');
        const wrapper = document.createElement('div');
        wrapper.className = 'tcbox';
        const p = document.createElement('p');
        p.className = 'ltx_p';
        p.textContent = text;
        wrapper.appendChild(p);
        svg.replaceWith(wrapper);
      });
    } catch (e) {}
  });
})();
