/* Chapter JS: behaviors for LaTeXML/ar5iv chapter pages */
(function () {
  function ready(fn) { if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', fn); else fn(); }

  ready(function () {
    // 1) Add left-positioned link anchors on hover for section/subsection titles in main text
    try {
      const headings = document.querySelectorAll('.ltx_title_section, .ltx_title_subsection, .ltx_title_paragraph');
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
          h.insertBefore(a, h.firstChild);
        }
      });
    } catch (e) {}

    // 1a) Add left-positioned link anchors for theorem-like environments
    try {
      const thmTitles = document.querySelectorAll('.ltx_title_theorem');
      thmTitles.forEach((t) => {
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
    } catch (e) {}

    // 1b) Add anchor links for equations (outside, to the right of the box)
    try {
      document.querySelectorAll('.ltx_equation[id], .ltx_equationgroup[id]').forEach((eq) => {
        if (eq.querySelector('.heading-anchor-outside')) return;
        const a = document.createElement('a');
        a.href = `#${eq.id}`;
        a.className = 'heading-anchor-outside';
        a.setAttribute('aria-label', 'Copy link to this equation');
        a.textContent = '🔗';
        eq.appendChild(a);
      });
    } catch (e) {}

    // 1c) Add anchor links for figures, tables, and algorithms (top-right corner of the container)
    try {
      document.querySelectorAll('.ltx_figure[id], .ltx_table[id], .ltx_float.ltx_float_algorithm[id]').forEach((box) => {
        if (box.querySelector('.heading-anchor')) return;
        const a = document.createElement('a');
        a.href = `#${box.id}`;
        a.className = 'heading-anchor';
        a.setAttribute('aria-label', 'Copy link to this figure/table/algorithm');
        a.textContent = '🔗';
        box.insertBefore(a, box.firstChild);
      });
    } catch (e) {}

    // 1e) Post-process algorithm captions: wrap caption text in parentheses with a trailing period
    try {
      // Remove any legacy IO labels if present
      document.querySelectorAll('.alg-io-label').forEach((n) => n.remove());

      document.querySelectorAll('.ltx_float.ltx_float_algorithm .ltx_caption').forEach((cap) => {
        if (cap.dataset.algCaptionWrapped === '1') return;
        const tag = cap.querySelector('.ltx_tag.ltx_tag_float');
        if (!tag) return;

        // Collect all following siblings (caption text and inline nodes)
        const nodes = [];
        let cur = tag.nextSibling;
        while (cur) {
          nodes.push(cur);
          cur = cur.nextSibling;
        }
        if (nodes.length === 0) return;

        // Create a wrapper for the caption text
        const wrapper = document.createElement('span');
        wrapper.className = 'alg-caption-text';
        nodes.forEach((node) => wrapper.appendChild(node));

        // Trim leading whitespace inside wrapper to avoid space after '('
        while (wrapper.firstChild && wrapper.firstChild.nodeType === Node.TEXT_NODE) {
          const v = wrapper.firstChild.nodeValue || '';
          if (/^\s+$/.test(v)) { wrapper.removeChild(wrapper.firstChild); continue; }
          wrapper.firstChild.nodeValue = v.replace(/^\s+/, '');
          break;
        }

        // Remove any trailing whitespace and one trailing period from wrapper; we'll add period outside ')'
        let trimmed = false;
        for (let n = wrapper.lastChild; n; n = wrapper.lastChild) {
          if (n.nodeType === Node.TEXT_NODE) {
            let val = n.nodeValue || '';
            // Remove trailing whitespace
            val = val.replace(/\s+$/, '');
            // Remove one trailing period or full stop char
            if (!trimmed && /[.。]$/.test(val)) { val = val.replace(/[.。]$/, ''); trimmed = true; }
            if (val.length === 0) { wrapper.removeChild(n); continue; }
            n.nodeValue = val; break;
          } else if (n.nodeType === Node.ELEMENT_NODE && (n.textContent || '').trim().length === 0) {
            wrapper.removeChild(n);
          } else {
            break;
          }
        }

        // Insert a single space after the tag if needed
        const space = document.createTextNode(' ');
        tag.parentNode.insertBefore(space, tag.nextSibling);

        // Insert opening parenthesis
        const open = document.createTextNode('(');
        space.parentNode.insertBefore(open, space.nextSibling);

        // Ensure trailing period inside the wrapper
        // Place wrapper after '('
        open.parentNode.insertBefore(wrapper, open.nextSibling);

        // Closing parenthesis and trailing period outside
        const close = document.createTextNode(')');
        wrapper.parentNode.insertBefore(close, wrapper.nextSibling);
        const dot = document.createTextNode('.');
        close.parentNode.insertBefore(dot, close.nextSibling);

        cap.dataset.algCaptionWrapped = '1';
      });
    } catch (e) {}

    // 1f) Ensure long text/math doesn't run under right-edge comments: measure and pad lines
    try {
      function adjustAlgCommentPadding(root) {
        const scope = root || document;
        scope.querySelectorAll('.ltx_float.ltx_float_algorithm .ltx_listingline').forEach((line) => {
          // Reset previous padding
          line.style.paddingRight = '';
          const rightComment = line.querySelector('.ltx_text[style*="float:right"]');
          if (!rightComment) return;
          // On mobile, comments wrap underneath; no padding needed
          if (window.matchMedia && window.matchMedia('(max-width: 600px)').matches) {
            return;
          }
          const cRect = rightComment.getBoundingClientRect();
          // Compute needed padding so text doesn't under-run the comment
          const overlapWidth = Math.max(0, (cRect.width + 8));
          line.style.paddingRight = overlapWidth + 'px';
        });
      }

      adjustAlgCommentPadding();
      window.addEventListener('resize', () => adjustAlgCommentPadding());
      // Re-adjust after fonts/math render settles
      setTimeout(() => adjustAlgCommentPadding(), 100);
      setTimeout(() => adjustAlgCommentPadding(), 300);
      setTimeout(() => adjustAlgCommentPadding(), 800);
    } catch (e) {}

    // 1d) Fallback: plain tabular without wrapping figure (link to nearest labeled container or self)
    try {
      document.querySelectorAll('.ltx_tabular').forEach((tbl, idx) => {
        if (tbl.querySelector('.heading-anchor')) return;
        const container = tbl.closest('.ltx_table[id], .ltx_figure[id]');
        let targetId = container && container.id;
        if (!targetId) { if (!tbl.id) tbl.id = `tabular-${idx + 1}`; targetId = tbl.id; }
        const a = document.createElement('a');
        a.href = `#${targetId}`;
        a.className = 'heading-anchor';
        a.setAttribute('aria-label', 'Copy link to this table');
        a.textContent = '🔗';
        tbl.insertBefore(a, tbl.firstChild);
      });
    } catch (e) {}

    // 2) Remove per-equation copy link toolbars if any were injected previously
    try { document.querySelectorAll('.eq-toolbar').forEach((n) => n.remove()); } catch (e) {}

    // 3) Footnote hover popovers (robust to nested markup)
    try {
      let currentPop;
      function removePop() { if (currentPop) { currentPop.remove(); currentPop = null; } }
      document.addEventListener('click', (e) => { if (!e.target.closest('.ltx_note')) removePop(); });
      document.querySelectorAll('.ltx_note.ltx_role_footnote').forEach((fn) => {
        const mark = fn.querySelector('.ltx_note_mark');
        const content = fn.querySelector('.ltx_note_content') || fn.querySelector('.ltx_note_outer .ltx_note_content');
        if (!mark || !content) return;
        const trigger = mark.closest('a, sup, span') || mark.parentElement; if (!trigger) return;
        trigger.style.cursor = 'help';
        trigger.addEventListener('mouseenter', () => {
          removePop();
          const pop = document.createElement('div');
          pop.className = 'footnote-pop';
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
    } catch (e) {}

    // 4) Smooth scroll for internal refs
    try {
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
    } catch (e) {}
  });
})();


