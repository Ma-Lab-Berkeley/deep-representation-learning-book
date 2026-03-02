/* chapter.js — Interactive features for make4ht chapter HTML output */

document.addEventListener("DOMContentLoaded", function () {
  "use strict";

  // =========================================================================
  // 1. Heading anchor links (sections, subsections, paragraphs)
  // =========================================================================

  document.querySelectorAll("h2[id], h3[id], h4[id]").forEach(function (h) {
    var targetId = h.id;
    if (!targetId) {
      // Try parent section
      var sec = h.closest("section[id], div[id]");
      if (sec) targetId = sec.id;
    }
    if (!targetId || h.querySelector(".heading-anchor")) return;

    var a = document.createElement("a");
    a.href = "#" + targetId;
    a.className = "heading-anchor";
    a.setAttribute("aria-label", "Copy link to this section");
    a.textContent = "\uD83D\uDD17"; // 🔗
    h.insertBefore(a, h.firstChild);
  });

  // =========================================================================
  // 1a. Theorem environment anchor links
  // =========================================================================

  document.querySelectorAll(".theorem-env[id]").forEach(function (thm) {
    var head = thm.querySelector(".head");
    if (!head || head.querySelector(".heading-anchor")) return;

    var a = document.createElement("a");
    a.href = "#" + thm.id;
    a.className = "heading-anchor";
    a.setAttribute("aria-label", "Copy link to this environment");
    a.textContent = "\uD83D\uDD17";
    head.insertBefore(a, head.firstChild);
  });

  // =========================================================================
  // 1b. Equation anchor links
  // =========================================================================

  document.querySelectorAll(".equation[id], table.equation[id]").forEach(function (eq) {
    if (eq.querySelector(".heading-anchor-outside")) return;

    var a = document.createElement("a");
    a.href = "#" + eq.id;
    a.className = "heading-anchor-outside";
    a.setAttribute("aria-label", "Copy link to this equation");
    a.textContent = "\uD83D\uDD17";
    eq.appendChild(a);
  });

  // =========================================================================
  // 1c. Figure / table / algorithm anchor links
  // =========================================================================

  document
    .querySelectorAll("figure[id], .figure[id], .algorithm-container[id]")
    .forEach(function (el) {
      if (el.querySelector(".heading-anchor")) return;

      var a = document.createElement("a");
      a.href = "#" + el.id;
      a.className = "heading-anchor";
      a.setAttribute("aria-label", "Copy link to this figure/table/algorithm");
      a.textContent = "\uD83D\uDD17";
      el.insertBefore(a, el.firstChild);
    });

  // =========================================================================
  // 2. Footnote hover popovers
  // =========================================================================

  var currentPop = null;

  function removePop() {
    if (currentPop && currentPop.parentNode) {
      currentPop.parentNode.removeChild(currentPop);
    }
    currentPop = null;
  }

  // make4ht with fn-in produces footnotes as inline elements.
  // Look for footnote marks (superscript links to footnotes)
  document
    .querySelectorAll('a.footnote-mark, sup.textsuperscript a[href^="#"]')
    .forEach(function (mark) {
      // Find the footnote content
      var href = mark.getAttribute("href");
      if (!href || !href.startsWith("#")) return;

      var fnId = href.slice(1);
      var fnContent = document.getElementById(fnId);
      if (!fnContent) return;

      var trigger = mark.closest("sup") || mark;

      trigger.addEventListener("mouseenter", function () {
        removePop();

        var pop = document.createElement("div");
        pop.className = "footnote-pop";

        // Clone content and clean up
        var clone = fnContent.cloneNode(true);
        // Remove back-references
        clone.querySelectorAll("a.footnote-backref").forEach(function (a) {
          a.remove();
        });
        // Remove leading superscript numbers
        var firstSup = clone.querySelector("sup");
        if (firstSup && /^\d+$/.test(firstSup.textContent.trim())) {
          firstSup.remove();
        }

        pop.innerHTML = clone.innerHTML || clone.textContent;
        document.body.appendChild(pop);

        var rect = trigger.getBoundingClientRect();
        var popRect = pop.getBoundingClientRect();
        pop.style.position = "absolute";
        pop.style.top = window.scrollY + rect.bottom + 6 + "px";
        pop.style.left =
          Math.min(
            rect.left,
            window.innerWidth - popRect.width - 12
          ) + "px";

        currentPop = pop;
      });

      trigger.addEventListener("mouseleave", function () {
        setTimeout(removePop, 120);
      });
    });

  // Global click dismisses popover
  document.addEventListener("click", function (e) {
    if (currentPop && !e.target.closest(".footnote-pop")) {
      removePop();
    }
  });

  // =========================================================================
  // 3. Smooth scroll for internal links
  // =========================================================================

  document.querySelectorAll('a[href^="#"]').forEach(function (a) {
    a.addEventListener("click", function (e) {
      var href = a.getAttribute("href");
      if (!href || href === "#") return;

      var target = document.querySelector(href);
      if (!target) return;

      e.preventDefault();
      history.pushState(null, "", href);
      target.scrollIntoView({ behavior: "smooth", block: "start" });
    });
  });

  // =========================================================================
  // 4. Algorithm caption formatting
  // =========================================================================

  document
    .querySelectorAll(".algorithm-container .caption")
    .forEach(function (cap) {
      if (cap.dataset.algCaptionWrapped) return;

      // Find the algorithm number/tag
      var tag = cap.querySelector(".id");
      if (!tag) return;

      // Collect remaining nodes after the tag
      var nodes = [];
      var sibling = tag.nextSibling;
      while (sibling) {
        nodes.push(sibling);
        sibling = sibling.nextSibling;
      }

      if (nodes.length === 0) return;

      // Wrap caption text in parens
      var wrapper = document.createElement("span");
      wrapper.className = "alg-caption-text";
      nodes.forEach(function (n) {
        wrapper.appendChild(n);
      });

      // Trim leading/trailing whitespace
      if (wrapper.firstChild && wrapper.firstChild.nodeType === 3) {
        wrapper.firstChild.textContent =
          wrapper.firstChild.textContent.replace(/^\s+/, "");
      }
      if (wrapper.lastChild && wrapper.lastChild.nodeType === 3) {
        wrapper.lastChild.textContent = wrapper.lastChild.textContent
          .replace(/\.\s*$/, "")
          .replace(/\s+$/, "");
      }

      tag.insertAdjacentText("afterend", " (");
      tag.nextSibling.after(wrapper);
      wrapper.insertAdjacentText("afterend", ").");

      cap.dataset.algCaptionWrapped = "1";
    });

  // =========================================================================
  // 5. Re-scroll to hash target after MathJax renders
  // =========================================================================

  if (window.location.hash) {
    function scrollToHash() {
      try {
        var el = document.querySelector(window.location.hash);
        if (el) el.scrollIntoView({ behavior: "smooth", block: "start" });
      } catch (_) {}
    }
    // After MathJax finishes typesetting, content positions shift
    if (window.MathJax && window.MathJax.startup) {
      window.MathJax.startup.promise.then(scrollToHash);
    } else {
      // Fallback: wait for layout to settle
      setTimeout(scrollToHash, 800);
    }
  }
});
