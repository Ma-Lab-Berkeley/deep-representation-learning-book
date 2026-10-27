/* chapter.js — Interactive features for make4ht chapter HTML output */

document.addEventListener("DOMContentLoaded", function () {
  "use strict";

  function prependAnchor(host, targetId, ariaLabel, className) {
    if (!host || !targetId) return;
    if (host.querySelector(".heading-anchor, .heading-anchor-outside")) return;

    var anchor = document.createElement("a");
    anchor.href = "#" + targetId;
    anchor.className = className || "heading-anchor";
    anchor.setAttribute("aria-label", ariaLabel);
    anchor.textContent = "\uD83D\uDD17"; // 🔗
    host.insertBefore(anchor, host.firstChild);
  }

  function scrollToHashTarget() {
    if (!window.location.hash) return;
    try {
      var target = document.querySelector(window.location.hash);
      if (target) {
        target.scrollIntoView({ behavior: "smooth", block: "start" });
      }
    } catch (_) {}
  }

  function afterMathJaxLayout(callback) {
    if (window.MathJax && window.MathJax.startup) {
      window.MathJax.startup.promise.then(callback);
    } else {
      setTimeout(callback, 800);
    }
  }

  function trimEdgeText(node, pattern) {
    if (!node || node.nodeType !== 3) return;
    node.textContent = node.textContent.replace(pattern, "");
  }

  // =========================================================================
  // 1. Heading anchor links (sections, subsections, paragraphs)
  // =========================================================================

  document.querySelectorAll("h2[id], h3[id], h4[id]").forEach(function (heading) {
    prependAnchor(
      heading,
      heading.id,
      "Copy link to this section",
      "heading-anchor"
    );
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

  document
    .querySelectorAll('a.footnote-mark, sup.textsuperscript a[href^="#"]')
    .forEach(function (mark) {
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

        var clone = fnContent.cloneNode(true);
        clone.querySelectorAll("a.footnote-backref").forEach(function (a) {
          a.remove();
        });
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
  // 2b. Citation hover popovers
  // =========================================================================
  // Citations link to bibliography entries on a separate page
  // (e.g. Chx8.html#X0-Wiener-1942). On hover, fetch that page once and show
  // the matching entry inline so readers don't have to leave the chapter.

  var bibPages = {};
  var citePop = null;
  var citeHideTimer = null;
  var citeShowTimer = null;

  function loadBibPage(url) {
    if (!bibPages[url]) {
      bibPages[url] = fetch(url)
        .then(function (r) {
          if (!r.ok) throw new Error(r.status);
          return r.text();
        })
        .then(function (html) {
          return new DOMParser().parseFromString(html, "text/html");
        })
        .catch(function () {
          delete bibPages[url];
          return null;
        });
    }
    return bibPages[url];
  }

  function findBibEntry(doc, id) {
    var dt = doc && doc.getElementById(id);
    if (!dt) return null;
    var dd = dt.nextElementSibling;
    if (!dd || dd.tagName !== "DD") return null;
    return { label: dt.textContent.trim(), body: dd };
  }

  function removeCitePop() {
    clearTimeout(citeHideTimer);
    if (citePop && citePop.parentNode) {
      citePop.parentNode.removeChild(citePop);
    }
    citePop = null;
  }

  function scheduleCiteHide() {
    clearTimeout(citeShowTimer);
    clearTimeout(citeHideTimer);
    citeHideTimer = setTimeout(removeCitePop, 200);
  }

  function showCitePop(trigger, entry) {
    removeCitePop();

    var pop = document.createElement("div");
    pop.className = "footnote-pop cite-pop";

    var label = document.createElement("span");
    label.className = "cite-pop-label";
    label.textContent = entry.label;
    pop.appendChild(label);

    var clone = entry.body.cloneNode(true);
    clone.querySelectorAll("[id]").forEach(function (el) {
      el.removeAttribute("id");
    });
    while (clone.firstChild) {
      pop.appendChild(clone.firstChild);
    }

    pop.addEventListener("mouseenter", function () {
      clearTimeout(citeHideTimer);
    });
    pop.addEventListener("mouseleave", scheduleCiteHide);

    document.body.appendChild(pop);

    var rect = trigger.getBoundingClientRect();
    var popRect = pop.getBoundingClientRect();
    var top = window.scrollY + rect.bottom + 6;
    if (
      rect.bottom + 6 + popRect.height > window.innerHeight &&
      rect.top - 6 - popRect.height > 0
    ) {
      top = window.scrollY + rect.top - 6 - popRect.height;
    }
    pop.style.top = top + "px";
    pop.style.left =
      Math.max(
        12,
        Math.min(
          window.scrollX + rect.left,
          window.scrollX + window.innerWidth - popRect.width - 12
        )
      ) + "px";

    citePop = pop;
  }

  document.querySelectorAll('a[href*="#X0-"]').forEach(function (link) {
    var url;
    try {
      url = new URL(link.getAttribute("href"), window.location.href);
    } catch (_) {
      return;
    }
    var id = decodeURIComponent(url.hash.slice(1));
    var samePage = url.pathname === window.location.pathname;
    url.hash = "";

    link.addEventListener("mouseenter", function () {
      clearTimeout(citeHideTimer);
      clearTimeout(citeShowTimer);
      citeShowTimer = setTimeout(function () {
        var docPromise = samePage
          ? Promise.resolve(document)
          : loadBibPage(url.href);
        docPromise.then(function (doc) {
          var entry = findBibEntry(doc, id);
          if (entry && link.matches(":hover")) showCitePop(link, entry);
        });
      }, 150);
    });

    link.addEventListener("mouseleave", scheduleCiteHide);
  });

  document.addEventListener("click", function (e) {
    if (citePop && !e.target.closest(".cite-pop")) {
      removeCitePop();
    }
  });

  document.addEventListener("keydown", function (e) {
    if (e.key === "Escape") removeCitePop();
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

      var tag = cap.querySelector(".id");
      if (!tag) return;

      var wrapper = document.createElement("span");
      wrapper.className = "alg-caption-text";
      while (tag.nextSibling) {
        wrapper.appendChild(tag.nextSibling);
      }
      if (!wrapper.firstChild) return;

      trimEdgeText(wrapper.firstChild, /^\s+/);
      trimEdgeText(wrapper.lastChild, /\.\s*$|\s+$/);

      tag.insertAdjacentText("afterend", " (");
      tag.nextSibling.after(wrapper);
      wrapper.insertAdjacentText("afterend", ").");

      cap.dataset.algCaptionWrapped = "1";
    });

  // =========================================================================
  // 5. Re-scroll to hash target after MathJax renders
  // =========================================================================

  if (window.location.hash) {
    afterMathJaxLayout(scrollToHashTarget);
  }
});
