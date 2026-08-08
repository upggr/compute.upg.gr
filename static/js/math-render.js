/**
 * Render LaTeX in .tex / [data-tex] nodes via KaTeX.
 * Exposes window.UPGMath.typeset(root?) for dynamic content (labs, tabs).
 */
(function () {
  function whenKatex(cb) {
    if (window.katex) {
      cb();
      return;
    }
    var tries = 0;
    var timer = setInterval(function () {
      tries += 1;
      if (window.katex || tries > 80) {
        clearInterval(timer);
        if (window.katex) cb();
      }
    }, 50);
  }

  function renderNode(el) {
    if (!window.katex || !el || el.getAttribute('data-math-rendered') === '1') {
      return;
    }
    var raw = (el.getAttribute('data-tex') || el.textContent || '').trim();
    if (!raw) return;
    var displayMode =
      el.classList.contains('tex-display') ||
      el.getAttribute('data-display') === '1';
    try {
      window.katex.render(raw, el, {
        throwOnError: false,
        displayMode: displayMode,
        output: 'html',
        trust: false,
        strict: 'ignore',
      });
      el.setAttribute('data-math-rendered', '1');
      el.classList.add('tex-rendered');
    } catch (err) {
      console.warn('KaTeX render failed', raw, err);
    }
  }

  function typeset(root) {
    whenKatex(function () {
      var scope = root || document;
      scope.querySelectorAll('.tex, [data-tex]').forEach(function (el) {
        if (el.getAttribute('data-math-dynamic') === '1') {
          el.removeAttribute('data-math-rendered');
        }
        renderNode(el);
      });
    });
  }

  function boot() {
    typeset(document);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', boot);
  } else {
    boot();
  }

  window.addEventListener('load', function () {
    typeset(document);
  });

  window.UPGMath = { typeset: typeset, renderNode: renderNode };
})();
