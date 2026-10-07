// jab: one file, two roles. In a page (content script) it finds the focusable controls, handles Ctrl+Q and draws the badges; in the extension's service worker it ranks the controls with the LightGBM model in ranker.json.
// Nothing is cached per page: every press scores the page as it is right now.
if (self.document) {
  window.__tab = (() => {
    const SEL = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[contenteditable=""],[contenteditable=true],[tabindex]:not([tabindex="-1"]),[role=button],[role=link],[role=textbox],[role=combobox],[role=searchbox],[role=checkbox],[role=tab],[role=menuitem]';
    const BOX = 'form,dialog,[role=dialog],[role=search],nav,header,footer,aside,main,section,fieldset,[role=banner],[role=navigation],[role=contentinfo]';
    const cut = (s, n) => Array.from(s).slice(0, n).join('');   // by code point: slicing a string can split an emoji into a lone surrogate
    const text = (e, n = 60) => cut((e?.innerText ?? e?.textContent ?? '').replace(/\s+/g, ' ').trim(), n);
    const alive = (e) => !e.disabled && !e.inert && e.getAttribute('aria-hidden') !== 'true' && e.checkVisibility({ visibilityProperty: true }) && e.getClientRects().length > 0;
    const label = (e) => e.getAttribute('aria-label') || text(e.labels?.[0]) || text(document.getElementById(e.getAttribute('aria-labelledby'))) || e.placeholder || e.title
      || (e.matches('input[type=submit],input[type=button]') ? e.value : text(e)) || e.getAttribute('alt') || e.querySelector('img')?.alt || e.name || '';
    const box = (e) => { const b = e.parentElement?.closest(BOX); return b ? { kind: b.getAttribute('role') || b.tagName.toLowerCase(), title: b.getAttribute('aria-label') || text(b.querySelector('legend,h1,h2,h3,h4'), 40) } : null; };
    const kind = (e) => e.tagName == 'INPUT' ? `input ${e.type}` : e.getAttribute('role') || e.tagName.toLowerCase();
    const flags = (e) => [e.autofocus && 'autofocus', e.required && 'required', e.matches('input,textarea') && e.value && !/^(submit|button|checkbox|radio)$/.test(e.type) && 'filled', e.checked && 'checked'].filter(Boolean);
    const third = (v) => Math.max(0, Math.min(2, Math.floor(3 * v)));
    const place = (r) => `${['top', 'middle', 'bottom'][third((r.y + scrollY) / Math.max(1, document.documentElement.scrollHeight))]}-${['left', 'center', 'right'][third(r.x / innerWidth)]}`;
    const size = (r) => r.width * r.height > 40000 ? 'large' : r.width * r.height > 6000 ? 'medium' : 'small';
    const q = (v) => Math.floor(Math.round(v) / 10 + 0.5) * 10;   // viewport pixels, to the nearest 10
    const skip = (e) => e.matches('a[href^="#"]') && /skip|jump to|main content/i.test(label(e));

    function collect(max = 80) {
      const els = [...document.querySelectorAll(SEL)].filter(alive).slice(0, max);
      window.__tabEls = els;
      return els.map((e, i) => {
        const r = e.getBoundingClientRect(), b = box(e), up = [];
        for (let p = e.parentElement; p && up.length < 4; p = p.parentElement) up.push(p.getAttribute('role') || p.tagName.toLowerCase());
        let before = ''; for (let p = e; p && !before; p = p.parentElement) before = text(p.previousElementSibling, 20);
        return { i, kind: kind(e), label: cut(label(e), 50), box: b, flags: flags(e), skip: skip(e), name: e.name || e.id || '', ph: e.placeholder || '',
          href: cut(e.getAttribute('href')?.replace(/^https?:\/\/[^/]+/, '') ?? '', 40), place: place(r), size: size(r),
          before, parent: text(e.parentElement, 20), up: up.join('<'), n: els.length,
          view: r.width >= 4 && r.height >= 4 && r.bottom >= 0 && r.top <= innerHeight && r.left <= innerWidth, x: q(r.x), y: q(r.y), w: q(r.width), h: q(r.height) };
      });
    }
    const focusIndex = () => window.__tabEls.indexOf(document.activeElement);
    return { collect, focusIndex };
  })();
  // Press Ctrl+Q: focus where this page most likely wants you next (again for the one after). Ctrl+Q types nothing and Chrome does not use it; every other key is untouched.
  const alive = () => !!chrome.runtime?.id;   // false once the extension has been reloaded or updated: this old copy of the script then does nothing at all, and a refreshed page gets a new one
  let ses;   // ses = {els: our top 3 picks, n: index of the current one}; it lasts until focus leaves our picks
  // The overlay: numbered badges beside the three picks, and a small spinning J while it works. It lives in a closed shadow root, so the page's own CSS and scripts cannot reach it.
  const overlay = (() => {
    const host = document.createElement('div');
    host.id = 'jab-overlay';
    host.style.cssText = 'all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483647;';
    const root = host.attachShadow({ mode: 'closed' });
    const sheet = new CSSStyleSheet();   // a constructed stylesheet: not blocked by a page's content-security policy, unlike an inline <style>
    sheet.replaceSync(`
      .badge { position: fixed; width: 18px; height: 18px; border-radius: 9px; background: #2b8f58; color: #fff; font: 700 11px/18px system-ui, sans-serif; text-align: center; box-shadow: 0 1px 3px rgba(0, 0, 0, .35); }
      .badge.now { background: #1e6e42; outline: 2px solid #fff; }
      .note { position: fixed; right: 16px; bottom: 56px; max-width: 280px; padding: 6px 10px; border-radius: 6px; background: #1f4f35; color: #fff; font: 12px/1.4 system-ui, sans-serif; box-shadow: 0 2px 6px rgba(0, 0, 0, .3); }
      .spinner { position: fixed; right: 16px; bottom: 16px; width: 28px; height: 28px; font: 700 15px/28px system-ui, sans-serif; text-align: center; color: #2b8f58; opacity: 0; transition: opacity .2s; }
      .spinner.on { opacity: .8; }
      .spinner::before { content: ''; position: absolute; inset: 0; box-sizing: border-box; border: 2px solid rgba(43, 143, 88, .22); border-top-color: #2b8f58; border-radius: 50%; animation: turn .9s linear infinite; }
      @keyframes turn { to { transform: rotate(360deg); } }`);
    root.adoptedStyleSheets = [sheet];
    const spinner = document.createElement('div');
    spinner.className = 'spinner';
    spinner.textContent = 'J';
    root.append(spinner);
    let badges = [], delay;

    const place = () => {   // each badge sits on the top-left corner of its control, and the one that has focus is drawn stronger
      if (!ses) return;
      badges.forEach((badge, i) => {
        const r = ses.els[i].getBoundingClientRect();
        badge.style.display = r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth ? '' : 'none';
        badge.style.left = `${Math.max(2, r.left - 8)}px`;
        badge.style.top = `${Math.max(2, r.top - 8)}px`;
        badge.classList.toggle('now', ses.els[i] == document.activeElement);
      });
    };
    for (const event of ['scroll', 'resize', 'focusin']) addEventListener(event, place, { capture: true, passive: true });
    setInterval(place, 300);   // catches layout shifts that fire no event

    return {
      show() {
        document.documentElement.append(host);
        badges.forEach((badge) => badge.remove());
        badges = ses.els.map((_, i) => Object.assign(document.createElement('div'), { className: 'badge', textContent: i + 1 }));
        root.append(...badges);
        host.dataset.badges = badges.length;
        place();
      },
      hide() { badges.forEach((badge) => badge.remove()); badges = []; host.dataset.badges = 0; },
      note(text) {   // a short message, so a failure is never silent
        document.documentElement.append(host);
        const note = Object.assign(document.createElement('div'), { className: 'note', textContent: text });
        root.append(note);
        host.dataset.note = text;
        setTimeout(() => note.remove(), 4000);
      },
      spin(on) {   // appears only if the work takes more than a blink
        clearTimeout(delay);
        if (on) delay = setTimeout(() => { document.documentElement.append(host); spinner.classList.add('on'); host.dataset.spinning = 1; }, 120);
        else { spinner.classList.remove('on'); host.dataset.spinning = 0; }
      },
    };
  })();
  addEventListener('focusout', (e) => { if (ses && !ses.els.includes(e.relatedTarget)) { ses = null; overlay.hide(); } }, true);   // focus left our picks (click elsewhere, window blur, ...): session over
  addEventListener('keydown', async (e) => {
    if (e.code != 'KeyQ' || !e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return;
    if (!alive()) return overlay.note('jab was reloaded: refresh this page to use it.');   // a stale copy of this script says so, and leaves Ctrl+Q alone
    e.preventDefault(); e.stopPropagation();
    if (e.repeat) return;   // holding the keys down is one press
    if (ses) { ses.n = (ses.els.indexOf(document.activeElement) + 1) % ses.els.length; return ses.els[ses.n].focus(); }   // inside a session: walk the stored picks, no recompute
    overlay.spin(true);
    try {
      const all = __tab.collect(), scores = await chrome.runtime.sendMessage({ controls: all, focus: __tab.focusIndex() });
      const top = scores.map((v, i) => [v ?? -Infinity, i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map(([, i]) => __tabEls[i]);   // the best three; the focused control scores -Infinity (null in the message)
      ses = { els: top };
      overlay.show();
      top[0].focus();   // the first press lands on the best pick
    } catch (error) {
      overlay.note(`jab could not answer: ${error.message}`);
    } finally {
      overlay.spin(false);
    }
  }, true);
} else {
  // The ranker: a LightGBM model (ranker.json, made by train.py) scores every control of the page from cheap features (position, size, kind, on screen, words in the label), no neural network.
  // rank(model, controls, focus): controls as jab.js collect() returns them, focus the index of the focused control or -1. Returns one score per control; the focused one gets -Infinity.
  const rank = (() => {
    const PLACE = { top: 0, middle: 1, bottom: 2, left: 0, center: 1, right: 2 }, SIZE = { small: 0, medium: 1, large: 2 }, length = (s) => [...s].length;
    const onScreen = (c) => c.w >= 4 && c.h >= 4 && c.y + c.h >= 0 && c.y <= 900 && c.x <= 1280;
    const guess = (c) => (/^(input|textarea|select)/.test(c.kind) ? 3 : c.kind.startsWith('button') ? 2 : 1) + (c.box?.kind == 'main') + 2 * onScreen(c) - 5 * c.skip - c.i / 100;
    const words = (c) => new Set(`${c.label} ${c.href} ${c.name} ${c.ph}`.toLowerCase().match(/[a-z]{3,}/g) ?? []);
    const features = (model, all, focus) => {   // the same features, in the same order, as train.py
      const controls = all.filter((c) => c.i != focus), shown = controls.filter(onScreen);
      return controls.map((c) => {
        const same = controls.filter((d) => d.kind == c.kind), [vertical, horizontal] = c.place.split('-');
        const f = { same_kind_rank: same.indexOf(c), same_kind_count: same.length, inputs_on_screen: shown.filter((d) => d.kind.startsWith('input')).length, screen_rank: shown.indexOf(c), on_screen_count: shown.length,
          controls_above: shown.filter((d) => d.y < c.y).length, no_parent_text: c.parent == '', before_len: length(c.before), box_titled: (c.box?.title ?? '') != '', depth: c.up.split('<').length,
          kind: model.kinds[c.kind] ?? -1, box: model.boxes[c.box?.kind ?? ''] ?? -1, on_screen: onScreen(c), x: c.x / 1280, y: c.y / 900, w: c.w / 1280, h: c.h / 900,
          vertical: PLACE[vertical], horizontal: PLACE[horizontal], size: SIZE[c.size], skip: c.skip, required: c.flags.includes('required'), autofocus: c.flags.includes('autofocus'), filled: c.flags.includes('filled'),
          index: c.i, index_frac: c.i / Math.max(1, c.n), count: c.n, label_len: length(c.label), has_placeholder: !!c.ph, has_href: !!c.href, nothing_focused: focus < 0,
          index_delta: focus >= 0 ? c.i - focus : 0, guess: guess(c) };
        const here = words(c);
        return [...model.features.map((name) => Number(f[name])), ...model.words.map((w) => (here.has(w) ? 1 : 0))];
      });
    };
    const walk = ([feature, threshold, left, right, leaf], x) => {   // a tree: go left when the feature is <= the threshold; a negative child ~n is leaf n
      if (!feature.length) return leaf[0];
      let node = 0;
      while (node >= 0) node = x[feature[node]] <= threshold[node] ? left[node] : right[node];
      return leaf[~node];
    };
    return (model, controls, focus) => {
      const rows = features(model, controls, focus), scores = controls.map(() => -Infinity);
      controls.filter((c) => c.i != focus).forEach((c, k) => { scores[controls.indexOf(c)] = model.trees.reduce((sum, tree) => sum + walk(tree, rows[k]), 0); });
      return scores;
    };
  })();

  const model = fetch(chrome.runtime.getURL('ranker.json')).then((r) => r.json());   // loaded once per worker start, in milliseconds
  chrome.runtime.onMessage.addListener(({ controls, focus }, _, send) => {
    model.then((m) => send(rank(m, controls, focus).map((v) => (Number.isFinite(v) ? v : null))));   // JSON cannot carry -Infinity
    return true;
  });
}
