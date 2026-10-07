// jab: one file, two roles. In a page (content script) it finds the focusable controls, handles Ctrl+Q and draws the badges; in the extension's service worker it ranks the controls with the LightGBM model in ranker.json.
// Nothing is cached per page: every press scores the page as it is right now.
if (self.document) {
  const SEL = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[contenteditable=""],[contenteditable=true],[tabindex]:not([tabindex="-1"]),[role=button],[role=link],[role=textbox],[role=combobox],[role=searchbox],[role=checkbox],[role=tab],[role=menuitem]';
  const BOX = 'form,dialog,[role=dialog],[role=search],nav,header,footer,aside,main,section,fieldset,[role=banner],[role=navigation],[role=contentinfo]';
  const SUBMIT = 'button:not([type=button],[type=reset]),input[type=submit],input[type=image]';
  const EDITABLE = '[contenteditable=""],[contenteditable=true],[role=textbox],[role=searchbox],[role=combobox]';   // chat boxes and rich editors are not inputs
  const TYPED = 'textarea,input:not([type=checkbox],[type=radio],[type=submit],[type=button],[type=reset],[type=image],[type=hidden],[type=file],[type=range],[type=color])';   // controls you type into
  const cut = (s, n) => Array.from(s).slice(0, n).join('');   // by code point: slicing a string can split an emoji into a lone surrogate
  const text = (e, n = 60) => cut((e?.innerText ?? e?.textContent ?? '').replace(/\s+/g, ' ').trim(), n);
  const alive = (e) => !e.disabled && !e.inert && e.getAttribute('aria-hidden') !== 'true' && e.checkVisibility({ visibilityProperty: true }) && e.getClientRects().length > 0;
  const label = (e) => e.getAttribute('aria-label') || text(e.labels?.[0]) || text(document.getElementById(e.getAttribute('aria-labelledby'))) || e.placeholder || e.title
    || (e.matches('input[type=submit],input[type=button]') ? e.value : text(e)) || e.getAttribute('alt') || e.querySelector('img')?.alt || e.name || '';
  const q = (v) => Math.floor(Math.round(v) / 10 + 0.5) * 10;   // viewport pixels, to the nearest 10
  let els = [], ses;   // els: the controls last listed; ses = our top 4 picks, which lasts until focus leaves them

  const collect = () => {   // every focusable, visible control, each described by the facts train.py's features use
    const all = [...document.querySelectorAll(SEL)].filter(alive), form = document.activeElement.closest('form');   // every focusable, visible control
    if (all.length <= 5000) els = all;
    else {   // a safety valve for endless pages (2 of 1,400 pages tried had more): the control you are in and its form, then what is on screen, then the rest in page order
      const shown = (e) => { const r = e.getBoundingClientRect(); return r.bottom >= 0 && r.top <= innerHeight && r.right >= 0 && r.left <= innerWidth; };
      const keep = new Set([...all.filter((e) => e == document.activeElement || form?.contains(e)), ...all.filter(shown), ...all].slice(0, 5000));
      els = all.filter((e) => keep.has(e));
    }
    return els.map((e, i) => {
      const r = e.getBoundingClientRect(), b = e.parentElement?.closest(BOX), f = e.form ?? e.closest('form'), st = getComputedStyle(e);
      const fill = st.backgroundColor.match(/[\d.]+/g)?.map(Number) ?? [0, 0, 0, 0], [red, green, blue] = fill, alpha = fill[3] ?? 1;   // how it looks: filled or not, how colourful, how round, how bold
      let before = ''; for (let p = e; p && !before; p = p.parentElement) before = text(p.previousElementSibling, 20);
      return { i, n: els.length, form: f ? [...document.forms].indexOf(f) : -1, action: cut((f?.getAttribute('action') ?? '').replace(/^https?:\/\/[^/]+/, ''), 30), submit: !!f && e.matches(SUBMIT), filled: e.matches(TYPED) ? !!e.value : e.matches(EDITABLE) && !!e.innerText.trim(), required: e.required === true, fill: alpha, sat: alpha ? (Math.max(red, green, blue) - Math.min(red, green, blue)) / (Math.max(red, green, blue) || 1) : 0, lum: alpha ? (0.299 * red + 0.587 * green + 0.114 * blue) / 255 : 0,
        round: Math.min(1, parseFloat(st.borderTopLeftRadius) / Math.max(1, Math.min(r.width, r.height) / 2) || 0), bold: (parseInt(st.fontWeight) || 400) / 1000, pointer: st.cursor == 'pointer', icon: !!e.querySelector('svg,img'), kind: e.tagName == 'INPUT' ? `input ${e.type}` : e.getAttribute('role') || e.tagName.toLowerCase(), label: cut(label(e), 50), box: b ? b.getAttribute('role') || b.tagName.toLowerCase() : '',
        name: e.name || e.id || '', ph: e.placeholder || '', href: cut(e.getAttribute('href')?.replace(/^https?:\/\/[^/]+/, '') ?? '', 40), before, parent: text(e.parentElement, 20), x: q(r.x), y: q(r.y), w: q(r.width), h: q(r.height) };
    });
  };

  // The overlay: numbered badges beside the four picks, a small spinning J while it works, and short notes so a failure is never silent. It lives in a closed shadow root, so the page's own CSS and scripts cannot reach it.
  const host = Object.assign(document.createElement('div'), { id: 'jab-overlay' }), root = host.attachShadow({ mode: 'closed' }), sheet = new CSSStyleSheet();   // a constructed stylesheet is not blocked by a page's CSP
  host.style.cssText = 'all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483647;';
  sheet.replaceSync(`
    .badge { position: fixed; width: 18px; height: 18px; border-radius: 9px; background: #2b8f58; color: #fff; font: 700 11px/18px system-ui, sans-serif; text-align: center; box-shadow: 0 1px 3px rgba(0, 0, 0, .35); }
    .badge.now { background: #1e6e42; outline: 2px solid #fff; }
    .note { position: fixed; right: 16px; bottom: 56px; max-width: 280px; padding: 6px 10px; border-radius: 6px; background: #1f4f35; color: #fff; font: 12px/1.4 system-ui, sans-serif; box-shadow: 0 2px 6px rgba(0, 0, 0, .3); }
    .spinner { position: fixed; right: 16px; bottom: 16px; width: 28px; height: 28px; font: 700 15px/28px system-ui, sans-serif; text-align: center; color: #2b8f58; opacity: 0; transition: opacity .2s; }
    .spinner.on { opacity: .8; }
    .spinner::before { content: ''; position: absolute; inset: 0; box-sizing: border-box; border: 2px solid rgba(43, 143, 88, .22); border-top-color: #2b8f58; border-radius: 50%; animation: turn .9s linear infinite; }
    @keyframes turn { to { transform: rotate(360deg); } }`);
  root.adoptedStyleSheets = [sheet];
  const spinner = Object.assign(document.createElement('div'), { className: 'spinner', textContent: 'J' });
  root.append(spinner);
  const draw = (cls, text) => { document.documentElement.append(host); const d = Object.assign(document.createElement('div'), { className: cls, textContent: text }); root.append(d); return d; };
  let badges = [], delay;
  const show = () => {   // each badge sits on the top-left corner of its control; the one that has focus is drawn stronger
    badges.forEach((badge, i) => {
      const r = ses.els[i].getBoundingClientRect();
      badge.style.cssText = `display: ${r.bottom > 0 && r.top < innerHeight && r.right > 0 && r.left < innerWidth ? '' : 'none'}; left: ${Math.max(2, r.left - 8)}px; top: ${Math.max(2, r.top - 8)}px`;
      badge.classList.toggle('now', ses.els[i] == document.activeElement);
    });
  };
  const note = (message) => { const n = draw('note', message); setTimeout(() => n.remove(), 4000); };
  for (const event of ['scroll', 'resize', 'focusin']) addEventListener(event, () => ses && show(), { capture: true, passive: true });
  setInterval(() => ses && show(), 300);   // catches layout shifts that fire no event

  addEventListener('focusout', (e) => { if (ses && !ses.els.includes(e.relatedTarget)) { ses = null; badges.forEach((badge) => badge.remove()); } }, true);   // focus left our picks (click elsewhere, window blur, ...): session over
  addEventListener('keydown', async (e) => {   // Ctrl+Q: focus where this page most likely wants you next (again for the one after). It types nothing and Chrome does not use it; every other key is untouched.
    if (e.code != 'KeyQ' || !e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return;
    if (!chrome.runtime?.id) return note('jab was reloaded: refresh this page to use it.');   // a stale copy of this script (after a reload or update) says so, and leaves Ctrl+Q alone
    e.preventDefault(); e.stopPropagation();
    if (e.repeat) return;   // holding the keys down is one press
    if (ses) return ses.els[(ses.els.indexOf(document.activeElement) + 1) % ses.els.length].focus();   // inside a session: walk the stored picks, no recompute
    delay = setTimeout(() => spinner.classList.add('on'), 120);   // appears only if the work takes more than a blink
    try {
      const scores = await chrome.runtime.sendMessage({ controls: collect(), focus: els.indexOf(document.activeElement), page: { host: location.hostname, path: location.pathname, title: document.title, desc: document.querySelector('meta[name=description]')?.content ?? '' } });
      ses = { els: scores.map((v, i) => [v ?? -Infinity, i]).sort((a, b) => b[0] - a[0]).slice(0, 4).map(([, i]) => els[i]) };   // the best four; the focused control has no score (null)
      badges = ses.els.map((_, i) => draw('badge', i + 1));
      show();
      ses.els[0].focus();   // the first press lands on the best pick
    } catch (error) {
      note(`jab could not answer: ${error.message}`);
    } finally {
      clearTimeout(delay); spinner.classList.remove('on');
    }
  }, true);
} else {
  // The ranker: a LightGBM model (ranker.json, made by train.py: a ranker and a pick-or-not classifier, averaged) scores every control of the page from cheap features (position, kind, what is on screen, words in the label), no neural network.
  // rank(model, controls, focus): controls as collect() returns them, focus the index of the focused control or -1. Returns one score per control; null for the focused one.
  const length = (s) => [...s].length, onScreen = (c) => c.w >= 4 && c.h >= 4 && c.y + c.h >= 0 && c.y <= 900 && c.x <= 1280;
  const isField = (c) => c.kind == 'textarea' || /^(input (text|search|email|password|tel|url|number)|textbox|searchbox|combobox)$/.test(c.kind);   // controls you type into
  const guess = (c) => (/^(input|textarea|select)/.test(c.kind) ? 3 : c.kind.startsWith('button') ? 2 : 1) + (c.box == 'main') + 2 * onScreen(c) - c.i / 100;
  const walk = ([feature, threshold, left, right, leaf], x) => {   // a tree: go left when the feature is <= the threshold; a negative child ~n is leaf n
    if (!feature.length) return leaf[0];
    let node = 0;
    while (node >= 0) node = x[feature[node]] <= threshold[node] ? left[node] : right[node];
    return leaf[~node];
  };
  const rank = (model, all, focus, page) => {   // the same features, in the same order, as train.py; every count is computed once per page, so thousands of controls stay cheap
    const controls = all.filter((c) => c.i != focus), shown = controls.filter(onScreen), focused = all.find((c) => c.i == focus);
    const colour = (c) => (c.fill >= 0.5 ? c.sat : 0), satRank = new Map([...shown].sort((a, b) => colour(b) - colour(a)).map((c, r) => [c.i, r]));   // 0 = the most colourful control on screen
    const shownY = shown.map((c) => c.y).sort((a, b) => a - b), shownRank = new Map(shown.map((c, r) => [c.i, r])), inputsShown = shown.filter((c) => c.kind.startsWith('input')).length;
    const kindCount = new Map(), kindSeen = new Map(), formFields = new Map(), firstField = new Map();
    for (const c of controls) kindCount.set(c.kind, (kindCount.get(c.kind) ?? 0) + 1);
    const fields = all.filter((c) => isField(c) && c.form >= 0);   // the typing fields that sit in a form
    for (const c of fields) { formFields.set(c.form, (formFields.get(c.form) ?? 0) + 1); if (!firstField.has(c.form)) firstField.set(c.form, c.i); }
    const below = (y) => { let lo = 0, hi = shownY.length; while (lo < hi) { const mid = (lo + hi) >> 1; if (shownY[mid] < y) lo = mid + 1; else hi = mid; } return lo; };   // how many on-screen controls are above y
    const emptyLeft = focused && focused.form >= 0 ? fields.filter((c) => !c.filled && c.form == focused.form && c.i != focus).length : 0;   // in the focused control's form
    const nf = model.features.length, x = new Float64Array(nf + model.words.length), column = new Map(model.words.map((w, j) => [w, nf + j]));   // one row, reused for every control
    for (const w of new Set(`${page.host} ${page.path} ${page.title} ${page.desc}`.toLowerCase().match(/[a-z]{3,}/g))) { const j = column.get('p:' + w); if (j !== undefined) x[j] = 1; }   // the page's words stay set
    const raw = model.models.map(() => new Float64Array(controls.length));
    controls.forEach((c, k) => {
      const sameRank = kindSeen.get(c.kind) ?? 0; kindSeen.set(c.kind, sameRank + 1);
      const inFocusForm = !!focused && c.form >= 0 && c.form == focused.form;
      const f = { same_kind_rank: sameRank, same_kind_count: kindCount.get(c.kind), inputs_on_screen: inputsShown, screen_rank: shownRank.get(c.i) ?? -1, on_screen_count: shown.length,
        controls_above: below(c.y), no_parent_text: c.parent == '', before_len: length(c.before), kind: model.kinds[c.kind] ?? -1, box: model.boxes[c.box] ?? -1,
        x: c.x / 1280, y: c.y / 900, w: c.w / 1280, h: c.h / 900, index: c.i, index_frac: c.i / Math.max(1, c.n), count: c.n, label_len: length(c.label), nothing_focused: focus < 0,
        index_delta: focus >= 0 ? c.i - focus : 0, guess: guess(c),
        required: c.required, is_submit: c.submit, dismissive: /^(clear|reset|cancel|close|dismiss|decline|no thanks|skip|back)\b/.test(c.label.toLowerCase()), first_field: isField(c) && c.form >= 0 && firstField.get(c.form) == c.i,
        form_fields: c.form >= 0 ? formFields.get(c.form) ?? 0 : 0, focus_filled: !!focused?.filled, in_focus_form: inFocusForm,
        after_focus: inFocusForm && c.i > focus, empty_fields_left: emptyLeft, empty_field: isField(c) && !c.filled,
        focus_dx: focused ? (c.x - focused.x) / 1280 : 0, focus_dy: focused ? (c.y - focused.y) / 900 : 0, focus_dist: focused ? Math.hypot(c.x - focused.x, c.y - focused.y) / 1000 : 0,
        submit_like: /\b(send|submit|go|search|post|reply|comment|save|continue|next|sign in|log in|login|apply|ask)\b/.test(c.label.toLowerCase()),
        fill: c.fill, sat: c.sat, lum: c.lum, round: c.round, bold: c.bold, pointer: c.pointer, icon: c.icon, sat_rank: satRank.get(c.i) ?? -1 };
      model.features.forEach((name, j) => { x[j] = Number(f[name]); });
      const active = [];
      for (const w of new Set(`${c.label} ${c.href} ${c.name} ${c.ph} ${c.action}`.toLowerCase().match(/[a-z]{3,}/g))) { const j = column.get('w:' + w); if (j !== undefined) { x[j] = 1; active.push(j); } }
      model.models.forEach((trees, m) => { let sum = 0; for (const tree of trees) sum += walk(tree, x); raw[m][k] = sum; });
      for (const j of active) x[j] = 0;
    });
    const standardised = raw.map((r) => {   // each model's scores, standardised within the page
      const mean = r.reduce((a, b) => a + b, 0) / r.length, sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / r.length) + 1e-9;
      return Array.from(r, (v) => (v - mean) / sd);
    });
    let k = 0;
    return all.map((c) => (c.i == focus ? null : (k++, standardised.reduce((sum, z) => sum + z[k - 1], 0))));   // the models' standardised scores added; the focused control has none
  };
  const model = fetch(chrome.runtime.getURL('ranker.json')).then((r) => r.json());   // loaded once per worker start, in milliseconds
  chrome.runtime.onMessage.addListener(({ controls, focus, page }, _, send) => { model.then((m) => send(rank(m, controls, focus, page))); return true; });
}
