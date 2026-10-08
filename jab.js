// jab: one file, two roles. In a page (content script) it finds the focusable controls, handles Ctrl+Q and draws the badges;
// in the extension's service worker it ranks the controls with the LightGBM models in ranker.json.
// Nothing is cached per page: every press scores the page as it is right now.
if (self.document) {
  const SEL =
    'a[href],button,input:not([type=hidden]),select,textarea,summary,[contenteditable=""],[contenteditable=true],[tabindex]:not([tabindex="-1"]),[role=button],[role=link],[role=textbox],[role=combobox],[role=searchbox],[role=checkbox],[role=tab],[role=menuitem]';
  const BOX =
    "form,dialog,[role=dialog],[role=alertdialog],[aria-modal=true],[role=search],nav,header,footer,aside,main,section,fieldset,[role=banner],[role=navigation],[role=contentinfo],[role=menubar],[role=toolbar],[role=menu],[role=tablist]";
  const SUBMIT = "button:not([type=button],[type=reset]),input[type=submit],input[type=image]";
  // controls you type into; chat boxes and rich editors are not inputs
  const TYPED =
    "textarea,input:not([type=checkbox],[type=radio],[type=submit],[type=button],[type=reset],[type=image],[type=hidden],[type=file],[type=range],[type=color])";
  const EDITABLE = '[contenteditable=""],[contenteditable=true],[role=textbox],[role=searchbox],[role=combobox]';
  const DIALOG = "dialog,[role=dialog],[role=alertdialog],[aria-modal=true]";

  // by code point: slicing a string can split an emoji into a lone surrogate
  const cut = (s, n) => Array.from(s).slice(0, n).join("");
  const text = (e, n = 60) => cut((e?.innerText ?? e?.textContent ?? "").replace(/\s+/g, " ").trim(), n);
  const path = (url) => url.replace(/^https?:\/\/[^/]+/, "");
  const alive = (e) =>
    !e.disabled &&
    !e.inert &&
    e.getAttribute("aria-hidden") !== "true" &&
    e.checkVisibility({ visibilityProperty: true }) &&
    e.getClientRects().length > 0;
  const label = (e) =>
    e.getAttribute("aria-label") ||
    text(e.labels?.[0]) ||
    (e.getAttribute("aria-labelledby") ?? "")
      .split(/\s+/)
      .map((id) => text(document.getElementById(id)))
      .join(" ")
      .trim() ||
    e.placeholder ||
    e.title ||
    (e.matches("input[type=submit],input[type=button]") ? e.value : text(e)) ||
    e.getAttribute("alt") ||
    e.querySelector("img")?.alt ||
    e.name ||
    "";
  const inView = (r) => r.bottom >= 0 && r.top <= innerHeight && r.right >= 0 && r.left <= innerWidth;
  const q = (v) => Math.floor(Math.round(v) / 10 + 0.5) * 10; // viewport pixels, to the nearest 10
  // every control, including those inside open shadow roots, in page order
  const deep = (root, out = []) => {
    for (const e of root.querySelectorAll("*")) {
      if (e.matches(SEL)) out.push(e);
      if (e.shadowRoot) deep(e.shadowRoot, out);
    }
    return out;
  };
  // the focused element, even inside a web component
  const active = () => {
    let a = document.activeElement;
    while (a?.shadowRoot?.activeElement) a = a.shadowRoot.activeElement;
    return a;
  };

  // Every focusable, visible control, and the facts about each that train.py's features use. Nothing outlives the call.
  const collect = () => {
    const sticks = new Map(),
      regions = new Map(),
      layers = new Map();
    const isFixed = (el) => {
      if (!el || el == document.documentElement) return false;
      if (!sticks.has(el))
        sticks.set(el, /fixed|sticky/.test(getComputedStyle(el).position) || isFixed(el.parentElement));
      return sticks.get(el);
    };
    // the nearest ancestor that sits in its own layer above the page: fixed, or positioned with a z-index
    const layerOf = (el) => {
      for (let p = el.parentElement; p && p != document.documentElement; p = p.parentElement) {
        if (!layers.has(p)) {
          const st = getComputedStyle(p),
            z = parseInt(st.zIndex);
          const own = st.position == "fixed" || (/absolute|sticky/.test(st.position) && z > 0);
          const r = own && p.getBoundingClientRect();
          layers.set(
            p,
            own
              ? {
                  id: layers.size,
                  z: z > 0 ? z : 0,
                  rect: [r.left, r.top, r.width, r.height],
                }
              : null,
          );
        }
        if (layers.get(p)) return layers.get(p);
      }
      return null;
    };
    // is the control what you would hit at its centre (1), covered by something on top (0), or not on screen (-1)?
    const topmost = (e, r) => {
      if (r.width < 4 || r.height < 4 || !inView(r)) return -1;
      const t = document.elementFromPoint(
        Math.min(Math.max(r.x + r.width / 2, 0), innerWidth - 1),
        Math.min(Math.max(r.y + r.height / 2, 0), innerHeight - 1),
      );
      return t && (t == e || e.contains(t) || t.contains(e) || t.shadowRoot?.contains(e)) ? 1 : 0;
    };
    let els = deep(document).filter(alive);
    if (els.length > 5000) {
      // a safety valve for endless pages: the control you are in and its form, then what is on screen, then the rest in page order
      const form = active().closest("form");
      const keep = new Set(
        [
          ...els.filter((e) => e == active() || form?.contains(e)),
          ...els.filter((e) => inView(e.getBoundingClientRect())),
          ...els,
        ].slice(0, 5000),
      );
      els = els.filter((e) => keep.has(e));
    }
    const controls = els.map((e, i) => {
      const r = e.getBoundingClientRect(),
        b = e.parentElement?.closest(BOX),
        f = e.form ?? e.closest("form");
      const st = getComputedStyle(e),
        lay = layerOf(e);
      // how it looks: filled or not, how colourful, how round, how bold
      const [red = 0, green = 0, blue = 0, alpha = 1] = st.backgroundColor.match(/[\d.]+/g)?.map(Number) ?? [
        0, 0, 0, 0,
      ];
      const [lx, ly, lw, lh] = lay?.rect ?? [0, 0, 0, 0];
      let before = "";
      for (let p = e; p && !before; p = p.parentElement) before = text(p.previousElementSibling, 20);
      if (b && !regions.has(b)) regions.set(b, regions.size);
      return {
        i,
        n: els.length,
        form: f ? [...document.forms].indexOf(f) : -1,
        action: cut(path(f?.getAttribute("action") ?? ""), 30),
        submit: !!f && e.matches(SUBMIT),
        filled: e.matches(TYPED) ? !!e.value : e.matches(EDITABLE) && !!e.innerText.trim(),
        required: e.required === true,
        hit: topmost(e, r),
        region: b ? regions.get(b) : -1,
        dialog: !!e.closest(DIALOG),
        bar: !!e.closest("[role=menubar],[role=toolbar]"),
        fixed: isFixed(e),
        // the layer it sits in: how much of the view it covers, how high it is stacked, how many screen edges it touches
        layer: lay ? lay.id : -1,
        lcover: lay ? Math.min(1, (lw * lh) / (innerWidth * innerHeight)) : 0,
        lw: lay ? Math.min(1, lw / innerWidth) : 0,
        lh: lay ? Math.min(1, lh / innerHeight) : 0,
        lz: lay ? Math.log10(1 + lay.z) / 6 : 0,
        ledge: lay
          ? [lx <= 2, ly <= 2, lx + lw >= innerWidth - 2, ly + lh >= innerHeight - 2].filter(Boolean).length
          : 0,
        fill: alpha,
        sat: alpha ? (Math.max(red, green, blue) - Math.min(red, green, blue)) / (Math.max(red, green, blue) || 1) : 0,
        lum: alpha ? (0.299 * red + 0.587 * green + 0.114 * blue) / 255 : 0,
        round: Math.min(1, parseFloat(st.borderTopLeftRadius) / Math.max(1, Math.min(r.width, r.height) / 2) || 0),
        bold: (parseInt(st.fontWeight) || 400) / 1000,
        pointer: st.cursor == "pointer",
        icon: !!e.querySelector("svg,img"),
        kind: e.tagName == "INPUT" ? `input ${e.type}` : e.getAttribute("role") || e.tagName.toLowerCase(),
        label: cut(label(e), 50),
        box: b ? b.getAttribute("role") || b.tagName.toLowerCase() : "",
        name: e.name || e.id || "",
        ph: e.placeholder || "",
        href: cut(path(e.getAttribute("href") ?? ""), 40),
        before,
        parent: text(e.parentElement, 20),
        x: q(r.x),
        y: q(r.y),
        w: q(r.width),
        h: q(r.height),
      };
    });
    return { els, controls };
  };

  // The overlay: numbered badges beside the four picks, a small spinning J while it works, and short notes so a failure is never silent.
  // It lives in a closed shadow root, so the page's own CSS and scripts cannot reach it; a constructed stylesheet is not blocked by a page's CSP.
  const host = Object.assign(document.createElement("div"), {
    id: "jab-overlay",
  });
  const root = host.attachShadow({ mode: "closed" }),
    sheet = new CSSStyleSheet();
  host.style.cssText = "all: initial; position: fixed; inset: 0; pointer-events: none; z-index: 2147483647;";
  sheet.replaceSync(`
    .badge { position: fixed; min-width: 18px; height: 18px; padding: 0 3px; box-sizing: border-box; border-radius: 9px; background: #2b6f4b; color: #fff; opacity: .85; font: 700 11px/18px system-ui, sans-serif; text-align: center; box-shadow: 0 1px 3px rgba(0, 0, 0, .35); }
    .badge.now { background: #00e676; color: #00210e; outline: 2px solid #00210e; box-shadow: 0 0 0 4px rgba(0, 230, 118, .55), 0 2px 6px rgba(0, 0, 0, .4); transform: scale(1.15); z-index: 1; }
    .badge.off { opacity: .65; }
    .note { position: fixed; right: 16px; bottom: 56px; max-width: 280px; padding: 6px 10px; border-radius: 6px; background: #1f4f35; color: #fff; font: 12px/1.4 system-ui, sans-serif; box-shadow: 0 2px 6px rgba(0, 0, 0, .3); }
    .spinner { position: fixed; right: 16px; bottom: 16px; width: 28px; height: 28px; font: 700 15px/28px system-ui, sans-serif; text-align: center; color: #2b8f58; opacity: 0; transition: opacity .2s; }
    .spinner.on { opacity: .8; }
    .spinner::before { content: ''; position: absolute; inset: 0; box-sizing: border-box; border: 2px solid rgba(43, 143, 88, .22); border-top-color: #2b8f58; border-radius: 50%; animation: turn .9s linear infinite; }
    @keyframes turn { to { transform: rotate(360deg); } }`);
  root.adoptedStyleSheets = [sheet];
  const draw = (className, textContent) => {
    document.documentElement.append(host);
    return root.appendChild(Object.assign(document.createElement("div"), { className, textContent }));
  };
  const spinner = draw("spinner", "J");
  const note = (message) => setTimeout((n) => n.remove(), 4000, draw("note", message));

  // A session: our top 4 picks and their badges. Ctrl+Q walks them; anything else you do ends it, and so do 10 s without a press.
  let ses = null,
    timer,
    busy = false,
    gen = 0;
  // each badge sits on the top-left corner of its control, or at the screen edge with an arrow when the control is off screen; the focused one is drawn stronger
  const show = () =>
    ses?.badges.forEach((badge, i) => {
      const r = ses.els[i].getBoundingClientRect();
      const arrow =
        r.top >= innerHeight ? "↓" : r.bottom <= 0 ? "↑" : r.left >= innerWidth ? "→" : r.right <= 0 ? "←" : "";
      badge.textContent = `${i + 1}${arrow}`;
      badge.hidden = !r.width && !r.height; // the page has hidden it
      badge.classList.toggle("off", !!arrow);
      badge.classList.toggle("now", ses.els[i] == active());
      badge.style.cssText = `left: ${Math.min(Math.max(2, r.left - 8), innerWidth - 30)}px; top: ${Math.min(Math.max(2, r.top - 8), innerHeight - 22)}px`;
    });
  const end = () => {
    gen++; // an answer still on its way is dropped
    clearTimeout(timer);
    ses?.badges.forEach((badge) => badge.remove());
    ses = null;
  };
  const touch = () => {
    clearTimeout(timer);
    timer = setTimeout(end, 10000);
  };
  for (const type of ["scroll", "resize", "focusin"]) addEventListener(type, show, { capture: true, passive: true });
  setInterval(show, 300); // catches layout shifts that fire no event
  // focus left our picks (checked once it has moved, so a web component's retargeted events cannot fool it)
  addEventListener("focusout", () => setTimeout(() => ses && !ses.els.includes(active()) && end()), true);
  // typing, clicking, submitting, navigating: the next press re-ranks from there
  for (const type of ["input", "submit", "pointerdown", "popstate", "hashchange", "pagehide"])
    addEventListener(type, end, true);
  // a text field fires "change" when focus leaves it, which is jab moving on: only a select, checkbox or radio changing counts
  addEventListener("change", (e) => !e.target.matches?.(TYPED) && end(), true);
  // Ctrl+Q types nothing and Chrome does not use it; any other key (Enter, Escape, arrows, Tab, ...) ends the session
  // While Ctrl+Q is held down, a digit 1-4 goes straight to that stop; any other time Ctrl+digit is left to Chrome.
  // Inside a session, Ctrl+Q moves to the next stop only on a tap (let go within 0.4 s, no digit): holding it is for choosing.
  let held = false,
    pick = 0,
    tapped = 0;
  const ctrl = (e) => e.ctrlKey && !e.altKey && !e.metaKey && !e.shiftKey;
  const isPress = (e) => e.code == "KeyQ" && ctrl(e);
  const digit = (e) => (held && ctrl(e) && /^Digit[1-4]$/.test(e.code) ? Number(e.code[5]) : 0);
  // Focus pick i. A pick that does not take the focus (the page has since hidden it, or a pop-up keeps the focus to itself)
  // is replaced by the next control in the ranking, or dropped when the ranking runs out; with no pick left the session ends.
  const land = (i) => {
    while (ses?.els.length) {
      i %= ses.els.length;
      const el = ses.els[i];
      el.focus();
      if (active() == el) return touch(), show();
      if (ses.rest.length) ses.els[i] = ses.rest.shift();
      else {
        ses.els.splice(i, 1);
        ses.badges.splice(i, 1)[0].remove();
      }
    }
    end();
    note("jab could not move the focus here: the page keeps it where it is.");
  };
  const step = () => {
    // walk the stored picks, no recompute; going round to 1 again flashes its badge
    const next = (ses.els.indexOf(active()) + 1) % ses.els.length;
    if (next == 0) ses.badges[0].animate([{ transform: "scale(1.7)" }, { transform: "none" }], 450);
    land(next);
  };
  addEventListener(
    "keyup",
    (e) => {
      if (e.key != "Control" && e.code != "KeyQ") return;
      if (tapped && performance.now() - tapped < 400 && ses) step();
      held = false;
      tapped = 0;
    },
    true,
  );
  addEventListener("blur", () => (held = tapped = 0));
  addEventListener(
    "keydown",
    (e) => !isPress(e) && !digit(e) && !["Control", "Shift", "Alt", "Meta"].includes(e.key) && end(),
    true,
  );

  addEventListener(
    "keydown",
    async (e) => {
      const n = digit(e);
      if (!isPress(e) && !n) return;
      if (!chrome.runtime?.id) return note("jab was reloaded: refresh this page to use it."); // a stale copy of this script leaves Ctrl+Q alone
      e.preventDefault();
      e.stopPropagation();
      if (e.repeat) return; // holding the keys down is one press
      held = true;
      if (n) {
        // Ctrl+Q, then a digit: that stop now, or as soon as the ranking arrives
        tapped = 0; // a digit, not a tap
        if (ses && n <= ses.els.length) {
          land(n - 1);
        } else if (busy) pick = n;
        return;
      }
      if (busy) return; // one press at a time
      if (ses?.els.every((el) => el.isConnected)) {
        tapped = performance.now(); // inside a session: the move waits for the keys to be let go (see keyup)
        return touch();
      }
      end(); // no session, or a pick left the page (re-rendered): start over
      busy = true;
      pick = 0;
      const mine = gen,
        from = active();
      const delay = setTimeout(() => spinner.classList.add("on"), 120); // appears only if the work takes more than a blink
      try {
        const { els, controls } = collect();
        const page = {
          host: location.hostname,
          path: location.pathname,
          title: document.title,
          desc: document.querySelector("meta[name=description]")?.content ?? "",
          scroll: Math.min(20, document.documentElement.scrollHeight / innerHeight),
        };
        const timeout = new Promise((_, fail) => setTimeout(() => fail(new Error("no answer within 5 s")), 5000));
        const scores = await Promise.race([
          timeout,
          chrome.runtime.sendMessage({
            controls,
            focus: els.indexOf(from),
            page,
          }),
        ]);
        if (mine != gen || active() != from) return; // something happened while it ranked (typing, a click, ...): drop the answer
        // the best four; the focused control has no score (null)
        const ranked = scores
          .map((v, i) => [v, els[i]])
          .filter(([v]) => v !== null)
          .sort((a, b) => b[0] - a[0])
          .slice(0, 24)
          .map(([, el]) => el);
        const best = ranked.slice(0, 4); // the next 20 stand in for a pick that turns out not to take the focus
        if (!best.length) return note("jab found nothing to jump to here.");
        ses = { els: best, rest: ranked.slice(4), badges: best.map((_, i) => draw("badge", i + 1)) };
        show();
        touch();
        land(Math.min(pick || 1, best.length) - 1); // the first press lands on the best pick (or the stop a digit asked for)
      } catch (error) {
        note(`jab could not answer: ${error.message}`);
      } finally {
        clearTimeout(delay);
        spinner.classList.remove("on");
        busy = false;
      }
    },
    true,
  );
} else {
  // The ranker: two LightGBM models (ranker.json, made by train.py: a ranker and a pick-or-not classifier) score every control of the page from cheap features
  // (position, kind, what is on screen, words in the label), no neural network. rank(model, controls, focus, page): controls as collect() returns them,
  // focus the index of the focused control or -1. Returns one score per control (each model's scores standardised within the page, then added); null for the focused one.
  const length = (s) => [...s].length;
  const words = (s) => new Set(s.toLowerCase().match(/[a-z]{3,}/g));
  const onScreen = (c) => c.w >= 4 && c.h >= 4 && c.y + c.h >= 0 && c.y <= 900 && c.x <= 1280;
  const isField = (c) =>
    /^(textarea|input (text|search|email|password|tel|url|number)|textbox|searchbox|combobox)$/.test(c.kind); // controls you type into
  const guess = (c) =>
    (/^(input|textarea|select)/.test(c.kind) ? 3 : c.kind.startsWith("button") ? 2 : 1) +
    (c.box == "main") +
    2 * onScreen(c) -
    c.i / 100;
  const count = (items) => items.reduce((m, k) => m.set(k, (m.get(k) ?? 0) + 1), new Map());
  // a tree: go left when the feature is <= the threshold; a negative child ~n is leaf n
  const walk = ([feature, threshold, left, right, leaf], x) => {
    if (!feature.length) return leaf[0];
    let node = 0;
    while (node >= 0) node = x[feature[node]] <= threshold[node] ? left[node] : right[node];
    return leaf[~node];
  };
  const standardise = (r) => {
    const mean = r.reduce((a, b) => a + b, 0) / r.length;
    const sd = Math.sqrt(r.reduce((a, b) => a + (b - mean) ** 2, 0) / r.length) + 1e-9;
    return r.map((v) => (v - mean) / sd);
  };

  // the same features, in the same order, as train.py; every count is computed once per page, so thousands of controls stay cheap
  const rank = (model, all, focus, page) => {
    const controls = all.filter((c) => c.i != focus),
      focused = all.find((c) => c.i == focus);
    const shown = controls.filter(onScreen),
      shownY = shown.map((c) => c.y).sort((a, b) => a - b),
      shownRank = new Map(shown.map((c, r) => [c.i, r]));
    const inputsShown = shown.filter((c) => c.kind.startsWith("input")).length;
    // how much a control stands out by being a coloured, filled shape; 0 = the most colourful control on screen
    const colour = (c) => (c.fill >= 0.5 ? c.sat : 0);
    const satRank = new Map([...shown].sort((a, b) => colour(b) - colour(a)).map((c, r) => [c.i, r]));
    // how much of what is on screen is covered by something on top
    const known = controls.filter((c) => c.hit >= 0),
      occludedShare = known.length ? known.filter((c) => c.hit == 0).length / known.length : 0;
    const layered = controls.filter((c) => c.layer >= 0),
      layerSize = count(layered.map((c) => c.layer));
    const layerKnown = count(known.filter((c) => c.layer >= 0).map((c) => c.layer)),
      layerHits = count(known.filter((c) => c.layer >= 0 && c.hit == 1).map((c) => c.layer));
    const regionSize = count(controls.filter((c) => c.region >= 0).map((c) => c.region)),
      kindCount = count(controls.map((c) => c.kind)),
      kindSeen = new Map();
    const fields = all.filter((c) => isField(c) && c.form >= 0),
      formFields = count(fields.map((c) => c.form)); // the typing fields that sit in a form
    const firstField = new Map();
    for (const c of fields) if (!firstField.has(c.form)) firstField.set(c.form, c.i);
    const inForm = (c, form) => form >= 0 && c.form == form;
    const focusForm = focused ? focused.form : -1,
      emptyLeft = fields.filter((c) => !c.filled && inForm(c, focusForm) && c.i != focus).length;
    const same = (c, key) => !!focused && c[key] >= 0 && c[key] == focused[key]; // in the same layer or region as the focused control
    // how many on-screen controls are above y
    const above = (y) => {
      let lo = 0,
        hi = shownY.length;
      while (lo < hi) {
        const mid = (lo + hi) >> 1;
        if (shownY[mid] < y) lo = mid + 1;
        else hi = mid;
      }
      return lo;
    };
    // one row, reused for every control: the features, then a 0/1 per word of the control, then per word of the page (those stay set)
    const nf = model.features.length,
      x = new Float64Array(nf + model.words.length),
      column = new Map(model.words.map((w, j) => [w, nf + j]));
    const columns = (prefix, text) =>
      [...words(text)].map((w) => column.get(prefix + w)).filter((j) => j !== undefined);
    for (const j of columns("p:", `${page.host} ${page.path} ${page.title} ${page.desc}`)) x[j] = 1;
    const raw = model.models.map(() => []);
    for (const c of controls) {
      const sameRank = kindSeen.get(c.kind) ?? 0,
        inFocusForm = inForm(c, focusForm);
      kindSeen.set(c.kind, sameRank + 1);
      const f = {
        same_kind_rank: sameRank,
        same_kind_count: kindCount.get(c.kind),
        inputs_on_screen: inputsShown,
        screen_rank: shownRank.get(c.i) ?? -1,
        on_screen_count: shown.length,
        controls_above: above(c.y),
        no_parent_text: c.parent == "",
        before_len: length(c.before),
        kind: model.kinds[c.kind] ?? -1,
        box: model.boxes[c.box] ?? -1,
        x: c.x / 1280,
        y: c.y / 900,
        w: c.w / 1280,
        h: c.h / 900,
        index: c.i,
        index_frac: c.i / Math.max(1, c.n),
        count: c.n,
        label_len: length(c.label),
        nothing_focused: focus < 0,
        index_delta: focus >= 0 ? c.i - focus : 0,
        guess: guess(c),
        required: c.required,
        is_submit: c.submit,
        dismissive: /^(clear|reset|cancel|close|dismiss|decline|no thanks|skip|back)\b/.test(c.label.toLowerCase()),
        first_field: isField(c) && c.form >= 0 && firstField.get(c.form) == c.i,
        form_fields: c.form >= 0 ? (formFields.get(c.form) ?? 0) : 0,
        focus_filled: !!focused?.filled,
        in_focus_form: inFocusForm,
        after_focus: inFocusForm && c.i > focus,
        empty_fields_left: emptyLeft,
        empty_field: isField(c) && !c.filled,
        focus_dx: focused ? (c.x - focused.x) / 1280 : 0,
        focus_dy: focused ? (c.y - focused.y) / 900 : 0,
        focus_dist: focused ? Math.hypot(c.x - focused.x, c.y - focused.y) / 1000 : 0,
        submit_like:
          /\b(send|submit|go|search|post|reply|comment|save|continue|next|sign in|log in|login|apply|ask)\b/.test(
            c.label.toLowerCase(),
          ),
        fill: c.fill,
        sat: c.sat,
        lum: c.lum,
        round: c.round,
        bold: c.bold,
        pointer: c.pointer,
        icon: c.icon,
        sat_rank: satRank.get(c.i) ?? -1,
        hit: c.hit,
        occluded_share: occludedShare,
        in_layer: c.layer >= 0,
        lcover: c.lcover,
        lw: c.lw,
        lh: c.lh,
        lz: c.lz,
        ledge: c.ledge,
        layer_size: c.layer >= 0 ? layerSize.get(c.layer) : 0,
        layer_hit_share: layerKnown.get(c.layer) ? (layerHits.get(c.layer) ?? 0) / layerKnown.get(c.layer) : 1,
        same_layer: same(c, "layer"),
        focus_in_layer: !!focused && focused.layer >= 0,
        same_region: same(c, "region"),
        region_size: c.region >= 0 ? regionSize.get(c.region) : 0,
        in_dialog: c.dialog,
        bar: c.bar,
        fixed: c.fixed,
        page_scroll: page.scroll ?? 1,
        // in a layer of its own, on top, with the page behind it covered: what the site is presenting
        foreground: c.layer >= 0 && c.hit == 1 ? occludedShare : 0,
      };
      model.features.forEach((name, j) => (x[j] = Number(f[name])));
      const set = columns("w:", `${c.label} ${c.href} ${c.name} ${c.ph} ${c.action}`);
      for (const j of set) x[j] = 1;
      model.models.forEach((trees, m) => raw[m].push(trees.reduce((sum, tree) => sum + walk(tree, x), 0)));
      for (const j of set) x[j] = 0;
    }
    const total = raw.map(standardise).reduce((a, b) => a.map((v, k) => v + b[k]));
    let k = 0;
    return all.map((c) => (c.i == focus ? null : total[k++]));
  };
  const model = fetch(chrome.runtime.getURL("ranker.json")).then((r) => r.json()); // loaded once per worker start, in milliseconds
  chrome.runtime.onMessage.addListener(({ controls, focus, page }, _, send) => {
    model.then((m) => send(rank(m, controls, focus, page)));
    return true;
  });
}
