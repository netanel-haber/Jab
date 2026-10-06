// Page side of the tab student: find the focusable controls and describe each at three context levels.
// Level 1 is the baseline (~x tokens per control, incl. whether it is on screen), level 2 adds attributes and viewport geometry (~2x), level 3 adds the text
// around the control (~3x). Used by collect.mjs for training data and, unchanged, by the extension at inference.
window.__tab = (() => {
  const SEL = 'a[href],button,input:not([type=hidden]),select,textarea,summary,[contenteditable=""],[contenteditable=true],[tabindex]:not([tabindex="-1"]),[role=button],[role=link],[role=textbox],[role=combobox],[role=searchbox],[role=checkbox],[role=tab],[role=menuitem]';
  const BOX = 'form,dialog,[role=dialog],[role=search],nav,header,footer,aside,main,section,fieldset,[role=banner],[role=navigation],[role=contentinfo]';
  const cut = (s, n) => Array.from(s).slice(0, n).join('');   // by code point: slicing a string can split an emoji and leave a lone surrogate the tokenizer rejects
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
  const level1 = (c) => `${c.kind} "${c.label}"${c.box ? ` [in ${c.box.kind}${c.box.title ? ': ' + c.box.title : ''}]` : ''}${c.flags.length ? ` {${c.flags}}` : ''} (${c.view ? 'on' : 'off'} screen)`;
  const describe = (c, level) => level == 1 ? level1(c)
    : level1(c) + `; ${[c.name && `name=${c.name}`, c.ph && `placeholder=${c.ph}`, c.href && `href=${c.href}`].filter(Boolean).join(' ')} at ${c.place} ${c.size} x=${c.x} y=${c.y} ${c.w}x${c.h}`
      + (level == 3 ? `; after "${c.before}"; in "${c.parent}"; #${c.i + 1}/${c.n}` : '');
  const state = (level, focused) => cut(`Page: ${document.title} (${location.hostname}). ${text(document.querySelector('h1'), 80)}`
    + (level >= 2 ? ` ${cut(document.querySelector('meta[name=description]')?.content ?? '', 120)}` : '')
    + (level >= 3 ? ` Sections: ${[...document.querySelectorAll('h2')].slice(0, 5).map((h) => text(h, 30)).join(' | ')}` : ''), [80, 120, 160][level - 1])   // a short state: the page's identity, not its copy
    + ` Focused now: ${focused ? level1(focused) : 'nothing'}.`;
  const focusIndex = () => window.__tabEls.indexOf(document.activeElement);
  const ring = () => { const s = getComputedStyle(document.activeElement); return (s.outlineStyle != 'none' && parseFloat(s.outlineWidth) > 0) || s.boxShadow != 'none'; };
  return { collect, describe, state, focusIndex, ring };
})();
// Press Ctrl+Q: focus where this page most likely wants you next (again for the one after). Ctrl+Q types nothing and Chrome does not use it; every other key is untouched.
const LEVEL = 1, K = 5;   // description detail (1 = ~x tokens per control, 2 = ~1.7x, 3 = ~3x; level 1 is as accurate and about twice as fast), candidates per question
const weight = (c) => (/^(input|textarea|select)/.test(c.kind) ? 3 : c.kind.startsWith('button') ? 2 : 1) + (c.box?.kind == 'main') + 2 * c.view - (c.skip ? 5 : 0) - c.i / 100;   // fields, buttons, links; main area first
const alive = () => !!chrome.runtime?.id;   // false once the extension has been reloaded or updated: this old copy of the script then does nothing at all, and a refreshed page gets a new one
const warm = () => alive() && document.visibilityState == 'visible' && chrome.runtime.sendMessage({ ensure: 1 }).catch(() => {});   // the model loads while you browse, so the first press does not wait for it
warm();
addEventListener('visibilitychange', warm);
let ses;   // ses = {els: our top 3 picks, n: index of the current one}; it lasts until focus leaves our picks
addEventListener('focusout', (e) => ses && !ses.els.includes(e.relatedTarget) && (ses = null), true);   // focus left our picks (click elsewhere, window blur, ...): session over
addEventListener('keydown', async (e) => {
  if (!alive() || e.code != 'KeyQ' || !e.ctrlKey || e.altKey || e.metaKey || e.shiftKey) return;   // (a dead copy leaves Ctrl+Q alone)
  e.preventDefault(); e.stopPropagation();
  if (e.repeat) return;   // holding the keys down is one press
  if (ses) { ses.n = (ses.els.indexOf(document.activeElement) + 1) % ses.els.length; return ses.els[ses.n].focus(); }   // inside a session: walk the stored picks, no recompute
  const all = __tab.collect(), at = __tab.focusIndex(), others = all.filter((c) => c.i != at).sort((a, b) => weight(b) - weight(a)).slice(0, K - (at >= 0)), cands = at >= 0 ? [...others, all[at]] : others;
  await chrome.runtime.sendMessage({ ensure: 1 });
  const p = await chrome.runtime.sendMessage({ state: __tab.state(LEVEL, all[at]), options: cands.map((c) => __tab.describe(c, LEVEL) + (c.i == at ? ' (focused)' : '')) });
  const top = p.map((v, k) => [cands[k].i == at ? -1 : v, cands[k].i]).sort((a, b) => b[0] - a[0]).slice(0, 3).map((x) => __tabEls[x[1]]);
  ses = { els: top }; top[0].focus();
}, true);
