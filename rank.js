// The ranker: a LightGBM model (ranker.json, made by train.py) scores every control of the page from cheap features (position, size, kind, on screen, words in the label), no neural network.
// rank(model, controls, focus): controls as content.js collect() returns them, focus the index of the focused control or -1. Returns one score per control; the focused one gets -Infinity.
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
if (typeof module != 'undefined') module.exports = rank;
