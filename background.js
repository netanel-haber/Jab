// service worker: opens the model page on request; model page: runs the student, closes itself after 10 idle minutes
if (self.document) {
  let idle; const score = Promise.all([import('./lib/ort.wasm.min.mjs'), import('./lib/transformers.min.js'), import('./lib/tabathome.mjs')]).then(([ort, tf, { load }]) => load(ort, tf, chrome.runtime.getURL('')));
  chrome.runtime.onMessage.addListener(({ state, options }, _, send) => {
    if (!options) return; clearTimeout(idle);
    score.then((run) => run(state, options)).then(send).finally(() => (idle = setTimeout(() => chrome.runtime.sendMessage({ close: 1 }), 6e5)));
    return true;
  });
} else chrome.runtime.onMessage.addListener((m, _, send) => {
  if (m.close) chrome.offscreen.closeDocument();
  if (m.ensure) chrome.offscreen.createDocument({ url: 'offscreen.html', reasons: ['WORKERS'], justification: 'model' }).catch(() => {}).then(() => send(1));
  return !!m.ensure;
});
