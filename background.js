// Service worker: ranks a page's controls with the LightGBM model in ranker.json (see rank.js). Nothing is cached per page: every press scores the page as it is right now.
importScripts('rank.js');
const model = fetch(chrome.runtime.getURL('ranker.json')).then((r) => r.json());   // loaded once per worker start, in milliseconds
chrome.runtime.onMessage.addListener(({ controls, focus }, _, send) => {
  model.then((m) => send(rank(m, controls, focus).map((v) => (Number.isFinite(v) ? v : null))));   // JSON cannot carry -Infinity
  return true;
});
