// service worker: opens the model page on request; model page: runs the model, closes itself after 10 idle minutes
// The model, in JavaScript: loads model.onnx and Qwen's tokenizer.json, encodes a choice question the way train.py does, and softmaxes the scores.
const DELIMITERS = ['<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>'];   // <state> <q> <opt> </opt> <decide>
const INSTR = 'The user is about to press Tab. Which control does this page want them to reach next?';       // the instruction the model was trained with

async function load(ort, { PreTrainedTokenizer }, base) {   // base: the extension's URL, or any URL the files are served from
  ort.env.wasm.wasmPaths = `${base}lib/`;
  const tokenizer = new PreTrainedTokenizer(await (await fetch(`${base}tokenizer.json`)).json(), {});
  const session = await ort.InferenceSession.create(`${base}model.onnx`, { executionProviders: ['wasm'] });
  const [S, Q, O, C, D] = DELIMITERS.map((t) => tokenizer.convert_tokens_to_ids(t));
  const tokens = (text) => tokenizer.encode(text.toWellFormed().replace(/<\|([A-Za-z0-9_]+)\|>/g, '<¦$1¦>'), { add_special_tokens: false });   // page text can never forge a delimiter
  const int64 = (a, dims) => new ort.Tensor('int64', BigInt64Array.from(a, BigInt), dims);
  return async (state, options) => {
    const ids = [S, ...tokens(state).slice(0, 383), Q, ...tokens(INSTR)], opts = [];
    for (const option of options) {
      ids.push(O, ...tokens(option), C);
      opts.push(ids.length - 1);
    }
    ids.push(D);
    const { scores } = await session.run({ ids: int64(ids, [1, ids.length]), decide: int64([ids.length - 1], []), opts: int64(opts, [opts.length]) });
    const z = Array.from(scores.data), max = Math.max(...z), e = z.map((v) => Math.exp(v - max));
    return e.map((v) => v / e.reduce((a, b) => a + b));
  };
}

if (self.document) {
  let idle; const score = Promise.all([import('./lib/ort.wasm.min.mjs'), import('./lib/transformers.min.js')]).then(([ort, tf]) => load(ort, tf, chrome.runtime.getURL('')));
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
