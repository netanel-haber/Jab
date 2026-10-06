// service worker: opens the model page on request; model page: runs the model, closes itself after 10 idle minutes
// The model, in JavaScript: loads model.onnx and Qwen's tokenizer.json, encodes a choice question the way train.py does, and softmaxes the scores.
const DELIMITERS = ['<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>'];   // <state> <q> <opt> </opt> <decide>
const INSTR = 'The user is about to press Tab. Which control does this page want them to reach next?';       // the instruction the model was trained with

async function load(ort, { PreTrainedTokenizer }, base) {   // base: the extension's URL, or any URL the files are served from
  ort.env.wasm.wasmPaths = `${base}lib/`;
  ort.env.wasm.numThreads = Math.min(8, Math.max(1, navigator.hardwareConcurrency - 2));   // ONNX Runtime's default is 4; more cores, faster, up to 8
  const [tokenizer, session] = await Promise.all([   // in parallel; the graph is already clean, so light optimization halves the session's load time at the same speed
    fetch(`${base}tokenizer.json`).then((r) => r.json()).then((json) => new PreTrainedTokenizer(json, {})),
    ort.InferenceSession.create(`${base}model.onnx`, { executionProviders: ['wasm'], graphOptimizationLevel: 'basic' }),
  ]);
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
  let idle;
  const rest = () => { clearTimeout(idle); idle = setTimeout(() => chrome.runtime.sendMessage({ close: 1 }), 6e5); };   // closes after 10 idle minutes; a page opening or a press counts as use
  const score = Promise.all([import('./lib/ort.wasm.min.mjs'), import('./lib/transformers.min.js')]).then(([ort, tf]) => load(ort, tf, chrome.runtime.getURL('')));   // starts loading as soon as this page exists
  rest();
  chrome.runtime.onMessage.addListener(({ ensure, state, options }, _, send) => {
    if (ensure) rest();
    if (!options) return;
    clearTimeout(idle);
    score.then((run) => run(state, options)).then(send).finally(rest);
    return true;
  });
} else chrome.runtime.onMessage.addListener((m, _, send) => {
  if (m.close) chrome.offscreen.closeDocument();
  if (m.ensure) chrome.offscreen.createDocument({ url: 'offscreen.html', reasons: ['WORKERS'], justification: 'model' }).catch(() => {}).then(() => send(1));
  return !!m.ensure;
});
