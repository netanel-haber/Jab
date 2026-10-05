// tab-at-home in JavaScript: loads the one-file ONNX model and the Qwen tokenizer from the extension, encodes a choice question the way train.py does, and softmaxes the scores.
const DELIMITERS = ['<|fim_prefix|>', '<|fim_middle|>', '<|box_start|>', '<|box_end|>', '<|fim_suffix|>'];   // <state> <q> <opt> </opt> <decide>
const INSTR = 'The user is about to press Tab. Which control does this page want them to reach next?';       // the instruction the model was trained with

export async function load(ort, { AutoTokenizer, env }, base) {   // base: the extension's URL, or any URL the model folder is served from
  ort.env.wasm.wasmPaths = `${base}lib/`;
  env.allowRemoteModels = false;
  env.localModelPath = base;
  const tokenizer = await AutoTokenizer.from_pretrained('model');
  const session = await ort.InferenceSession.create(`${base}model/model.onnx`, { executionProviders: ['wasm'] });
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
