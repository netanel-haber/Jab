# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch>=2.6,<2.9", "transformers>=5.17,<6", "numpy"]
# ///
"""Tab at home, which control should Tab reach next?   uv run train.py PRETRAINED OUT DATA.jsonl.gz
Cuts a Qwen3 model (PRETRAINED: Hugging Face name or path, e.g. Qwen/Qwen3-0.6B-Base) to 8 of its 28 layers and 16k of its 151k tokens, and trains it, with a pointer head,
for 1,400 steps (10 minutes on an RTX 3500 Ada; the last 40% with int2 weights in the forward pass). Saves OUT/final; rerun the same command to resume. DATA is not included:
one JSON object per line, {"split": "train"|"eval", "cohort", "cand", "ok": [option indexes the labeller accepts], "rec": {"state": page text, "questions": [{"instr", "options": [control descriptions], "label": index}]}}."""
import gzip, json, os, sys
from collections import Counter

import numpy as np, torch, torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
pretrained, out, data = sys.argv[1:4]
VOCAB, HEAD_DIM, STEPS, BATCH, LR, HEAD_LR, QAT_FROM = 16000, 256, 1400, 16, 8e-5, 5e-4, 0.6
DELIMITERS = ["<|fim_prefix|>", "<|fim_middle|>", "<|box_start|>", "<|box_end|>", "<|fim_suffix|>"]   # <state> <q> <opt> </opt> <decide>: rarely used Qwen tokens
device = "cuda" if torch.cuda.is_available() else "cpu"
os.makedirs(f"{out}/final", exist_ok=True)
with gzip.open(data, "rt", encoding="utf-8") as f:
    records = [json.loads(line) for line in f]
tok = AutoTokenizer.from_pretrained(pretrained)
model = AutoModel.from_pretrained(pretrained, dtype=torch.float32).to(device)
head = torch.nn.Linear(model.config.hidden_size, 2 * HEAD_DIM).to(device)   # a query and a key per position

# Vocabulary: the delimiters, pad, and the tokens the training pages use most. Any other token reads as pad.
train_texts = [t for r in records if r["split"] == "train" for t in [r["rec"]["state"], *r["rec"]["questions"][0]["options"]]]
keep = {tok.convert_tokens_to_ids(t) for t in DELIMITERS} | {tok.pad_token_id}
keep = sorted(keep | {t for t, _ in Counter(t for text in train_texts for t in tok(text, add_special_tokens=False).input_ids).most_common(VOCAB)})
new_id = {old: new for new, old in enumerate(keep)}
pad = new_id[tok.pad_token_id]
model.embed_tokens = torch.nn.Embedding.from_pretrained(model.embed_tokens.weight.data[keep], freeze=False)
model.config.vocab_size = len(keep)
# Layers: the first and last four.
model.layers = model.layers[:4] + model.layers[-4:]
model.config.num_hidden_layers = 8
model.config.layer_types = model.config.layer_types[:4] + model.config.layer_types[-4:]

def fake_quant(w):   # w as onnxruntime's int2 MatMulNBits will see it: blocks of 32 on the grid {-2, -1, 0, 1} x scale, scale = the block's largest weight / -2
    blocks = w.reshape(*w.shape[:-1], -1, 32)
    scale = blocks.gather(-1, blocks.abs().argmax(-1, keepdim=True)) / -2 + 1e-12
    return w + (((blocks / scale).round().clamp(-2, 1) * scale).reshape(w.shape) - w).detach()   # straight-through: the gradient ignores the rounding
step = 0
for m in model.modules():   # int2 weights in the forward pass for the last 40% of the steps
    if isinstance(m, torch.nn.Linear):
        m.forward = lambda x, m=m: F.linear(x, fake_quant(m.weight) if step > QAT_FROM * STEPS else m.weight, m.bias)
model.gradient_checkpointing_enable()
model.train()
# Each record becomes token ids: <state> page <q> instruction <opt> option </opt> ... <decide>, and the index of every </opt>.
tokens = lambda text: [new_id.get(t, pad) for t in tok(text, add_special_tokens=False).input_ids]
state, question, box_start, box_end, decide = (new_id[tok.convert_tokens_to_ids(t)] for t in DELIMITERS)
for r in records:
    q = r["rec"]["questions"][0]
    ids, ends = [state] + tokens(r["rec"]["state"])[:383] + [question] + tokens(q["instr"]), []   # (a state is cut to 383 tokens, as in the browser)
    for option in q["options"]:
        ids += [box_start] + tokens(option) + [box_end]
        ends.append(len(ids) - 1)
    r["ids"], r["ends"], r["label"] = ids + [decide], ends, q["label"]
train = [r for r in records if r["split"] == "train"]
held = [r for r in records if r["split"] == "eval"]

def logits(batch):   # one score per option: the key at its </opt> against the query at <decide> (right padding needs no mask under causal attention)
    ids = pad_sequence([torch.tensor(r["ids"]) for r in batch], batch_first=True, padding_value=pad).to(device)
    with torch.autocast(device, dtype=torch.bfloat16):
        queries_keys = head(model(input_ids=ids, use_cache=False).last_hidden_state)
    return [(h[r["ends"], HEAD_DIM:] @ h[len(r["ids"]) - 1, :HEAD_DIM]).float() / HEAD_DIM**0.5 for h, r in zip(queries_keys, batch)]

@torch.no_grad()
def evaluate(rows):   # top-1, top-3 and "any" (the top answer is one of the controls the labeller listed), per cohort and candidate set
    hits = {}
    for i in range(0, len(rows), 8):
        for r, z in zip(rows[i:i + 8], logits(rows[i:i + 8])):
            rank = int((~(z < z[r["label"]])).sum())   # ties and nan count against the model
            hits.setdefault(f"{r['cohort']}-{r['cand']}", []).append([rank == 1, rank <= 3, int(z.argmax()) in r["ok"]])
    return {k: np.mean(v, 0).round(3).tolist() for k, v in hits.items()}
opt = torch.optim.AdamW([{"params": model.parameters(), "lr": LR}, {"params": head.parameters(), "lr": HEAD_LR}], betas=(0.9, 0.95), weight_decay=0)
sched = get_cosine_schedule_with_warmup(opt, 50, STEPS)
parts, state_file = {"model": model, "head": head, "opt": opt, "sched": sched}, f"{out}/state.pt"
if os.path.exists(state_file):
    saved = torch.load(state_file, map_location=device)
    step = saved.pop("step")
    for name, part in parts.items():
        part.load_state_dict(saved[name])

while step < STEPS:
    batch = [train[j] for j in np.random.default_rng(step).integers(len(train), size=BATCH)]   # a function of the step alone, so resuming is exact
    loss = -sum(z.log_softmax(-1)[r["label"]] for r, z in zip(batch, logits(batch))) / len(batch)
    opt.zero_grad()
    loss.backward()
    opt.step()
    sched.step()
    step += 1
    if step % 200 == 0:   # save, and report loss and accuracy on every 7th held-out record
        torch.save({name: part.state_dict() for name, part in parts.items()} | {"step": step}, state_file)
        print(f"step {step}/{STEPS}  loss {loss.item():.3f}{'  int2' if step > QAT_FROM * STEPS else ''}  held-out (top1, top3, any): {evaluate(held[::7])}", flush=True)
report = evaluate(held)
print("final held-out (top1, top3, any):", report, flush=True)
model.save_pretrained(f"{out}/final")
torch.save(head.state_dict(), f"{out}/final/head.pt")
json.dump({"eval": report, "steps": step, "vocab": keep}, open(f"{out}/final/metrics.json", "w"))
