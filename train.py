# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "lightgbm", "scikit-learn"]
# ///
"""jab: which controls does a person want next?   uv run train.py DATA.jsonl.gz ranker.json [--all] [--cases FILE ...] [--cases-eval FILE] [--rollback]
Trains a LightGBM LambdaRank ranker and a "wanted or not" classifier over every control of a page, and writes their trees as ranker.json for jab.js. DATA (not included) has one labelled step per line:
{"split": "train"|"eval", "page": {"host", "path", "title", "desc", "scroll"}, "controls": [jab.js collect()], "focus": index or -1, "target": the control wanted next, "picks": the ordered top 4 a person would want, "ok": others that are also fine}.
Prints, for the "eval" pages: on landing, whether the first pick is wanted and how many of the 4 stops are; over all steps, top-1 / top-4. --all also trains on them, for the file you ship.
Case studies (steps of the same shape, plus "accept": other acceptable controls) go in a ledger next to the output (--cases FILE); every run applies the whole ledger on top of the base models, which are trained once and
kept in OUT.state/, with a policy-gradient step (softmax over the page's controls; reward the acceptable ones; replayed ordinary steps so nothing is forgotten). --rollback empties the ledger; changing the step and rerunning relearns only it."""
import argparse, bisect, collections, gzip, hashlib, json, math, pathlib, random, re, shutil
import numpy as np, lightgbm as lgb

ap = argparse.ArgumentParser(); ap.add_argument("data"); ap.add_argument("out"); ap.add_argument("--all", action="store_true"); ap.add_argument("--cases", nargs="*", default=[]); ap.add_argument("--cases-eval"); ap.add_argument("--rollback", action="store_true"); ap.add_argument("--rounds", type=int, default=100); ap.add_argument("--case-share", type=float, default=0.2); ap.add_argument("--lr", type=float, default=0.05)
args = ap.parse_args(); data, out = args.data, args.out
read = lambda path: [json.loads(line) for line in (gzip.open(path, "rt", encoding="utf-8") if str(path).endswith(".gz") else open(path, encoding="utf-8"))]
steps = read(data); train = [s for s in steps if args.all or s["split"] == "train"]; held = [s for s in steps if s["split"] == "eval"]
on_screen = lambda c: c["w"] >= 4 and c["h"] >= 4 and c["y"] + c["h"] >= 0 and c["y"] <= 900 and c["x"] <= 1280
words = lambda c: set(re.findall(r"[a-z]{3,}", f"{c['label']} {c['href']} {c['name']} {c['ph']} {c['action']}".lower()))
page_words = lambda p: set(re.findall(r"[a-z]{3,}", f"{p['host']} {p['path']} {p['title']} {p['desc']}".lower()))
is_field = lambda c: c["kind"] == "textarea" or re.fullmatch(r"input (text|search|email|password|tel|url|number)|textbox|searchbox|combobox", c["kind"]) is not None   # controls you type into
submit_like = lambda c: re.search(r"\b(send|submit|go|search|post|reply|comment|save|continue|next|sign in|log in|login|apply|ask)\b", c["label"].lower()) is not None   # reads like the action that finishes what you typed
dismissive = lambda c: re.match(r"(clear|reset|cancel|close|dismiss|decline|no thanks|skip|back)\b", c["label"].lower()) is not None
guess = lambda c: (3 if c["kind"].startswith(("input", "textarea", "select")) else 2 if c["kind"].startswith("button") else 1) + (c["box"] == "main") + 2 * on_screen(c) - c["i"] / 100   # a hand-written score, one feature among the rest
# The vocabularies come from the training pages: control kinds, page regions, and the 1,500 commonest words in labels, links and names.
KINDS = {k: i for i, (k, _) in enumerate(collections.Counter(c["kind"] for s in train for c in s["controls"]).most_common(30))}
BOXES = {k: i for i, (k, _) in enumerate(collections.Counter(c["box"] for s in train for c in s["controls"]).most_common(15))}
WORDS = {w: i for i, (w, _) in enumerate(collections.Counter(w for s in train for c in s["controls"] for w in words(c)).most_common(1500))}
PAGE_WORDS = {w: i for i, (w, _) in enumerate(collections.Counter(w for s in train for w in page_words(s["page"])).most_common(300))}
FEATURES = ["same_kind_rank", "same_kind_count", "inputs_on_screen", "screen_rank", "on_screen_count", "controls_above", "no_parent_text", "before_len", "kind", "box", "x", "y", "w", "h", "index", "index_frac", "count", "label_len", "nothing_focused", "index_delta", "guess",
            "required", "is_submit", "dismissive", "first_field", "form_fields", "focus_filled", "in_focus_form", "after_focus", "empty_fields_left", "empty_field", "focus_dx", "focus_dy", "focus_dist", "submit_like", "fill", "sat", "lum", "round", "bold", "pointer", "icon", "sat_rank", "hit", "occluded_share", "in_layer", "lcover", "lw", "lh", "lz", "ledge", "layer_size", "layer_hit_share", "same_layer", "focus_in_layer", "same_region", "region_size", "in_dialog", "bar", "fixed", "page_scroll", "foreground"]   # ranker.json carries this list; after it come one 0/1 per word
def features(step):   # one row per control except the focused one (it is already there); every count below is computed once per page, so a page of thousands of controls stays cheap
    focused = next((c for c in step["controls"] if c["i"] == step["focus"]), None); controls = [c for c in step["controls"] if c["i"] != step["focus"]]; shown = [c for c in controls if on_screen(c)]
    shown_y = sorted(c["y"] for c in shown); shown_rank = {c["i"]: r for r, c in enumerate(shown)}; inputs_shown = sum(c["kind"].startswith("input") for c in shown)
    colour = lambda c: c["sat"] if c["fill"] >= 0.5 else 0; sat_rank = {c["i"]: r for r, c in enumerate(sorted(shown, key=lambda c: -colour(c)))}   # how much a control stands out by being a coloured, filled shape; 0 = the most colourful control on screen
    kind_count = collections.Counter(c["kind"] for c in controls); kind_seen = collections.Counter(); fields = [c for c in step["controls"] if is_field(c) and c["form"] >= 0]; form_fields = collections.Counter(c["form"] for c in fields); first_field = {}   # the typing fields that sit in a form
    for c in fields: first_field.setdefault(c["form"], c["i"])
    known = [c for c in controls if c["hit"] >= 0]; occluded_share = sum(c["hit"] == 0 for c in known) / len(known) if known else 0   # how much of what is on screen is covered by something on top
    layer_size = collections.Counter(c["layer"] for c in controls if c["layer"] >= 0); layer_hits = collections.Counter(c["layer"] for c in known if c["hit"] == 1 and c["layer"] >= 0); layer_known = collections.Counter(c["layer"] for c in known if c["layer"] >= 0); region_size = collections.Counter(c["region"] for c in controls if c["region"] >= 0)
    empty_left = sum(not c["filled"] and c["form"] == focused["form"] for c in fields if c["i"] != step["focus"]) if focused and focused["form"] >= 0 else 0   # in the focused control's form
    page_bag = np.zeros(len(PAGE_WORDS)); page_bag[[PAGE_WORDS[w] for w in page_words(step["page"]) if w in PAGE_WORDS]] = 1; rows = []   # the same for every control of the page
    for c in controls:
        same_rank = kind_seen[c["kind"]]; kind_seen[c["kind"]] += 1; in_focus_form = bool(focused) and c["form"] >= 0 and c["form"] == focused["form"]
        f = dict(same_kind_rank=same_rank, same_kind_count=kind_count[c["kind"]], inputs_on_screen=inputs_shown, screen_rank=shown_rank.get(c["i"], -1), on_screen_count=len(shown),
                 controls_above=bisect.bisect_left(shown_y, c["y"]), no_parent_text=c["parent"] == "", before_len=len(c["before"]), kind=KINDS.get(c["kind"], -1), box=BOXES.get(c["box"], -1), x=c["x"] / 1280, y=c["y"] / 900, w=c["w"] / 1280, h=c["h"] / 900, index=c["i"], index_frac=c["i"] / max(1, c["n"]), count=c["n"], label_len=len(c["label"]),
                 nothing_focused=step["focus"] < 0, index_delta=c["i"] - step["focus"] if step["focus"] >= 0 else 0, guess=guess(c), required=c["required"], is_submit=c["submit"], dismissive=dismissive(c), first_field=is_field(c) and c["form"] >= 0 and first_field[c["form"]] == c["i"],
                 form_fields=form_fields[c["form"]] if c["form"] >= 0 else 0, focus_filled=bool(focused and focused["filled"]), in_focus_form=in_focus_form, after_focus=in_focus_form and c["i"] > step["focus"], empty_fields_left=empty_left, empty_field=is_field(c) and not c["filled"],
                 focus_dx=(c["x"] - focused["x"]) / 1280 if focused else 0, focus_dy=(c["y"] - focused["y"]) / 900 if focused else 0, focus_dist=math.hypot(c["x"] - focused["x"], c["y"] - focused["y"]) / 1000 if focused else 0, submit_like=submit_like(c),   # where the control sits relative to the one you are in
                 fill=c["fill"], sat=c["sat"], lum=c["lum"], round=c["round"], bold=c["bold"], pointer=c["pointer"], icon=c["icon"], sat_rank=sat_rank.get(c["i"], -1), hit=c["hit"], occluded_share=occluded_share, in_layer=c["layer"] >= 0, lcover=c["lcover"], lw=c["lw"], lh=c["lh"], lz=c["lz"], ledge=c["ledge"],   # how it looks; where it sits in the stack of layers above the page
                 layer_size=layer_size[c["layer"]] if c["layer"] >= 0 else 0, layer_hit_share=layer_hits[c["layer"]] / layer_known[c["layer"]] if layer_known[c["layer"]] else 1, same_layer=bool(focused) and c["layer"] >= 0 and c["layer"] == focused["layer"], focus_in_layer=bool(focused) and focused["layer"] >= 0,
                 same_region=bool(focused) and c["region"] >= 0 and c["region"] == focused["region"], region_size=region_size[c["region"]] if c["region"] >= 0 else 0, in_dialog=c["dialog"], bar=c["bar"], fixed=c["fixed"], page_scroll=step["page"].get("scroll", 1.0),
                 foreground=occluded_share if c["layer"] >= 0 and c["hit"] == 1 else 0); bag = np.zeros(len(WORDS)); bag[[WORDS[w] for w in words(c) if w in WORDS]] = 1; rows.append(np.concatenate([[float(f[name]) for name in FEATURES], bag, page_bag]))   # in a layer of its own, on top, with the page behind it covered: what the site is presenting
    return np.array(rows), [c["i"] for c in controls]
def relevance(s, ids):   # 3 the control wanted next, 2 the picks still ahead of it, 1 the also-fine ones, 0 the rest (and the picks already visited)
    p = s["picks"]; ahead = set(p[p.index(s["target"]) + 1:]) if s["target"] in p else set(); ok = set(s.get("ok", []))
    return [3 if i == s["target"] else 2 if i in ahead else 1 if i in ok else 0 for i in ids]
state = pathlib.Path(out + ".state"); state.mkdir(exist_ok=True)   # base models: trained once per data and features, then kept so case studies can be applied on top any number of times
fingerprint = hashlib.sha1(pathlib.Path(data).read_bytes() + repr((FEATURES, args.all, 2)).encode()).hexdigest()
if (state / "fingerprint").exists() and (state / "fingerprint").read_text() == fingerprint:
    base = [lgb.Booster(model_file=str(state / f"base-{name}.txt")) for name in ("ranker", "classifier")]; print("base models loaded from", state)
else:
    rows = [features(s) for s in train]; X = np.concatenate([r[0] for r in rows]); y = np.concatenate([relevance(s, r[1]) for s, r in zip(train, rows)])
    params = dict(n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
    base = [lgb.LGBMRanker(**params).fit(X, y, group=[len(r[1]) for r in rows]).booster_, lgb.LGBMClassifier(**params).fit(X, y >= 2).booster_]; (state / "fingerprint").write_text(fingerprint)   # one learns the order, one learns "is this one of the picks?"
    for name, model in zip(("ranker", "classifier"), base): model.save_model(str(state / f"base-{name}.txt"))
z = lambda v: (v - v.mean()) / (v.std() + 1e-9); scores = lambda models, rows: z(models[0].predict(rows)) + z(models[1].predict(rows, raw_score=True))   # each model's scores, standardised within the page, then added
def accuracy(models, prepared):   # on landing (focus -1): first stop wanted, share of the 4 stops wanted; every step: target first, target within 4
    land, every = np.zeros(2), np.zeros(2); n = 0
    for (rows, ids), s in prepared:
        order = [ids[j] for j in np.argsort(-scores(models, rows))]; every += [order[0] == s["target"], s["target"] in order[:4]]
        if s["focus"] < 0 and not s.get("accept"): wanted = {*s["picks"], *s.get("ok", [])}; n += 1; land += [order[0] in wanted, len(wanted & set(order[:4])) / 4]
    return [*(land / max(n, 1)), *(every / len(prepared))]
prepare = lambda steps_: [(features(s), s) for s in steps_]; show = lambda name, *r: print(name + ":", "   ->   ".join(f"first stop wanted {a[0]:.3f}  stops wanted {a[1]:.3f}  top-1 {a[2]:.3f}  top-4 {a[3]:.3f}" for a in r))
ledger = state / "cases"; ledger.mkdir(exist_ok=True)   # the case-study step, applied on top of the base models (re-entrant: the same ledger always gives the same result)
if args.rollback: [path.unlink() for path in ledger.iterdir()]; print("rolled back: the case-study ledger is empty")
for path in args.cases: shutil.copy(path, ledger / (hashlib.sha1(pathlib.Path(path).read_bytes()).hexdigest()[:12] + pathlib.Path(path).suffix))   # the same file twice is one entry
cases = [s for path in sorted(ledger.iterdir()) for s in read(path)]; held_rows = prepare(held); cases_eval = prepare(read(args.cases_eval)) if args.cases_eval else []; models = base
if cases:
    print(f"case-study ledger: {len(cases)} steps from {len(list(ledger.iterdir()))} file(s)"); replay = random.Random(0).sample(train, min(1500, len(train))); weight = args.case_share * len(replay) / len(cases)   # the cases get this share of the update's weight
    groups = [(features(s), s, 1.0) for s in replay] + [(features(s), s, weight) for s in cases]; X = np.concatenate([g[0][0] for g in groups]); sizes = [len(g[0][1]) for g in groups]; accepted = [np.array([i in {g[1]["target"], *g[1].get("accept", [])} for i in g[0][1]], dtype=float) for g in groups]; weights = [g[2] for g in groups]
    def policy_gradient(preds, _):   # loss = -log P(an acceptable control is chosen) under softmax(scores) within each page, weighted per step
        grad, hess, at = np.zeros_like(preds), np.zeros_like(preds), 0
        for size, ok, w in zip(sizes, accepted, weights):
            s = preds[at:at + size]; p = np.exp(s - s.max()); p /= p.sum(); mass = max((p * ok).sum(), 1e-9); grad[at:at + size] = w * (p - p * ok / mass); hess[at:at + size] = w * np.maximum(p * (1 - p), 1e-6); at += size
        return grad, hess
    tune = dict(objective=policy_gradient, learning_rate=args.lr, num_leaves=15, min_data_in_leaf=5, feature_fraction=0.8, verbose=-1)
    models = [lgb.train(tune, lgb.Dataset(X, label=np.zeros(len(X)), free_raw_data=False), num_boost_round=args.rounds, init_model=b) for b in base]
    case_rows = prepare(cases); show("held-out steps", accuracy(base, held_rows), accuracy(models, held_rows)); show(f"{len(cases)} case steps", accuracy(base, case_rows), accuracy(models, case_rows))
    if cases_eval: show("unseen case steps", accuracy(base, cases_eval), accuracy(models, cases_eval))
else: show(f"{len(held)} held-out steps{' (also trained on)' if args.all else ''}", accuracy(base, held_rows))
def flatten(node, f, thr, left, right, leaf):   # ranker.json: each tree as flat arrays [feature, threshold, left, right, leaf values]; a negative child ~n is leaf n
    if "leaf_index" in node: leaf.append(round(node["leaf_value"], 5)); return ~(len(leaf) - 1)
    i = len(f); f.append(node["split_feature"]); thr.append(node["threshold"]); left.append(0); right.append(0); left[i] = flatten(node["left_child"], f, thr, left, right, leaf); right[i] = flatten(node["right_child"], f, thr, left, right, leaf); return i
exported = [[(lambda t: (flatten(info["tree_structure"], *t), list(t))[1])(([], [], [], [], [])) for info in m.dump_model()["tree_info"]] for m in models]
used = sorted({f - len(FEATURES) for trees in exported for t in trees for f in t[0] if f >= len(FEATURES)}); renumber = {len(FEATURES) + w: len(FEATURES) + j for j, w in enumerate(used)}   # only the words some tree asks about
for t in [t for trees in exported for t in trees]: t[0] = [renumber.get(f, f) for f in t[0]]
by_index = {**{i: "w:" + w for w, i in WORDS.items()}, **{len(WORDS) + i: "p:" + w for w, i in PAGE_WORDS.items()}}   # "w:" a word of the control, "p:" a word of the page
json.dump({"features": FEATURES, "kinds": KINDS, "boxes": BOXES, "words": [by_index[w] for w in used], "models": exported}, open(out, "w"), separators=(",", ":"))
print(f"{len(exported)} models of {len(exported[0])} trees and {len(used)} words written to {out}")
