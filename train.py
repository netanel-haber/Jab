# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "lightgbm", "scikit-learn"]
# ///
"""jab, which control should Tab reach next?   uv run train.py DATA.jsonl.gz ranker.json [--all]
Trains a LightGBM ranker (LambdaRank) that scores every control of a page, and writes the trees as ranker.json for jab.js to run in the browser.
DATA is not included: one JSON object per line, one per labelled step: {"split": "train"|"eval", "controls": [what jab.js collect() returns for the page, up to 80],
"focus": index of the focused control or -1, "target": index of the control a person would want next, "picks": all the controls the labeller picked}.
Prints top-1 / top-3 on the "eval" pages. --all also trains on them, for the file you ship."""
import collections, gzip, json, re, sys
import numpy as np, lightgbm as lgb

data, out = sys.argv[1:3]
steps = [json.loads(line) for line in gzip.open(data, "rt", encoding="utf-8")]
train = [s for s in steps if "--all" in sys.argv or s["split"] == "train"]
held = [s for s in steps if s["split"] == "eval"]

PLACE = {"top": 0, "middle": 1, "bottom": 2, "left": 0, "center": 1, "right": 2}
SIZE = {"small": 0, "medium": 1, "large": 2}
on_screen = lambda c: c["w"] >= 4 and c["h"] >= 4 and c["y"] + c["h"] >= 0 and c["y"] <= 900 and c["x"] <= 1280
words = lambda c: set(re.findall(r"[a-z]{3,}", f"{c['label']} {c['href']} {c['name']} {c['ph']}".lower()))
def guess(c):   # a hand-written score (fields, buttons, links; main area; on screen; not a skip link), one feature among the rest
    return (3 if c["kind"].startswith(("input", "textarea", "select")) else 2 if c["kind"].startswith("button") else 1) + ((c["box"] or {}).get("kind") == "main") + 2 * on_screen(c) - 5 * c["skip"] - c["i"] / 100

# The vocabularies come from the training pages: control kinds, page regions, and the 1,500 commonest words in labels, links and names.
KINDS = {k: i for i, (k, _) in enumerate(collections.Counter(c["kind"] for s in train for c in s["controls"]).most_common(30))}
BOXES = {k: i for i, (k, _) in enumerate(collections.Counter((c["box"] or {}).get("kind", "") for s in train for c in s["controls"]).most_common(15))}
WORDS = {w: i for i, (w, _) in enumerate(collections.Counter(w for s in train for c in s["controls"] for w in words(c)).most_common(1500))}
FEATURES = ["same_kind_rank", "same_kind_count", "inputs_on_screen", "screen_rank", "on_screen_count", "controls_above", "no_parent_text", "before_len", "box_titled", "depth", "kind", "box", "on_screen", "x", "y", "w", "h",
            "vertical", "horizontal", "size", "skip", "required", "autofocus", "filled", "index", "index_frac", "count", "label_len", "has_placeholder", "has_href", "nothing_focused", "index_delta", "guess"]   # ranker.json carries this list; after it come one 0/1 per word

def features(step):   # one row per control except the focused one (it is already there)
    controls = [c for c in step["controls"] if c["i"] != step["focus"]]
    shown = [c for c in controls if on_screen(c)]
    rows = []
    for c in controls:
        same = [d for d in controls if d["kind"] == c["kind"]]
        vertical, horizontal = c["place"].split("-")
        f = dict(same_kind_rank=same.index(c), same_kind_count=len(same), inputs_on_screen=sum(d["kind"].startswith("input") for d in shown), screen_rank=shown.index(c) if c in shown else -1, on_screen_count=len(shown),
                 controls_above=sum(d["y"] < c["y"] for d in shown), no_parent_text=c["parent"] == "", before_len=len(c["before"]), box_titled=(c["box"] or {}).get("title", "") != "", depth=len(c["up"].split("<")),
                 kind=KINDS.get(c["kind"], -1), box=BOXES.get((c["box"] or {}).get("kind", ""), -1), on_screen=on_screen(c), x=c["x"] / 1280, y=c["y"] / 900, w=c["w"] / 1280, h=c["h"] / 900,
                 vertical=PLACE[vertical], horizontal=PLACE[horizontal], size=SIZE[c["size"]], skip=c["skip"], required="required" in c["flags"], autofocus="autofocus" in c["flags"], filled="filled" in c["flags"],
                 index=c["i"], index_frac=c["i"] / max(1, c["n"]), count=c["n"], label_len=len(c["label"]), has_placeholder=bool(c["ph"]), has_href=bool(c["href"]), nothing_focused=step["focus"] < 0,
                 index_delta=c["i"] - step["focus"] if step["focus"] >= 0 else 0, guess=guess(c))
        bag = np.zeros(len(WORDS)); bag[[WORDS[w] for w in words(c) if w in WORDS]] = 1
        rows.append(np.concatenate([[float(f[name]) for name in FEATURES], bag]))
    return np.array(rows), [c["i"] for c in controls]

X, y, group = [], [], []
for step in train:   # relevance 2 for the labelled target, 1 for the labeller's other picks, 0 for the rest
    rows, ids = features(step)
    X.append(rows); y.append([2 if i == step["target"] else 1 if i in step["picks"] else 0 for i in ids]); group.append(len(ids))
ranker = lgb.LGBMRanker(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1)
ranker.fit(np.concatenate(X), np.concatenate(y), group=group)

top1 = top3 = 0
for step in held:
    rows, ids = features(step)
    order = [ids[j] for j in np.argsort(-ranker.predict(rows))]
    top1 += order[0] == step["target"]; top3 += step["target"] in order[:3]
print(f"{len(held)} held-out steps{' (also trained on)' if '--all' in sys.argv else ''}: top-1 {top1 / len(held):.3f}, top-3 {top3 / len(held):.3f}")

# ranker.json: every tree as flat arrays [feature, threshold, left, right, leaf values] (a negative child ~n is leaf n), and only the words some tree asks about.
def flatten(node, f, thr, left, right, leaf):
    if "leaf_index" in node: leaf.append(round(node["leaf_value"], 5)); return ~(len(leaf) - 1)
    i = len(f); f.append(node["split_feature"]); thr.append(node["threshold"]); left.append(0); right.append(0)
    left[i] = flatten(node["left_child"], f, thr, left, right, leaf); right[i] = flatten(node["right_child"], f, thr, left, right, leaf)
    return i
trees = []
for info in ranker.booster_.dump_model()["tree_info"]:
    tree = ([], [], [], [], []); flatten(info["tree_structure"], *tree); trees.append(list(tree))
used = sorted({f - len(FEATURES) for t in trees for f in t[0] if f >= len(FEATURES)})
renumber = {len(FEATURES) + w: len(FEATURES) + j for j, w in enumerate(used)}
for t in trees: t[0] = [renumber.get(f, f) for f in t[0]]
by_index = {i: w for w, i in WORDS.items()}
json.dump({"features": FEATURES, "kinds": KINDS, "boxes": BOXES, "words": [by_index[w] for w in used], "trees": trees}, open(out, "w"), separators=(",", ":"))
print(f"{len(trees)} trees and {len(used)} words written to {out}")
