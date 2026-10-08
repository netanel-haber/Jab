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

import argparse
import bisect
import collections
import gzip
import hashlib
import json
import math
import pathlib
import random
import re
import shutil

import lightgbm as lgb
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("data")
ap.add_argument("out")
ap.add_argument("--all", action="store_true")
ap.add_argument("--cases", nargs="*", default=[])
ap.add_argument("--cases-eval")
ap.add_argument("--rollback", action="store_true")
ap.add_argument("--rounds", type=int, default=100)
ap.add_argument("--case-share", type=float, default=0.2)
ap.add_argument("--lr", type=float, default=0.05)
args = ap.parse_args()


def read(path):
    with (gzip.open if str(path).endswith(".gz") else open)(path, "rt", encoding="utf-8") as lines:
        return [json.loads(line) for line in lines]


steps = read(args.data)
train = [s for s in steps if args.all or s["split"] == "train"]
held = [s for s in steps if s["split"] == "eval"]

matches = lambda pattern, text: re.search(pattern, text) is not None
words_of = lambda text: set(re.findall(r"[a-z]{3,}", text.lower()))
words = lambda c: words_of(f"{c['label']} {c['href']} {c['name']} {c['ph']} {c['action']}")
page_words = lambda p: words_of(f"{p['host']} {p['path']} {p['title']} {p['desc']}")
on_screen = lambda c: c["w"] >= 4 and c["h"] >= 4 and c["y"] + c["h"] >= 0 and c["y"] <= 900 and c["x"] <= 1280
# controls you type into
is_field = lambda c: matches(
    r"^(textarea|input (text|search|email|password|tel|url|number)|textbox|searchbox|combobox)$",
    c["kind"],
)
# reads like the action that finishes what you typed
submit_like = lambda c: matches(
    r"\b(send|submit|go|search|post|reply|comment|save|continue|next|sign in|log in|login|apply|ask)\b",
    c["label"].lower(),
)
dismissive = lambda c: matches(
    r"^(clear|reset|cancel|close|dismiss|decline|no thanks|skip|back)\b",
    c["label"].lower(),
)
# a hand-written score, one feature among the rest
guess = lambda c: (
    (3 if c["kind"].startswith(("input", "textarea", "select")) else 2 if c["kind"].startswith("button") else 1)
    + (c["box"] == "main")
    + 2 * on_screen(c)
    - c["i"] / 100
)

# The vocabularies come from the training pages: control kinds, page regions, the 1,500 commonest words of controls and the 300 of pages.
# Ties go alphabetically, so the same data always gives the same columns (a set's order changes from run to run).
vocabulary = lambda items, n: {
    k: i for i, (k, _) in enumerate(sorted(collections.Counter(items).items(), key=lambda kv: (-kv[1], kv[0]))[:n])
}
KINDS = vocabulary((c["kind"] for s in train for c in s["controls"]), 30)
BOXES = vocabulary((c["box"] for s in train for c in s["controls"]), 15)
WORDS = vocabulary((w for s in train for c in s["controls"] for w in words(c)), 1500)
PAGE_WORDS = vocabulary((w for s in train for w in page_words(s["page"])), 300)


def describe(step):
    """One dict of features per control except the focused one (it is already there), in the order ranker.json lists them.
    Every count is computed once per page, so a page of thousands of controls stays cheap."""
    focus = step["focus"]
    focused = next((c for c in step["controls"] if c["i"] == focus), None)
    controls = [c for c in step["controls"] if c["i"] != focus]
    shown = [c for c in controls if on_screen(c)]
    shown_y = sorted(c["y"] for c in shown)
    shown_rank = {c["i"]: r for r, c in enumerate(shown)}
    inputs_shown = sum(c["kind"].startswith("input") for c in shown)
    # how much a control stands out by being a coloured, filled shape; 0 = the most colourful control on screen
    colour = lambda c: c["sat"] if c["fill"] >= 0.5 else 0
    sat_rank = {c["i"]: r for r, c in enumerate(sorted(shown, key=lambda c: -colour(c)))}
    kind_count = collections.Counter(c["kind"] for c in controls)
    kind_seen = collections.Counter()
    fields = [c for c in step["controls"] if is_field(c) and c["form"] >= 0]  # the typing fields that sit in a form
    form_fields = collections.Counter(c["form"] for c in fields)
    first_field = {}
    for c in fields:
        first_field.setdefault(c["form"], c["i"])
    # how much of what is on screen is covered by something on top
    known = [c for c in controls if c["hit"] >= 0]
    occluded_share = sum(c["hit"] == 0 for c in known) / len(known) if known else 0
    layer_size = collections.Counter(c["layer"] for c in controls if c["layer"] >= 0)
    layer_hits = collections.Counter(c["layer"] for c in known if c["hit"] == 1 and c["layer"] >= 0)
    layer_known = collections.Counter(c["layer"] for c in known if c["layer"] >= 0)
    region_size = collections.Counter(c["region"] for c in controls if c["region"] >= 0)
    in_form = lambda c, form: form >= 0 and c["form"] == form
    focus_form = focused["form"] if focused else -1
    empty_left = sum(not c["filled"] and in_form(c, focus_form) for c in fields if c["i"] != focus)
    same = lambda c, key: (
        bool(focused) and c[key] >= 0 and c[key] == focused[key]
    )  # in the same layer or region as the focused control
    for c in controls:
        same_rank = kind_seen[c["kind"]]
        kind_seen[c["kind"]] += 1
        in_focus_form = in_form(c, focus_form)
        yield (
            c,
            {
                "same_kind_rank": same_rank,
                "same_kind_count": kind_count[c["kind"]],
                "inputs_on_screen": inputs_shown,
                "screen_rank": shown_rank.get(c["i"], -1),
                "on_screen_count": len(shown),
                "controls_above": bisect.bisect_left(shown_y, c["y"]),
                "no_parent_text": c["parent"] == "",
                "before_len": len(c["before"]),
                "kind": KINDS.get(c["kind"], -1),
                "box": BOXES.get(c["box"], -1),
                "x": c["x"] / 1280,
                "y": c["y"] / 900,
                "w": c["w"] / 1280,
                "h": c["h"] / 900,
                "index": c["i"],
                "index_frac": c["i"] / max(1, c["n"]),
                "count": c["n"],
                "label_len": len(c["label"]),
                "nothing_focused": focus < 0,
                "index_delta": c["i"] - focus if focus >= 0 else 0,
                "guess": guess(c),
                # its form: is it the submit button, the first field, how much is still empty
                "required": c["required"],
                "is_submit": c["submit"],
                "dismissive": dismissive(c),
                "first_field": is_field(c) and c["form"] >= 0 and first_field[c["form"]] == c["i"],
                "form_fields": form_fields[c["form"]] if c["form"] >= 0 else 0,
                "focus_filled": bool(focused and focused["filled"]),
                "in_focus_form": in_focus_form,
                "after_focus": in_focus_form and c["i"] > focus,
                "empty_fields_left": empty_left,
                "empty_field": is_field(c) and not c["filled"],
                # where it sits relative to the control you are in
                "focus_dx": (c["x"] - focused["x"]) / 1280 if focused else 0,
                "focus_dy": (c["y"] - focused["y"]) / 900 if focused else 0,
                "focus_dist": math.hypot(c["x"] - focused["x"], c["y"] - focused["y"]) / 1000 if focused else 0,
                "submit_like": submit_like(c),
                # how it looks
                "fill": c["fill"],
                "sat": c["sat"],
                "lum": c["lum"],
                "round": c["round"],
                "bold": c["bold"],
                "pointer": c["pointer"],
                "icon": c["icon"],
                "sat_rank": sat_rank.get(c["i"], -1),
                # where it sits in the stack of layers above the page
                "hit": c["hit"],
                "occluded_share": occluded_share,
                "in_layer": c["layer"] >= 0,
                "lcover": c["lcover"],
                "lw": c["lw"],
                "lh": c["lh"],
                "lz": c["lz"],
                "ledge": c["ledge"],
                "layer_size": layer_size[c["layer"]] if c["layer"] >= 0 else 0,
                "layer_hit_share": layer_hits[c["layer"]] / layer_known[c["layer"]] if layer_known[c["layer"]] else 1,
                "same_layer": same(c, "layer"),
                "focus_in_layer": bool(focused) and focused["layer"] >= 0,
                "same_region": same(c, "region"),
                "region_size": region_size[c["region"]] if c["region"] >= 0 else 0,
                "in_dialog": c["dialog"],
                "bar": c["bar"],
                "fixed": c["fixed"],
                "page_scroll": step["page"].get("scroll", 1.0),
                # in a layer of its own, on top, with the page behind it covered: what the site is presenting
                "foreground": occluded_share if c["layer"] >= 0 and c["hit"] == 1 else 0,
            },
        )


FEATURES = list(
    next(f for s in train for _, f in describe(s))
)  # ranker.json carries this list; after it come one 0/1 per word


def features(step):
    """The rows the models see: the features, then a 0/1 per word of the control, then a 0/1 per word of the page."""
    page_bag = np.zeros(len(PAGE_WORDS))
    page_bag[[PAGE_WORDS[w] for w in page_words(step["page"]) if w in PAGE_WORDS]] = 1
    ids, rows = [], []
    for c, f in describe(step):
        bag = np.zeros(len(WORDS))
        bag[[WORDS[w] for w in words(c) if w in WORDS]] = 1
        ids.append(c["i"])
        rows.append(np.concatenate([list(map(float, f.values())), bag, page_bag]))
    return np.array(rows), ids


def relevance(s, ids):
    """3 the control wanted next, 2 the picks still ahead of it, 1 the also-fine ones, 0 the rest (and the picks already visited)."""
    p = s["picks"]
    ahead = set(p[p.index(s["target"]) + 1 :]) if s["target"] in p else set()
    ok = set(s.get("ok", []))
    return [3 if i == s["target"] else 2 if i in ahead else 1 if i in ok else 0 for i in ids]


# Base models: trained once per data and features, then kept so case studies can be applied on top any number of times.
# One learns the order, one learns "is this one of the picks?".
state = pathlib.Path(args.out + ".state")
state.mkdir(exist_ok=True)
NAMES = ("ranker", "classifier")
fingerprint = hashlib.sha1(
    pathlib.Path(args.data).read_bytes() + repr((FEATURES, WORDS, PAGE_WORDS, args.all, 3)).encode()
).hexdigest()
if (state / "fingerprint").exists() and (state / "fingerprint").read_text() == fingerprint:
    base = [lgb.Booster(model_file=str(state / f"base-{name}.txt")) for name in NAMES]
    print("base models loaded from", state)
else:
    rows = [features(s) for s in train]
    X = np.concatenate([r[0] for r in rows])
    y = np.concatenate([relevance(s, r[1]) for s, r in zip(train, rows)])
    params = {
        "n_estimators": 400,
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 10,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.8,
        "random_state": 0,
        "deterministic": True,  # with row-wise histograms, the same data gives the same trees on every run
        "force_row_wise": True,
        "verbose": -1,
    }
    base = [
        lgb.LGBMRanker(**params).fit(X, y, group=[len(r[1]) for r in rows]).booster_,
        lgb.LGBMClassifier(**params).fit(X, y >= 2).booster_,
    ]
    for name, model in zip(NAMES, base):
        model.save_model(str(state / f"base-{name}.txt"))
    (state / "fingerprint").write_text(fingerprint)

z = lambda v: (v - v.mean()) / (v.std() + 1e-9)
# each model's scores, standardised within the page, then added
scores = lambda models, rows: z(models[0].predict(rows)) + z(models[1].predict(rows, raw_score=True))


def accuracy(models, prepared):
    """On landing (focus -1): first stop wanted, share of the 4 stops wanted. Every step: target first, target within 4."""
    land, every, n = np.zeros(2), np.zeros(2), 0
    for (rows, ids), s in prepared:
        order = [ids[j] for j in np.argsort(-scores(models, rows))]
        every += [order[0] == s["target"], s["target"] in order[:4]]
        if s["focus"] < 0 and not s.get("accept"):
            wanted = {*s["picks"], *s.get("ok", [])}
            n += 1
            land += [order[0] in wanted, len(wanted & set(order[:4])) / 4]
    return [*(land / max(n, 1)), *(every / len(prepared))]


prepare = lambda steps_: [(features(s), s) for s in steps_]

# The case-study step, applied on top of the base models (re-entrant: the same ledger always gives the same result).
ledger = state / "cases"
ledger.mkdir(exist_ok=True)
if args.rollback:
    for path in ledger.iterdir():
        path.unlink()
    print("rolled back: the case-study ledger is empty")
for path in map(pathlib.Path, args.cases):  # named by content: the same file twice is one entry
    shutil.copy(path, ledger / (hashlib.sha1(path.read_bytes()).hexdigest()[:12] + path.suffix))
cases = [s for path in sorted(ledger.iterdir()) for s in read(path)]
models = base
if cases:
    print(f"case-study ledger: {len(cases)} steps from {len(list(ledger.iterdir()))} file(s)")
    replay = random.Random(0).sample(train, min(1500, len(train)))
    weight = args.case_share * len(replay) / len(cases)  # the cases get this share of the update's weight
    groups = [
        (
            features(s),
            {s["target"], *s.get("accept", [])},
            1.0 if k < len(replay) else weight,
        )
        for k, s in enumerate(replay + cases)
    ]
    X = np.concatenate([rows for (rows, _), _, _ in groups])
    blocks = [(len(ids), np.array([i in ok for i in ids], dtype=float), w) for (_, ids), ok, w in groups]

    def policy_gradient(preds, _):
        """loss = -log P(an acceptable control is chosen) under softmax(scores) within each page, weighted per step"""
        grad, hess, at = np.zeros_like(preds), np.zeros_like(preds), 0
        for size, ok, w in blocks:
            p = np.exp(preds[at : at + size] - preds[at : at + size].max())
            p /= p.sum()
            grad[at : at + size] = w * (p - p * ok / max((p * ok).sum(), 1e-9))
            hess[at : at + size] = w * np.maximum(p * (1 - p), 1e-6)
            at += size
        return grad, hess

    tune = {
        "objective": policy_gradient,
        "learning_rate": args.lr,
        "num_leaves": 15,
        "min_data_in_leaf": 5,
        "feature_fraction": 0.8,
        "seed": 0,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }
    dataset = lgb.Dataset(X, label=np.zeros(len(X)), free_raw_data=False)
    models = [lgb.train(tune, dataset, num_boost_round=args.rounds, init_model=b) for b in base]

for name, which in [
    (f"{len(held)} held-out steps{' (also trained on)' if args.all else ''}", held),
    (f"{len(cases)} case steps", cases),
    ("unseen case steps", read(args.cases_eval) if args.cases_eval else []),
]:
    if which:  # before -> after the case-study step, when there is one
        rows = prepare(which)
        results = [accuracy(base, rows)] + ([accuracy(models, rows)] if cases else [])
        print(
            f"{name}:",
            "   ->   ".join(
                f"first stop wanted {a[0]:.3f}  stops wanted {a[1]:.3f}  top-1 {a[2]:.3f}  top-4 {a[3]:.3f}"
                for a in results
            ),
        )


def flatten(node, tree):
    """ranker.json: each tree as flat arrays [feature, threshold, left, right, leaf values]; a negative child ~n is leaf n"""
    f, thr, left, right, leaf = tree
    if "leaf_index" in node:
        leaf.append(round(node["leaf_value"], 5))
        return ~(len(leaf) - 1)
    i = len(f)
    f.append(node["split_feature"])
    thr.append(node["threshold"])
    left.append(0)
    right.append(0)
    left[i] = flatten(node["left_child"], tree)
    right[i] = flatten(node["right_child"], tree)
    return i


exported = []
for model in models:
    exported.append([])
    for info in model.dump_model()["tree_info"]:
        tree = ([], [], [], [], [])
        flatten(info["tree_structure"], tree)
        exported[-1].append(list(tree))
# only the words some tree asks about go in ranker.json: "w:" a word of the control, "p:" a word of the page
used = sorted({f for trees in exported for t in trees for f in t[0] if f >= len(FEATURES)})
renumber = {f: len(FEATURES) + j for j, f in enumerate(used)}
for t in (t for trees in exported for t in trees):
    t[0] = [renumber.get(f, f) for f in t[0]]
names = {
    **{len(FEATURES) + i: "w:" + w for w, i in WORDS.items()},
    **{len(FEATURES) + len(WORDS) + i: "p:" + w for w, i in PAGE_WORDS.items()},
}
ranker = {
    "features": FEATURES,
    "kinds": KINDS,
    "boxes": BOXES,
    "words": [names[f] for f in used],
    "models": exported,
}
pathlib.Path(args.out).write_text(json.dumps(ranker, separators=(",", ":")))
print(f"{len(exported)} models of {len(exported[0])} trees and {len(used)} words written to {args.out}")
