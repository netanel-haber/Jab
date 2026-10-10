# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "scipy", "lightgbm", "scikit-learn", "regex"]
# ///
"""jab: which controls does a person want next?   uv run train.py steps.jsonl.gz ranker.json [--all]

Trains two LightGBM models over every control of a page, a LambdaRank ranker and a "how much is it wanted" model, and writes their
trees as ranker.json for jab.js, which adds their scores (each standardised within the page). The shipped file: --all.
steps.jsonl.gz (not included): 6,333 steps on 1,244 real sites, labelled by Claude from screenshots, and 8,674 steps on 5,000
generated pages (news, results, articles, feeds, listings, docs; landing pages, forms, search pages, apps, products, cookie
pop-ups; six languages), labelled by rule, collected by jab.js's own collector. One labelled step per line:
{"split": "train"|"eval", "type": the kind of page ("real" for a real site), "page": {"path", "title", "desc", "scroll", "h1",
"query", "prose", "base"}, "controls": [jab.js collect()], "focus": the focused control or -1, "target": the control wanted next,
"picks": the ordered 4 a person would want, "ok": others that are also fine}.
Prints, for the "eval" steps, overall and per kind of page: on landing, whether the first stop is wanted and how many of the 4 stops
are; over all steps, top-1 and top-4. --all also trains on them, for the file you ship."""

import argparse
import bisect
import collections
import gzip
import json
import math
import pathlib
import re

import lightgbm as lgb
import numpy as np
import regex
import scipy.sparse as sp

ap = argparse.ArgumentParser()
ap.add_argument("steps")
ap.add_argument("out")
ap.add_argument("--all", action="store_true")
args = ap.parse_args()

with gzip.open(args.steps, "rt", encoding="utf-8") as lines:
    steps = [json.loads(line) for line in lines]
train = [s for s in steps if args.all or s["split"] == "train"]
held = [s for s in steps if s["split"] == "eval"]

# Words in any script (two letters or more, vowel marks kept), without the commonest little words.
STOP = set(
    """the and for with from that this your you our are was were has have had not but all any can will into about its his her
    their they them there here than then also just been being more most very what when who how which where why per via off
    el la los las de del en un una uno unos que por con para es se al lo su sus le les des du et au aux est dans qui sur pas
    der die das und den dem des ein eine einer mit von zu im ist auf für nicht sie es wir ihr
    של את על עם זה זו כל או גם לא הוא היא אם כי
    и в во на с со по к ко из от для не что это как""".split()
)
words = lambda text: [w for w in regex.findall(r"\p{L}[\p{L}\p{M}]+", text.lower()) if w not in STOP]
control_words = lambda c: set(words(f"{c['label']} {c['href']} {c['name']} {c['ph']} {c['action']}"))
context_words = lambda c: set(words(f"{c['before']} {c['parent']} {c['around']}"))
page_words = lambda p: set(words(f"{p['path']} {p['title']} {p['desc']} {p['h1']} {p['query']}"))
href_path = lambda c: c["href"].split("#")[0].split("?")[0]  # where a link leads, without its query and fragment

matches = lambda pattern, text: re.search(pattern, text) is not None
on_screen = lambda c: c["w"] >= 4 and c["h"] >= 4 and c["y"] + c["h"] >= 0 and c["y"] <= 900 and c["x"] <= 1280
# controls you type into
is_field = lambda c: matches(
    r"^(textarea|input (text|search|email|password|tel|url|number)|textbox|searchbox|combobox)$", c["kind"]
)
# reads like the action that finishes what you typed, or like turning something down
submit_like = lambda c: matches(
    r"\b(send|submit|go|search|post|reply|comment|save|continue|next|sign in|log in|login|apply|ask)\b",
    c["label"].lower(),
)
dismissive = lambda c: matches(r"^(clear|reset|cancel|close|dismiss|decline|no thanks|skip|back)\b", c["label"].lower())


def vocabulary(items, n):
    """the n commonest, ties alphabetically, so the same data always gives the same columns"""
    counts = collections.Counter(items)
    return {k: i for i, k in enumerate(sorted(counts, key=lambda k: (-counts[k], k))[:n])}


KINDS = vocabulary((c["kind"] for s in train for c in s["controls"]), 30)
BOXES = vocabulary((c["box"] for s in train for c in s["controls"]), 15)
# the columns after the features: a word of the control ("w:"), of the text around it ("c:"), of the page ("p:")
WORDS = [
    *(f"w:{w}" for w in vocabulary((w for s in train for c in s["controls"] for w in control_words(c)), 1500)),
    *(f"c:{w}" for w in vocabulary((w for s in train for c in s["controls"] for w in context_words(c)), 400)),
    *(f"p:{w}" for w in vocabulary((w for s in train for w in page_words(s["page"])), 300)),
]
COLUMN = {w: j for j, w in enumerate(WORDS)}


def describe(step):
    """One dict of features per control except the focused one (it is already there), in the order ranker.json lists them.
    Every count is computed once per page, so a page of thousands of controls stays cheap."""
    page, focus = step["page"], step["focus"]
    focused = next((c for c in step["controls"] if c["i"] == focus), None)
    controls = [c for c in step["controls"] if c["i"] != focus]
    shown = [c for c in controls if on_screen(c)]
    shown_y = sorted(c["y"] for c in shown)
    rank_of = lambda ordered: {c["i"]: r for r, c in enumerate(ordered)}
    shown_rank = rank_of(shown)
    inputs_shown = sum(c["kind"].startswith("input") for c in shown)
    # how much a control stands out by being a coloured, filled shape; 0 = the most colourful control on screen
    sat_rank = rank_of(sorted(shown, key=lambda c: -(c["sat"] if c["fill"] >= 0.5 else 0)))
    font_rank = rank_of(sorted(shown, key=lambda c: -c["font"]))  # 0 = the largest type on screen
    heading_rank = rank_of(c for c in controls if c["head"])
    content_rank = rank_of(c for c in controls if (c["art"] or c["main"]) and not c["chrome"])
    kind_count = collections.Counter(c["kind"] for c in controls)
    kind_seen = collections.Counter()
    label_count = collections.Counter(c["label"].strip().lower() for c in controls)
    href_count = collections.Counter(href_path(c) for c in controls)
    fields = [c for c in step["controls"] if is_field(c) and c["form"] >= 0]  # the typing fields that sit in a form
    form_fields = collections.Counter(c["form"] for c in fields)
    first_field = {}
    for c in fields:
        first_field.setdefault(c["form"], c["i"])
    in_form = lambda c, form: form >= 0 and c["form"] == form
    focus_form = focused["form"] if focused else -1
    empty_left = sum(not c["filled"] and in_form(c, focus_form) for c in fields if c["i"] != focus)
    # how much of what is on screen is covered by something on top, and by which layer
    known = [c for c in controls if c["hit"] >= 0]
    occluded_share = sum(c["hit"] == 0 for c in known) / len(known) if known else 0
    layer_size = collections.Counter(c["layer"] for c in controls if c["layer"] >= 0)
    layer_known = collections.Counter(c["layer"] for c in known if c["layer"] >= 0)
    layer_hits = collections.Counter(c["layer"] for c in known if c["hit"] == 1 and c["layer"] >= 0)
    region_size = collections.Counter(c["region"] for c in controls if c["region"] >= 0)
    same = (
        lambda c, key: bool(focused) and c[key] >= 0 and c[key] == focused[key]
    )  # in the focused control's layer or region
    title_words = set(words(f"{page['title']} {page['h1']}"))
    query_words = set(words(page["query"]))
    for c in controls:
        label = set(words(c["label"]))
        segments = [p for p in href_path(c).split("/") if p]
        in_focus_form = in_form(c, focus_form)
        yield (
            c,
            {
                # where it is on the page and on screen
                "x": c["x"] / 1280,
                "y": c["y"] / 900,
                "w": c["w"] / 1280,
                "h": c["h"] / 900,
                "index": c["i"],
                "index_frac": c["i"] / max(1, c["n"]),
                "count": c["n"],
                "screen_rank": shown_rank.get(c["i"], -1),
                "on_screen_count": len(shown),
                "controls_above": bisect.bisect_left(shown_y, c["y"]),
                "page_scroll": page["scroll"],
                # what it is
                "kind": KINDS.get(c["kind"], -1),
                "box": BOXES.get(c["box"], -1),
                "same_kind_rank": kind_seen[c["kind"]],
                "same_kind_count": kind_count[c["kind"]],
                "inputs_on_screen": inputs_shown,
                "label_len": len(c["label"]),
                "label_words": len(words(c["label"])),
                "no_parent_text": c["parent"] == "",
                "before_len": len(c["before"]),
                # where it sits relative to the control you are in
                "nothing_focused": focus < 0,
                "index_delta": c["i"] - focus if focused else 0,
                "focus_dx": (c["x"] - focused["x"]) / 1280 if focused else 0,
                "focus_dy": (c["y"] - focused["y"]) / 900 if focused else 0,
                "focus_dist": math.hypot(c["x"] - focused["x"], c["y"] - focused["y"]) / 1000 if focused else 0,
                "focus_filled": bool(focused and focused["filled"]),
                "focus_in_layer": bool(focused) and focused["layer"] >= 0,
                "same_layer": same(c, "layer"),
                "same_region": same(c, "region"),
                # its form: the submit button, the first field, how much is still empty
                "required": c["required"],
                "is_submit": c["submit"],
                "submit_like": submit_like(c),
                "dismissive": dismissive(c),
                "first_field": is_field(c) and c["form"] >= 0 and first_field[c["form"]] == c["i"],
                "form_fields": form_fields[c["form"]] if c["form"] >= 0 else 0,
                "in_focus_form": in_focus_form,
                "after_focus": in_focus_form and c["i"] > focus,
                "empty_fields_left": empty_left,
                "empty_field": is_field(c) and not c["filled"],
                # how it looks
                "fill": c["fill"],
                "sat": c["sat"],
                "lum": c["lum"],
                "round": c["round"],
                "bold": c["bold"],
                "pointer": c["pointer"],
                "icon": c["icon"],
                "sat_rank": sat_rank.get(c["i"], -1),
                # the stack of layers above the page: on top or covered, its own layer's size and place, what the site is presenting
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
                "foreground": occluded_share if c["layer"] >= 0 and c["hit"] == 1 else 0,
                "region_size": region_size[c["region"]] if c["region"] >= 0 else 0,
                "in_dialog": c["dialog"],
                "bar": c["bar"],
                "fixed": c["fixed"],
                # content or frame: a headline (heading level, which heading, type size next to the running text), in an article
                # or main, in the site's frame, which of the content's controls it is
                "heading": c["head"],
                "heading_rank": heading_rank.get(c["i"], -1),
                "font": c["font"] / page["base"],
                "font_rank": font_rank.get(c["i"], -1),
                "in_article": c["art"],
                "in_main": c["main"],
                "in_frame": c["chrome"],
                "content_rank": content_rank.get(c["i"], -1),
                # one of many alike (cards, results, rows; which one), with a picture, in running text, said or linked again
                "alike": c["like"],
                "alike_index": c["at"],
                "image": c["img"],
                "prose": c["prose"],
                "label_repeats": label_count[c["label"].strip().lower()] - 1 if c["label"].strip() else 0,
                "href_repeats": href_count[href_path(c)] - 1 if href_path(c) else 0,
                # where it leads: another site, a spot on this page, how deep, a slug of words or a number (an article, a product)
                "external": c["ext"],
                "fragment": c["href"].startswith("#"),
                "href_depth": len(segments),
                "slug_words": len([w for w in re.split(r"[-_]+", segments[-1]) if w]) if segments else 0,
                "href_number": matches(r"[0-9]{3,}", href_path(c)),
                # what it says next to what the page is about
                "title_overlap": len(label & title_words) / len(label) if label else 0,
                "query_overlap": len(label & query_words) / len(label) if label else 0,
                "page_prose": page["prose"],
            },
        )
        kind_seen[c["kind"]] += 1


FEATURES = list(
    next(f for s in train for _, f in describe(s))
)  # ranker.json lists them; after them, a 0/1 column per word


def rows(steps_):
    """The rows the models see, every step's controls one after another: the features, then a 0/1 per word of the control, of the
    text around it and of the page, as one sparse matrix; and each step's control ids."""
    dense, cols, ids = [], [], []
    for s in steps_:
        page = [f"p:{w}" for w in page_words(s["page"])]
        ids.append([])
        for c, f in describe(s):
            ids[-1].append(c["i"])
            dense.append(list(f.values()))
            mine = page + [f"w:{w}" for w in control_words(c)] + [f"c:{w}" for w in context_words(c)]
            cols.append(sorted({COLUMN[w] for w in mine if w in COLUMN}))
    indptr = np.cumsum([0] + [len(c) for c in cols])
    bags = sp.csr_matrix(
        (np.ones(indptr[-1]), np.array([j for c in cols for j in c], dtype=np.int64), indptr),
        shape=(len(cols), len(WORDS)),
    )
    return sp.hstack([sp.csr_matrix(np.array(dense, dtype=float).reshape(-1, len(FEATURES))), bags], format="csr"), ids


def relevance(s, ids):
    """3 the control wanted next, 2 the picks still ahead of it, 1 the also-fine ones, 0 the rest (and the picks already visited)."""
    p = s["picks"]
    ahead = set(p[p.index(s["target"]) + 1 :]) if s["target"] in p else set()
    return [3 if i == s["target"] else 2 if i in ahead else 1 if i in s["ok"] else 0 for i in ids]


X, ids = rows(train)
y = np.concatenate([relevance(s, i) for s, i in zip(train, ids)])
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
# The ranker learns the order; the other model how much each control is wanted (the next one 1, the picks after it 2/3, the also-fine
# 1/3), so that the two agree on the order too.
models = [
    lgb.LGBMRanker(**params).fit(X, y, group=[len(i) for i in ids]).booster_,
    lgb.LGBMRegressor(objective="cross_entropy", **params).fit(X, y / 3).booster_,
]

z = lambda v: (v - v.mean()) / (v.std() + 1e-9)


def report(name, steps_):
    """On landing (nothing focused): is the first stop wanted, and how many of the 4 stops are. Over all steps: the target first,
    the target within the 4. For all steps and per kind of page, each model's scores standardised within the page and added."""
    X, ids = rows(steps_)
    scores = [m.predict(X, raw_score=True) for m in models]
    tally = collections.defaultdict(lambda: np.zeros(6))
    start = 0
    for s, i in zip(steps_, ids):
        end = start + len(i)
        order = [i[j] for j in np.argsort(-(z(scores[0][start:end]) + z(scores[1][start:end])), kind="stable")]
        start = end
        if not order:
            continue
        row = [0, 0, 0, order[0] == s["target"], s["target"] in order[:4], 1]
        if s["focus"] < 0:
            wanted = {*s["picks"], *s["ok"]}
            row[:3] = [order[0] in wanted, len(wanted & set(order[:4])) / 4, 1]
        for kind in {"all", s["type"]}:
            tally[kind] += row
    print(f"{name}:")
    for kind, t in sorted(tally.items(), key=lambda kv: (kv[0] != "all", kv[0])):
        print(
            f"  {kind:>9}: first stop wanted {t[0] / max(t[2], 1):.3f}  stops wanted {t[1] / max(t[2], 1):.3f}",
            f" top-1 {t[3] / t[5]:.3f}  top-4 {t[4] / t[5]:.3f}  ({t[5]:.0f} steps)",
        )


if held:
    report(f"{len(held)} held-out steps{' (also trained on)' if args.all else ''}", held)


def flat(root):
    """ranker.json: a tree as flat arrays [feature, threshold, left, right, leaf values]; a negative child ~n is leaf n"""
    f, thr, left, right, leaf = [], [], [], [], []

    def visit(node):
        if "leaf_index" in node:
            leaf.append(round(node["leaf_value"], 5))
            return ~(len(leaf) - 1)
        i = len(f)
        f.append(node["split_feature"])
        thr.append(node["threshold"])
        left.append(0)
        right.append(0)
        left[i] = visit(node["left_child"])
        right[i] = visit(node["right_child"])
        return i

    visit(root)
    return [f, thr, left, right, leaf]


trees = [[flat(info["tree_structure"]) for info in m.dump_model()["tree_info"]] for m in models]
# only the words some tree asks about go in ranker.json, numbered after the features
used = sorted({f for model in trees for t in model for f in t[0] if f >= len(FEATURES)})
column = {f: len(FEATURES) + j for j, f in enumerate(used)}
for t in (t for model in trees for t in model):
    t[0] = [column.get(f, f) for f in t[0]]
ranker = {
    "features": FEATURES,
    "kinds": KINDS,
    "boxes": BOXES,
    "words": [WORDS[f - len(FEATURES)] for f in used],
    "models": trees,
}
pathlib.Path(args.out).write_text(json.dumps(ranker, separators=(",", ":")))
print(f"2 models of {len(trees[0])} trees and {len(used)} words written to {args.out}")
