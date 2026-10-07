# jab

Jev-style tab, not affiliated with Jev.

As in:

-----------

> **Mom can we have `tab`?**

> We have `tab` at home.

-----------

Press Ctrl+Q: jab focuses the control the page most likely wants next, and marks the top three with small 1/2/3 badges; press it again to walk to the next one. Load this folder as an unpacked extension (chrome://extensions, Developer mode, Load unpacked).

## How it works

`jab.js` lists the page's focusable controls (up to 80) with plain facts about each: kind, label, where it is and how big, whether it is on screen, its flags, the words in its label and link. in the extension's service worker it scores them, running the LightGBM trees in `ranker.json` (400 trees, 0.5 MB). It takes about 0.1 s per press, runs on the CPU, caches nothing, and sends nothing anywhere.

## Accuracy

Measured on 123 held-out pages the model never saw in training, with the extension loaded in real Chrome and a single Ctrl+Q press on each live page: it focuses the control I labelled first **50% of the time** (top-1), and one of my labelled picks 78% of the time. The scoring formula alone gets 18%. The labels are my own picks of what a visitor wants first on a page, so "right" means "what I would pick".

## Retrain

`uv run train.py data.jsonl.gz ranker.json` (the data is not included; its format is at the top of `train.py`). Add `--all` to also train on the held-out pages for the file you ship.
