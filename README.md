# jab

Jev-style tab, not affiliated with Jev.

As in:

-----------

> **Mom can we have `tab`?**

> We have `tab` at home.

-----------

Press Ctrl+Q: jab focuses the control the page most likely wants next, and marks the top three with small 1/2/3 badges; press it again to walk to the next one. Load this folder as an unpacked extension (chrome://extensions, Developer mode, Load unpacked).

## How it works

`jab.js` lists the page's focusable controls (up to 80) with plain facts about each: kind, label, where it is and how big, whether it is on screen, its form (is it a submit button, is it filled in, is it the first field), and the words in its label, its link and the page's address and title. The extension's service worker scores them with two LightGBM models whose standardised scores are averaged (`ranker.json`: a LambdaRank ranker and a pick-or-not classifier, 400 trees each, 1 MB). A press takes about 25 ms from keydown to focus, including waking the service worker, on the CPU; nothing is cached and nothing leaves the browser.

It knows what you have typed: focus a search box, type, press Ctrl+Q, and the next stop is the form's GO button (not CLEAR); in a two-field form it goes to the next empty field first.

## Accuracy

On held-out pages the model never saw in training (602 labelled steps from about 340 pages, of every kind: shopping, news, blogs, finance, government, web apps, docs, education, travel, health, media), the first pick is the labelled one 47% of the time (top-1) and one of the top three 71% of the time; the scoring formula alone gets 18% and 36%. The labels are picks of what a visitor wants first on a page: 909 pages by me and 500 by Haiku looking at screenshots, so "right" means "what that labeller would pick".

## Retrain

`uv run train.py data.jsonl.gz ranker.json` (the data is not included; its format is at the top of `train.py`). Add `--all` to also train on the held-out pages for the file you ship.
