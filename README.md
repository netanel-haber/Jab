# jab

Jev-style tab, not affiliated with Jev.

As in:

-----------

> **Mom can we have `tab`?**

> We have `tab` at home.

-----------

Press Ctrl+Q: jab focuses the control the page most likely wants next, and marks the top four with small 1-4 badges; press it again to walk to the next one. Load this folder as an unpacked extension (chrome://extensions, Developer mode, Load unpacked).

## How it works

`jab.js` lists every focusable control on the page (no cap; only a page with more than 5,000 gets its most relevant 5,000) with plain facts about each: kind, label, where it is and how big, whether it is on screen, how it looks (filled, colourful, round, bold, icon only), its form (is it a submit button, is it filled in, is it the first field), where it sits relative to the control you are in, and the words in its label, its link and the page's address and title. The extension's service worker scores them with two LightGBM models whose standardised scores are averaged (`ranker.json`: a LambdaRank ranker and a pick-or-not classifier, 1 MB). A press takes about 20 ms on a normal page and 0.4 s on a page of 4,000 controls, on the CPU; nothing is cached and nothing leaves the browser.

It knows what you have typed: focus a search box or a chat composer, type, press Ctrl+Q, and the next stop is the form's GO or send button (not CLEAR); in a two-field form it goes to the next empty field first.

## Accuracy

On held-out pages the model never saw in training (579 labelled steps from about 340 pages, of every kind: shopping, news, blogs, finance, government, web apps, docs, education, travel, health, media, with every control of the page in play), the first pick is the labelled one 44% of the time (top-1), one of the top three 72% of the time and one of the top four 78%; the scoring formula alone gets 18%, 34% and 42%. The labels are picks of what a visitor wants first on a page: 909 pages by me and 500 by Haiku looking at screenshots, so "right" means "what that labeller would pick". Two independent labelling passes over the same 24 pages agree on the first pick only 54% of the time (79% within each other's top three), so the labels, more than the model, limit how high a top-1 can be measured.

## Retrain, and case studies on top

`uv run train.py data.jsonl.gz ranker.json` (the data is not included; its format is at the top of `train.py`). Add `--all` to also train on the held-out pages for the file you ship. It takes about a minute.

When jab gets a page wrong, turn it into a case: a step of the same shape (the page, the focus, the control you wanted; optionally other acceptable ones). `uv run train.py data.jsonl.gz ranker.json --cases my-case.jsonl` adds it to a ledger kept next to the output (`ranker.json.state/`) and applies the whole ledger on top of the base models, which are trained once and kept there. The step is a policy-gradient update: a softmax over the page's controls is the policy and each case rewards the acceptable control, mixed with replayed ordinary steps so nothing else is forgotten. Running it again with the same cases gives the same result, and adding a case never means starting over. `--cases-eval FILE` scores cases the update never saw, to check that a fix carries over.
