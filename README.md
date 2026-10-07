# jab

Jev-style tab, not affiliated with Jev.

As in:

-----------

> **Mom can we have `tab`?**

> We have `tab` at home.

-----------

Press Ctrl+Q: jab focuses the control the page most likely wants next, and marks the top four with small 1-4 badges; press it again to walk to the next one. Load this folder as an unpacked extension (chrome://extensions, Developer mode, Load unpacked).

## How it works

`jab.js` lists every focusable control on the page (no cap; only a page with more than 5,000 gets its most relevant 5,000) with plain facts about each: kind, label, where it is and how big, whether it is on screen, how it looks (filled, colourful, round, bold, icon only), its form (is it a submit button, is it filled in, is it the first field), where it sits relative to the control you are in, and how it sits in the page's layers (is it in a layer of its own above the page, how much of the view that layer covers, is the control on top where it is or covered, is the page behind it covered, is the page scrollable at all), plus the words in its label, its link and the page's address and title. The extension's service worker scores them with two LightGBM models whose standardised scores are averaged (`ranker.json`, 1 MB). A press takes about 20-30 ms on a normal page and 0.8 s on a page of 4,000 controls, on the CPU; nothing is cached and nothing leaves the browser.

Two rules were taught by case studies, as general rules over that metadata and not as sites: after you type, the control that finishes what you typed comes next; and a part of the page that the site foregrounds (in its own layer, on top, with the page behind it covered) and a bar the page keeps in reach (never scrolls away) earn extra weight, without the page's main search field losing its place.

## Accuracy

On held-out pages the model never saw in training (586 labelled steps from about 340 pages, of every kind: shopping, news, blogs, finance, government, web apps, docs, education, travel, health, media, with every control of the page in play), the first pick is the labelled one 46% of the time (top-1), one of the top three 71% of the time and one of the top four 76%; the scoring formula alone gets 18%, 34% and 41%. The labels are picks of what a visitor wants first on a page: 909 pages by me and 500 by Haiku looking at screenshots, so "right" means "what that labeller would pick". Two independent labelling passes over the same 24 pages agree on the first pick only 54% of the time (79% within each other's top three), so the labels, more than the model, limit how high a top-1 can be measured.

On generated pages that exhibit the case-study rules (50 per case study, in structures the model never trained on), real Chrome with the real extension puts the control the rule favours among the first four stops 95% of the time, and the sticky-header and cookie-notice counter-examples stay at 100%.

## Retrain, and case studies on top

`uv run train.py data.jsonl.gz ranker.json` (the data is not included; its format is at the top of `train.py`). Add `--all` to also train on the held-out pages for the file you ship. It takes about a minute.

When jab gets a page wrong, find the general rule behind it, give the model the metadata to see it, generate about 50 varied pages that exhibit it (plus counter-examples) and turn them into case steps: the same shape as the data (the page, the focus, the control you wanted; `accept` lists other acceptable ones). `uv run train.py data.jsonl.gz ranker.json --cases my-case.jsonl` adds it to a ledger kept next to the output (`ranker.json.state/`) and applies the whole ledger on top of the base models, which are trained once and kept there. The step is a policy-gradient update: a softmax over the page's controls is the policy and each case rewards the acceptable control, mixed with replayed ordinary steps so nothing else is forgotten. Running it again with the same cases gives the same result, and adding a case never means starting over. `--cases-eval FILE` scores cases the update never saw, to check that a fix carries over. `--rollback` empties the ledger and exports the base models as they are; changing how the case step works means editing it and running again, which redoes only that step.
