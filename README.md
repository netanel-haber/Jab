# jab

Ctrl+Q tabs to what you want next, using 2MB of decision trees from 2016.

> **Mom can we have `tab`?**

> We have `tab` at home.

| Keys | |
|---|---|
| Ctrl+Q | Go to the best guess, badge the top 4 |
| Tap again | Next of the 4 |
| Hold Ctrl+Q + 1-4 | That one |

Anything else you do ends it. Runs locally in ~20 ms; nothing leaves the browser.

Install: chrome://extensions, Developer mode, Load unpacked.

Retrain: `uv run train.py data.jsonl.gz ranker.json` (data not included; format and case-study options in `train.py`).

Built on [LightGBM](https://github.com/microsoft/LightGBM) (Microsoft, 2016; Ke et al., NeurIPS 2017), [NumPy](https://numpy.org) and [scikit-learn](https://scikit-learn.org). Training pages labelled by Claude.

© 2026 Netanel Haber, MIT License.
