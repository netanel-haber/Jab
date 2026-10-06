# Jab

Jev-style tab, not affiliated with Jev.

As in:

-----------

> **Mom can we have `tab`?**

> We have `tab` at home.

-----------

Hold Q and press Tab: a small int2 model in your browser focuses the control the page most likely wants next. Load this folder as an unpacked extension.

To retrain: `uv run train.py Qwen/Qwen3-0.6B-Base out data.jsonl.gz` writes `out/model.onnx` (the data is not included; its format is in `train.py`).
