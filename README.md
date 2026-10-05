# tab-at-home 

As in:

-----------

> **Mom can we have `tab`?**

> We have `tab` at home.

-----------

Hold Q and press Tab: a small int2 model in your browser focuses the control the page most likely wants next.

`uv run train.py Qwen/Qwen3-0.6B-Base out data.jsonl.gz` trains it and writes `out/model.onnx` (the data is not included; its format is in `train.py`). Copy that to `model/` with Qwen's `tokenizer.json`, then load this folder as an unpacked extension.
