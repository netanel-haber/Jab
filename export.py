# /// script
# requires-python = ">=3.12,<3.14"
# dependencies = ["torch>=2.6,<2.9", "transformers>=5.17,<6", "onnx", "onnxruntime>=1.23"]
# ///
"""Export a train.py run to one int2 ONNX file for the browser:   uv run export.py OUT_DIR model.onnx
Inputs: Qwen token ids (the 16k vocabulary is inside; any other token reads as pad), the index of <decide>, the index of each option's </opt>. Output: one score per option.
Matmuls are int2, the embedding int4, the small head stays full precision (it is a Gemm, which the quantizer leaves alone)."""
import json, sys, torch
from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
from transformers import AutoModel, AutoTokenizer

run, target = sys.argv[1:3]
keep = json.load(open(f"{run}/final/metrics.json"))["vocab"]
tok, model = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B-Base"), AutoModel.from_pretrained(f"{run}/final", dtype=torch.float32).eval()
model.register_buffer("remap", torch.full((len(tok),), keep.index(tok.pad_token_id)))   # Qwen token id -> 16k id
model.remap[keep] = torch.arange(len(keep))
model.head = torch.nn.Linear(model.config.hidden_size, 512)
model.head.load_state_dict(torch.load(f"{run}/final/head.pt"))
backbone = model.forward

def scores(ids, decide, opts):
    h = model.head(backbone(input_ids=model.remap[ids], use_cache=False).last_hidden_state[0])
    return h[opts, 256:] @ h[decide, :256] / 16   # each option's key against <decide>'s query

model.forward = scores
torch.onnx.export(model, (torch.arange(8)[None], torch.tensor(7), torch.tensor([3, 5])), target, dynamo=False, opset_version=20, input_names=["ids", "decide", "opts"],
                  output_names=["scores"], dynamic_axes={"ids": {1: "L"}, "opts": {0: "K"}, "scores": {0: "K"}})
for bits, op in [(4, "Gather"), (2, "MatMul")]:   # the embedding can only go down to 4 bits
    quantizer = MatMulNBitsQuantizer(target, bits=bits, block_size=32, is_symmetric=True, accuracy_level=4, op_types_to_quantize=(op,), quant_axes=(("MatMul", 0), ("Gather", 1)))
    quantizer.process()
    quantizer.model.save_model_to_file(target, use_external_data_format=False)   # one file
