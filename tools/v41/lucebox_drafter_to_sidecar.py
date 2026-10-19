#!/usr/bin/env python3
"""lucebox_drafter_to_sidecar.py - a Lucebox DSpark drafter GGUF (deepseek41-dflash-draft: blk.N.* + dflash.*) as
the DSpark sidecar this engine loads (STRATA_SERVE_DSPARK: the checkpoint's mtp.N.* names, v41_sidecar.py's layout).
The tensors are copied byte for byte in their quantized types (the draft is verified by the trunk, so its
precision only moves the acceptance rate); only the names change, the inverse of Lucebox's
convert_ds4_dspark_draft_to_gguf.py table.

    python3 tools/v41/lucebox_drafter_to_sidecar.py IN.gguf OUT.gguf   (gguf-py from a llama.cpp checkout on sys.path)
"""
import os
import sys

import numpy as np

for p in (os.environ.get("GGUF_PY"), "/home/yuya/llm-src/lucebox/server/deps/llama.cpp/gguf-py",
          os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "third_party", "llama.cpp", "gguf-py")):
    if p and os.path.isdir(p):
        sys.path.insert(0, p)
        break
import gguf  # noqa: E402

BLOCK = {
    "attn_norm.weight": "attn_norm.weight",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_output_a.weight": "attn.wo_a.weight",
    "attn_output_b.weight": "attn.wo_b.weight",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_attn_base.weight": "hc_attn_base",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "hc_ffn_base.weight": "hc_ffn_base",
    "ffn_gate_exps.weight": "ffn.experts.w1.weight",
    "ffn_up_exps.weight": "ffn.experts.w3.weight",
    "ffn_down_exps.weight": "ffn.experts.w2.weight",
}


def main():
    src, dst = sys.argv[1], sys.argv[2]
    r = gguf.GGUFReader(src)
    arch = r.fields["general.architecture"].contents()
    if arch != "deepseek41-dflash-draft":
        sys.exit(f"{src}: architecture {arch!r} is not a Lucebox DSpark drafter")
    n_blk = int(r.fields["deepseek41-dflash-draft.block_count"].contents())
    last = n_blk - 1
    GLOBAL = {
        "dflash.fc.weight": "mtp.0.main_proj.weight",
        "dflash.hidden_norm.weight": "mtp.0.main_norm.weight",
        "output_norm.weight": f"mtp.{last}.norm.weight",
        "dflash.dspark.markov.w1": f"mtp.{last}.markov_head.embed.weight",
        "dflash.dspark.markov.w2": f"mtp.{last}.markov_head.head.weight",
        "dflash.dspark.confidence.weight": f"mtp.{last}.confidence_head.proj.weight",
        "dflash.dspark.confidence.bias": f"mtp.{last}.confidence_head.proj.bias",
    }
    w = gguf.GGUFWriter(dst, "deepseek41")
    w.add_name("DeepSeek-V4.1 DSpark draft (sidecar from a Lucebox drafter)")
    w.add_string("sidecar.source", os.path.basename(src))
    n = 0
    for t in r.tensors:
        name = t.name
        if name.startswith("blk."):
            b, rest = name[4:].split(".", 1)
            if rest not in BLOCK:
                sys.exit(f"{name}: no sidecar name for it")
            new = f"mtp.{int(b)}.{BLOCK[rest]}"
        elif name in GLOBAL:
            new = GLOBAL[name]
        else:
            sys.exit(f"{name}: no sidecar name for it")
        data = t.data
        if t.tensor_type in (gguf.GGMLQuantizationType.F32, gguf.GGMLQuantizationType.F16):
            data = np.ascontiguousarray(data)
            if new.endswith("confidence_head.proj.weight") and data.ndim == 1:
                data = data.reshape(1, -1)   # the checkpoint's [1, dim] Linear weight: ggml [dim, 1]
            w.add_tensor(new, data)
        else:
            w.add_tensor(new, np.ascontiguousarray(data), raw_shape=t.data.shape, raw_dtype=t.tensor_type)
        n += 1
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file(progress=False)
    w.close()
    print(f"{n} tensors -> {dst} ({os.path.getsize(dst) / 2**30:.2f} GiB)")


if __name__ == "__main__":
    main()
