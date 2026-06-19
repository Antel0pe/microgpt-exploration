"""
Run inference from a saved microgpt.py checkpoint.

Pass a checkpoint on the CLI, or set CHECKPOINT_FILENAME below.
Checkpoint names are resolved under saved_runs/.
"""

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone


SAVED_RUNS_DIR = "saved_runs"
OUTPUT_NAMES_DIR = "output_names"
CHECKPOINT_FILENAME = "first_run"
NUM_SAMPLES = 20
TEMPERATURE = 0.5


def resolve_checkpoint_path(checkpoint_filename):
    if not checkpoint_filename:
        raise SystemExit("Set CHECKPOINT_FILENAME or pass a checkpoint on the CLI.")

    checkpoint_path = os.path.join(SAVED_RUNS_DIR, checkpoint_filename)
    if os.path.isdir(checkpoint_path):
        checkpoint_path = os.path.join(checkpoint_path, "model.json")
    return checkpoint_path


def load_checkpoint(checkpoint_path):
    with open(checkpoint_path) as f:
        payload = json.load(f)
    return payload["config"], payload["uchars"], payload["state_dict"]


def linear(x, w):
    return [sum(wi * xi for wi, xi in zip(wo, x)) for wo in w]


def softmax(logits):
    max_val = max(logits)
    exps = [math.exp(val - max_val) for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x):
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def make_gpt(config, state_dict):
    n_layer = config["n_layer"]
    n_head = config["n_head"]
    head_dim = config["head_dim"]

    def gpt(token_id, pos_id, keys, values):
        tok_emb = state_dict["wte"][token_id]
        pos_emb = state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(tok_emb, pos_emb)]
        x = rmsnorm(x)

        for li in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"layer{li}.attn_wq"])
            k = linear(x, state_dict[f"layer{li}.attn_wk"])
            v = linear(x, state_dict[f"layer{li}.attn_wv"])
            keys[li].append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs:hs + head_dim]
                k_h = [ki[hs:hs + head_dim] for ki in keys[li]]
                v_h = [vi[hs:hs + head_dim] for vi in values[li]]
                attn_logits = [
                    sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5
                    for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))
                    for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]

            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
            x = [max(0, xi) for xi in x]
            x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        return linear(x, state_dict["lm_head"])

    return gpt


def write_output(checkpoint_path, samples):
    run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    os.makedirs(OUTPUT_NAMES_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_NAMES_DIR, f"{run_name}.txt")
    lines = [
        f"checkpoint: {checkpoint_path}",
        f"temperature: {TEMPERATURE}",
        "",
        *samples,
    ]
    with open(output_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    return output_path


def run_inference(config, uchars, state_dict, checkpoint_path):
    n_layer = config["n_layer"]
    block_size = config["block_size"]
    vocab_size = config["vocab_size"]
    bos = config["BOS"]
    gpt = make_gpt(config, state_dict)

    samples = []
    print("--- inference (new, hallucinated names) ---")
    for sample_idx in range(NUM_SAMPLES):
        keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
        token_id = bos
        sample = []
        for pos_id in range(block_size):
            logits = gpt(token_id, pos_id, keys, values)
            probs = softmax([logit / TEMPERATURE for logit in logits])
            token_id = random.choices(range(vocab_size), weights=probs)[0]
            if token_id == bos:
                break
            sample.append(uchars[token_id])
        sample_line = f"sample {sample_idx + 1:2d}: {''.join(sample)}"
        samples.append(sample_line)
        print(sample_line)
    output_path = write_output(checkpoint_path, samples)
    print(f"output: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "checkpoint",
        nargs="?",
        help="Run folder or checkpoint file under saved_runs/. Overrides CHECKPOINT_FILENAME.",
    )
    parser.add_argument(
        "--checkpoint",
        dest="checkpoint_option",
        help="Run folder or checkpoint file under saved_runs/. Overrides CHECKPOINT_FILENAME.",
    )
    args = parser.parse_args()

    random.seed(42)
    checkpoint_filename = args.checkpoint_option or args.checkpoint or CHECKPOINT_FILENAME
    checkpoint_path = resolve_checkpoint_path(checkpoint_filename)
    config, uchars, state_dict = load_checkpoint(checkpoint_path)
    print(f"checkpoint: {checkpoint_path}")
    run_inference(config, uchars, state_dict, checkpoint_path)


if __name__ == "__main__":
    main()