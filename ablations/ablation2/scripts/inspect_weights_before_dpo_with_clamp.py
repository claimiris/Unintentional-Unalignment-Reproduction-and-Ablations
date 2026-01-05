import argparse, torch, math, sys
from transformers import AutoTokenizer, AutoModelForCausalLM
import pandas as pd
import torch.nn.functional as F
from datasets import load_dataset
import numpy as np
from scipy.stats import spearmanr

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--pref_path", type=str, required=True)
parser.add_argument("--data", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--max_examples", type=int, default=500)
parser.add_argument("--kl_coeff", type=float, default=1.0)
args = parser.parse_args()

device = torch.device(args.device if torch.cuda.is_available() else "cpu")

print("Loading model:", args.model)
tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.float16 if "cuda" in args.device else None)
model.to(device)
model.eval()

# Helper to ensure we get query/chosen/rejected
def format_example_like_repo(example):
    query = example.get("query") or example.get("prompt")
    # Robustly find chosen/rejected
    if "chosen" in example and "rejected" in example:
        chosen = example["chosen"]
        rejected = example["rejected"]
    elif "text_w" in example and "text_l" in example:
        chosen = example["text_w"]
        rejected = example["text_l"]
    else:
        chosen = example.get("output_1", "")
        rejected = example.get("output_2", "")
        
    return query, chosen, rejected

print("Loading preference similarity file:", args.pref_path)
pref = torch.load(args.pref_path, map_location="cpu")

if "ln_ches_scores" in pref: ln_vals = pref["ln_ches_scores"]
elif "ln_ches" in pref: ln_vals = pref["ln_ches"]
elif "ches_scores" in pref: ln_vals = pref["ches_scores"]
else: raise RuntimeError("No ln_ches_scores/ln_ches/ches_scores found in pref file")

sample_indices = pref.get("sample_indices", None)

print("Loading dataset (train jsonl):", args.data)
ds = load_dataset("json", data_files=args.data, split="train")
N = len(ds)
print("Dataset length:", N)

ln_ches_full = np.zeros(N, dtype=float)
if sample_indices is not None:
    for local_idx, global_idx in enumerate(sample_indices):
        if 0 <= global_idx < N:
            ln_ches_full[global_idx] = float(ln_vals[local_idx])
else:
    if len(ln_vals) == N:
        ln_ches_full = np.array(ln_vals, dtype=float)
    else:
        
        print(f"WARNING: Score length ({len(ln_vals)}) != Dataset length ({N}). Truncating.")
        ln_ches_full[:min(len(ln_vals), N)] = np.array(ln_vals[:min(len(ln_vals), N)], dtype=float)

# helper: compute logprob of response tokens conditioned on prompt
@torch.no_grad()
def logprob_of_response(prompt_text, response_text):
    full = prompt_text + response_text
    tok = tokenizer(full, return_tensors="pt", truncation=False).to(device)
    
    with torch.no_grad():
        out = model(**tok, output_hidden_states=False)
        logits = out.logits
    
    logprobs = torch.log_softmax(logits, dim=-1)
    input_ids = tok["input_ids"][0]
    
    prompt_tok = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(device)
    prompt_len = prompt_tok["input_ids"].shape[1]
    
    resp_ids = input_ids[prompt_len:]
    if resp_ids.numel() == 0: return float("-inf"), 0
    
    lp = 0.0
    lps = logprobs[0, prompt_len:, :].float()
    for i, tok_id in enumerate(resp_ids):
        lp += float(lps[i, int(tok_id)])
    return float(lp), int(resp_ids.numel())

# iterate examples
limit = min(args.max_examples, N) if args.max_examples > 0 else N
print(f"Using first {limit} examples for diagnostics")

results = []
clamped_low = 0
clamped_high = 0

for i in range(limit):
    ex = ds[i]
    query, chosen, rejected = format_example_like_repo(ex)
    
    if not query and "instruction" in ex:
        query = f"<|prompter|>{ex['instruction']}\n{ex.get('input','')}<|assistant|>"
    

    if chosen is None: chosen = ""
    if rejected is None: rejected = ""
    
    text_w = chosen if query and chosen.startswith(query) else (query or "") + chosen
    text_l = rejected if query and rejected.startswith(query) else (query or "") + rejected

    lp_w, lw = logprob_of_response(query, text_w[len(query):] if query else text_w)
    lp_l, ll = logprob_of_response(query, text_l[len(query):] if query else text_l)
    
    raw_pi_ratio = lp_w - lp_l

    # clamp
    clamped_ratio = raw_pi_ratio
    if raw_pi_ratio > 500.0:
        clamped_ratio = 500.0
        clamped_high += 1
    elif raw_pi_ratio < -500.0:
        clamped_ratio = -500.0
        clamped_low += 1

    loss = -F.logsigmoid(torch.tensor(args.kl_coeff * clamped_ratio)).item()

    results.append({
        "index": i,
        "ln_ches": float(ln_ches_full[i]),
        "pi_logratio": raw_pi_ratio,       # Raw
        "dpo_surrogate_loss": loss,        # Clamped
        "len_w": lw,
        "len_l": ll
    })

    if (i+1) % 50 == 0: print(f"processed {i+1}/{limit}")

print("\nDone.")
print(f"Clamping Report: {clamped_low} low (< -10), {clamped_high} high (> 10)")

if not results:
    print("Error: No results found. Check your dataset and limit.")
    sys.exit(1)

df = pd.DataFrame(results)

print("\n---- SUMMARIES (Pandas) ----")
print(df[["ln_ches", "pi_logratio", "dpo_surrogate_loss"]].describe(percentiles=[0.1, 0.5, 0.9, 0.95]))

clean_df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["ln_ches", "pi_logratio", "dpo_surrogate_loss"])

if len(clean_df) > 1:
    corr_pi = spearmanr(clean_df['ln_ches'], clean_df['pi_logratio'])[0]
    corr_loss = spearmanr(clean_df['ln_ches'], clean_df['dpo_surrogate_loss'])[0]
    print(f"\nSpearman(ln_ches, pi_logratio): {corr_pi:.4f}")
    print(f"Spearman(ln_ches, loss):        {corr_loss:.4f}")
else:
    print("\nNot enough valid data for correlations.")

# alpha check
p95_ln = df["ln_ches"].quantile(0.95)
print(f"\nln_ches p95: {p95_ln:.4f}")


w_targets = [0.9, 0.7, 0.5, 0.3, 0.1]
print("w_target -> alpha -> mean_weight / effective_mass / frac_near_zero")
for wt in w_targets:
    alpha_suggested = (1.0 - wt) / (p95_ln + 1e-12)
    weights = np.clip(1.0 - alpha_suggested * df["ln_ches"], 0.0, 1.0)
    mean_w = float(weights.mean())
    eff_mass = float(weights.sum())
    frac_zero = float((weights <= 0.01).mean())
    print(f"w{wt:.2f}: alpha={alpha_suggested:.4f} mean_w={mean_w:.4f} eff_mass={eff_mass:.1f} frac_zero={frac_zero:.3f}")
    
alphas = [0.1, 0.3, 0.5, 1.0, 0.01]
print("\nalpha grid diagnostics:")
for a in alphas:
    weights = np.clip(1.0 - a * df["ln_ches"], 0.0, 1.0)
    print(f"alpha {a}: mean_w={weights.mean():.4f}, eff_mass={weights.sum():.1f}, frac_zero={(weights<=0.01).mean():.3f}")

out_csv = "diagnostics_pre_dpo_clamped.csv"
df.to_csv(out_csv, index=False)
print(f"\nSaved diagnostics to {out_csv}")