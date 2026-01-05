import argparse
import os
from datetime import datetime

import datasets
import torch
from tqdm import tqdm

import common.utils.logging as logging_utils
from utils.script_utils import load_tokenizer_and_model

PADDING_TOKEN = "<|padding|>"
PROMPT_TOKEN = "<|prompter|>"
ASSISTANT_TOKEN = "<|assistant|>"
EOS_TOKEN = "<|endoftext|>"

def get_per_layer_norm_modules(model):
    """
    Returns a list of (layer_idx, module) tuples corresponding
    to the output norm of each transformer block, in order.
    """
    norm_modules = []

    for name, module in model.named_modules():
        lname = name.lower()
        cls = module.__class__.__name__.lower()

        # heuristics that work for OLMo / LLaMA / Gemma
        if (
            ('layernorm' in cls or 'rmsnorm' in cls)
            and ('layers.' in lname or 'blocks.' in lname)
            and ('post' in lname or 'output' in lname or lname.endswith('norm'))
        ):
            # extract layer index
            parts = lname.replace('blocks.', 'layers.').split('layers.')
            if len(parts) > 1:
                try:
                    layer_idx = int(parts[1].split('.')[0])
                    norm_modules.append((layer_idx, module))
                except ValueError:
                    pass

    # sort by layer index
    norm_modules = sorted(norm_modules, key=lambda x: x[0])
    return norm_modules
    
class LayerNormCapture:
    def __init__(self):
        self.pre = {}
        self.post = {}

    def make_hook(self, layer_idx):
        def hook(module, inputs, output):
            self.pre[layer_idx] = inputs[0].detach().cpu()
            self.post[layer_idx] = output.detach().cpu()
        return hook

    def clear(self):
        """Clears the captured hidden states between forward passes."""
        self.pre = {}
        self.post = {}


# --- DATASET FORMATTING HELPERS ---

def __orig_alpacafarm_create_format_input_func():
    def format_input_func(example):
        new_example = {}
        instruction, input_text = example["instruction"], example["input"]
        if input_text:
            query = f"{PROMPT_TOKEN}{instruction}\n{input_text}{ASSISTANT_TOKEN}"
        else:
            query = f"{PROMPT_TOKEN}{instruction}{ASSISTANT_TOKEN}"

        if example["preference"] == 1:
            selected = example["output_1"]
            rejected = example["output_2"]
        else:
            selected = example["output_2"]
            rejected = example["output_1"]

        new_example["query"] = query
        new_example["text_w"] = f"{query}{selected}"
        new_example["text_l"] = f"{query}{rejected}"
        return new_example

    return format_input_func


def __ultrafeedback_create_format_input_func():
    def format_input_func(example):
        new_example = {}
        chosen, rejected = example["chosen"], example["rejected"]
        query = f"{PROMPT_TOKEN}{chosen[0]['content']}{ASSISTANT_TOKEN}"

        new_example["query"] = query
        new_example["text_w"] = f"{query}{chosen[1]['content']}"
        new_example["text_l"] = f"{query}{rejected[1]['content']}"
        return new_example

    return format_input_func


def __create_chat_template_format_input_func(tokenizer, query_field: str = "query", chosen_field: str = "chosen", rejected_field: str = "rejected"):
    def format_input_func(example):
        new_example = {}

        query = [{"role": "user", "content": example[query_field]}]
        query = tokenizer.apply_chat_template(query, tokenize=False, add_generation_prompt=True)
        new_example["query"] = query
        new_example["text_w"] = f"{query}" + example[chosen_field]
        new_example["text_l"] = f"{query}" + example[rejected_field]
        return new_example

    return format_input_func


DATASET_CREATE_FORMAT_INPUT_FUNC = {
    "tatsu-lab/alpaca_farm": __orig_alpacafarm_create_format_input_func,
    "HuggingFaceH4/ultrafeedback_binarized": __ultrafeedback_create_format_input_func
}


# --- DATASET LOADING & PROCESSING ---

def __get_dataset(dataset_name: str, cache_dir: str = None):
    if dataset_name == "tatsu-lab/alpaca_farm":
        data_url = "https://huggingface.co/datasets/tatsu-lab/alpaca_farm/resolve/main/alpaca_human_preference.json"
        return datasets.load_dataset( "json",data_files=data_url, split="train",cache_dir=cache_dir ) 
    elif dataset_name == "HuggingFaceH4/ultrafeedback_binarized":
        return datasets.load_dataset(dataset_name, split="train_prefs", cache_dir=cache_dir)
    else:
        # Loads dataset from JSON file for all other datasets
        return datasets.load_dataset("json", data_files=dataset_name, split="train")


def __subsample_dataset(dataset, num_train_samples: int = -1, train_samples_random_seed: int = -1):
    if num_train_samples < 0:
        return torch.arange(len(dataset)), dataset

    if train_samples_random_seed > 0:
        perm = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(train_samples_random_seed))
    else:
        perm = torch.randperm(len(dataset))

    num_samples = min(num_train_samples, len(dataset))
    sample_indices = perm[:num_samples]
    dataset = dataset.select(sample_indices)
    return sample_indices, dataset


def __prepare_and_tokenize_dataset(sample_indices, dataset_name, dataset, tokenizer, max_input_length: int,
                                   chat_query_field: str = "query", chat_chosen_field: str = "chosen", chat_rejected_field: str = "rejected"):
    if not tokenizer.chat_template:
        format_input_func = DATASET_CREATE_FORMAT_INPUT_FUNC[dataset_name]()
    else:
        format_input_func = __create_chat_template_format_input_func(tokenizer, query_field=chat_query_field,
                                                                     chosen_field=chat_chosen_field, rejected_field=chat_rejected_field)

    dataset = dataset.map(format_input_func, batched=False)
    dataset = dataset.select_columns(["query", "text_w", "text_l"])

    max_input_length = max_input_length if max_input_length > 0 else None

    def tokenize_examples(example: dict):
        query_input_ids = tokenizer(example["query"], padding=False, truncation=max_input_length is not None,
                                    max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        text_w_input_ids = tokenizer(example["text_w"], padding=False, truncation=max_input_length is not None,
                                     max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        text_l_input_ids = tokenizer(example["text_l"], padding=False, truncation=max_input_length is not None,
                                     max_length=max_input_length, return_tensors="pt", add_special_tokens=not tokenizer.chat_template).input_ids
        return {
            "query": query_input_ids,
            "text_w": text_w_input_ids,
            "text_l": text_l_input_ids
        }

    dataset = dataset.map(tokenize_examples, batched=False)
    dataset.set_format(type="torch")

    indices_to_include = []
    for i, example in enumerate(dataset):
        query_len = example["query"][0].shape[0]
        preferred_token_ids = example["text_w"][0][query_len:]
        dispreferred_token_ids = example["text_l"][0][query_len:]

        if query_len == 0 or preferred_token_ids.shape[0] == 0 or dispreferred_token_ids.shape[0] == 0:
            continue

        indices_to_include.append(i)

    dataset = dataset.select(indices_to_include)
    return sample_indices[indices_to_include], dataset


def __update_tokenizer_setting_and_chat_tokens(tokenizer):
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "right"

    if not tokenizer.eos_token:
        tokenizer.eos_token = EOS_TOKEN

    if not tokenizer.chat_template:
        if not tokenizer.pad_token:
            tokenizer.add_special_tokens({"pad_token": PADDING_TOKEN})
        tokenizer.add_special_tokens({"additional_special_tokens": [PROMPT_TOKEN, ASSISTANT_TOKEN]})
    else:
        if not tokenizer.pad_token:
            tokenizer.pad_token = tokenizer.unk_token if tokenizer.unk_token else tokenizer.eos_token


def __trim_padding(input_ids, tokenizer):
    return input_ids[torch.argmax((input_ids != tokenizer.vocab[tokenizer.eos_token]).to(torch.int)):]


# --- EDIT DISTANCE METRICS ---

# Taken from the torchaudio edit_distance function
def __normalized_edit_distance(seq1, seq2):
    len_sent2 = len(seq2)
    dold = list(range(len_sent2 + 1))
    dnew = [0 for _ in range(len_sent2 + 1)]

    for i in range(1, len(seq1) + 1):
        dnew[0] = i
        for j in range(1, len_sent2 + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)

        dnew, dold = dold, dnew

    return int(dold[-1]) / max(len(seq1), len(seq2))


def get_and_log_preferred_dispreferred_normalized_edit_distance(logger, dataset):
    per_example_normalized_edit_distance = []

    logger.info(f"Starting normalized edit distance computation")
    for example in tqdm(dataset):
        query_len = example["query"][0].shape[0]
        preferred_token_ids = example["text_w"][0][query_len:]
        dispreferred_token_ids = example["text_l"][0][query_len:]

        if preferred_token_ids.shape[0] == 0 or dispreferred_token_ids.shape[0] == 0:
            raise ValueError("Found example with preferred or dispreferred outputs have length zero after truncating at maximal allowed length for "
                             "prompt + output.")

        pref_dispref_normalized_edit_distance = __normalized_edit_distance(preferred_token_ids, dispreferred_token_ids)
        per_example_normalized_edit_distance.append(pref_dispref_normalized_edit_distance)

    logger.info("\n===========================================================================================================================\n"
                "Edit distance metrics\n"
                "===========================================================================================================================")

    per_example_normalized_edit_distance = torch.tensor(per_example_normalized_edit_distance).to(torch.float)
    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"Normalized edit distance of preferred and dispreferred outputs\n"
                f"Mean: {per_example_normalized_edit_distance.mean()} , Min: {per_example_normalized_edit_distance.min()} , "
                f"25th percentile: {torch.quantile(per_example_normalized_edit_distance, q=0.25)} , "
                f"Median: {per_example_normalized_edit_distance.median()} , "
                f"75th percentile: {torch.quantile(per_example_normalized_edit_distance, q=0.75)} , "
                f"Max: {per_example_normalized_edit_distance.max()}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")

    return per_example_normalized_edit_distance


# --- CORE ABLATION LOGIC (LAYER-WISE) ---

def log_metric_stats(logger, name, scores):
    """
    Logs the statistical block for a given metric tensor, matching the original file's format.
    """
    # Ensure scores are float for readable logging
    scores = scores.float()
    
    if scores.numel() == 0:
        logger.info(f"\n------------------------------------------------------------\n"
                    f"{name}: No data available (Empty Tensor)\n"
                    f"------------------------------------------------------------")
        return

    logger.info(f"\n------------------------------------------------------------------------------------------------------------------------------\n"
                f"{name}:\n"
                f"Mean: {scores.mean():.4f} , "
                f"Min: {scores.min():.4f} , "
                f"25th percentile: {torch.quantile(scores, q=0.25):.4f} , "
                f"Median: {scores.median():.4f} , "
                f"75th percentile: {torch.quantile(scores, q=0.75):.4f} , "
                f"Max: {scores.max():.4f}\n"
                f"------------------------------------------------------------------------------------------------------------------------------")


def extract_layer_data(model_outputs, query_len):
    """
    Extracts Summed Embedding and Last Token Embedding for EVERY layer.
    """
    summed_embeddings = []
    last_embeddings = []
    
    # Calculate response length (Total - (Query_Start_Index))
    # We use the same slicing logic as the original file: [query_len - 1:]
    first_layer = model_outputs.hidden_states[0]
    full_seq_len = first_layer.shape[1]
    response_len = float(full_seq_len - (query_len - 1))
    
    for hidden_state in model_outputs.hidden_states:
        # hidden_state shape: [1, seq_len, dim]
        
        # 1. Slice: Get only the response tokens
        response_hidden = hidden_state[0, query_len - 1:, :] 
        
        # 2. Sum (for CHES)
        summed = response_hidden.sum(dim=0)
        summed_embeddings.append(summed)
        
        # 3. Last Token (for Inner Product)
        last = response_hidden[-1]
        last_embeddings.append(last)
        
    return summed_embeddings, last_embeddings, response_len

def get_and_log_layer_wise_metrics(logger, dataset, model, device, tokenizer):
    """
    Computes layer-wise CHES, length-normalized CHES, and last-token inner product
    for POST-norm (existing hidden_states) and PRE-norm (inputs to each layer's
    output normalization module). Returns six tensors:
      (post_ches_tensor, post_ln_tensor, post_inner_tensor,
       pre_ches_tensor,  pre_ln_tensor,  pre_inner_tensor)

    Assumes helper functions `get_per_layer_norm_modules(model)` and class
    `LayerNormCapture` (with method make_hook(layer_idx)) are defined earlier
    in the file.
    """
    import torch

    # Data structure to hold lists of lists: [Layer_Index][Example_Index]
    layer_ches_storage_post = None
    layer_ln_ches_storage_post = None
    layer_inner_storage_post = None

    # Pre-norm storages (to be initialized after we know num_layers)
    layer_ches_storage_pre = None
    layer_ln_ches_storage_pre = None
    layer_inner_storage_pre = None

    # --- NEW: set up per-layer norm hooks (register once) ---
    norm_modules = get_per_layer_norm_modules(model)  # expected -> list of (layer_idx, module)
    norm_capture = LayerNormCapture()
    norm_handles = []
    if norm_modules:
        logger.info(f"Registering hooks for {len(norm_modules)} norm modules (layers: {[t[0] for t in norm_modules]})")
        for layer_idx, norm_module in norm_modules:
            handle = norm_module.register_forward_hook(norm_capture.make_hook(layer_idx))
            norm_handles.append(handle)
    else:
        logger.warning("No per-layer norm modules found; pre-norm capture will be unavailable.")

    logger.info(f"Starting Layer-wise computation for CHES, LN-CHES, and Inner Products (pre & post)...")
    model.to(device)
    model.eval()

    for example in tqdm(dataset):
        # --- 1. Prepare Inputs ---
        query_len = __trim_padding(example["query"][0], tokenizer).shape[0]
        preferred = __trim_padding(example["text_w"][0], tokenizer).to(device)
        dispreferred = __trim_padding(example["text_l"][0], tokenizer).to(device)

        if query_len == 0 or query_len >= preferred.shape[0] or query_len >= dispreferred.shape[0]:
            logger.warn("Skipping example due to length issues.")
            continue

        # --- 2. POST-NORM Forward Passes (as before) ---
        with torch.no_grad():
            out_w = model(input_ids=preferred.unsqueeze(0), output_hidden_states=True)
            out_l = model(input_ids=dispreferred.unsqueeze(0), output_hidden_states=True)

        # --- 3. Extract Data (post-norm) ---
        w_sums, w_lasts, len_w = extract_layer_data(out_w, query_len)
        l_sums, l_lasts, len_l = extract_layer_data(out_l, query_len)

        # Initialize storage if this is the first valid example
        if layer_ches_storage_post is None:
            num_layers = len(w_sums)
            # POST-norm (existing)
            layer_ches_storage_post = [[] for _ in range(num_layers)]
            layer_ln_ches_storage_post = [[] for _ in range(num_layers)]
            layer_inner_storage_post = [[] for _ in range(num_layers)]
            # PRE-norm (NEW)
            layer_ches_storage_pre = [[] for _ in range(num_layers)]
            layer_ln_ches_storage_pre = [[] for _ in range(num_layers)]
            layer_inner_storage_pre = [[] for _ in range(num_layers)]

        # --- 4. Compute POST-NORM Metrics Per Layer (unchanged) ---
        for i in range(num_layers):
            h_p_sum = w_sums[i].float()
            h_m_sum = l_sums[i].float()
            h_p_last = w_lasts[i].float()
            h_m_last = l_lasts[i].float()

            # A. CHES Score (post)
            ches_post = torch.dot(h_p_sum, h_m_sum) - (torch.norm(h_p_sum) ** 2)
            layer_ches_storage_post[i].append(ches_post.cpu())

            # B. Length-Normalized CHES (post)
            term1 = torch.dot(h_p_sum, h_m_sum) / (len_w * len_l)
            term2 = (torch.norm(h_p_sum) ** 2) / (len_w ** 2)
            ln_ches_post = term1 - term2
            layer_ln_ches_storage_post[i].append(ln_ches_post.cpu())

            # C. Last-token inner product (post)
            inner_post = torch.dot(h_p_last, h_m_last)
            layer_inner_storage_post[i].append(inner_post.cpu())

        # --------------------------------------------------------------------
        # --- 5. PRE-NORM capture: obtain pre-norm inputs for both pref & disp ---
        # Because norm_capture gets overwritten on each forward, capture separately:
        #   - run preferred forward, copy norm_capture.pre -> pref_pre_dict
        #   - run dispreferred forward, copy norm_capture.pre -> disp_pre_dict
        # Note: these additional forwards are lightweight (no grad) and needed to get
        # the *inputs* to each layer's norm module.
        # --------------------------------------------------------------------
        # capture preferred pre-norms
        with torch.no_grad():
            norm_capture.clear()
            _ = model(input_ids=preferred.unsqueeze(0), output_hidden_states=True)
            # copy mapping: layer_idx -> tensor [1, seq, d]
            pref_pre_dict = {k: v.clone() for k, v in norm_capture.pre.items()}

        # capture dispreferred pre-norms
        with torch.no_grad():
            norm_capture.clear()
            _ = model(input_ids=dispreferred.unsqueeze(0), output_hidden_states=True)
            disp_pre_dict = {k: v.clone() for k, v in norm_capture.pre.items()}

        # --- 6. Compute PRE-NORM metrics for each layer i ---
        for i in range(1,num_layers):
            # If capture missing (heuristic failure), append NaN to mark missing
            if (i-1 not in pref_pre_dict) or (i-1 not in disp_pre_dict):
                layer_ches_storage_pre[i].append(torch.tensor(float('nan')))
                layer_ln_ches_storage_pre[i].append(torch.tensor(float('nan')))
                layer_inner_storage_pre[i].append(torch.tensor(float('nan')))
                continue

            w_pre = pref_pre_dict[i-1]   # [1, seq, d] CPU
            l_pre = disp_pre_dict[i-1]   # [1, seq, d] CPU

            # slice response tokens exactly as extract_layer_data: [query_len - 1:]
            w_resp_pre = w_pre[0, query_len - 1:, :].float()  # [resp_len, d]
            l_resp_pre = l_pre[0, query_len - 1:, :].float()

            if w_resp_pre.shape[0] == 0 or l_resp_pre.shape[0] == 0:
                layer_ches_storage_pre[i].append(torch.tensor(float('nan')))
                layer_ln_ches_storage_pre[i].append(torch.tensor(float('nan')))
                layer_inner_storage_pre[i].append(torch.tensor(float('nan')))
                continue

            w_pre_sum = w_resp_pre.sum(dim=0)
            l_pre_sum = l_resp_pre.sum(dim=0)

            # A. CHES (pre)
            ches_pre = torch.dot(w_pre_sum, l_pre_sum) - (torch.norm(w_pre_sum) ** 2)
            layer_ches_storage_pre[i].append(ches_pre.cpu())

            # B. LN-CHES (pre)
            term1_pre = torch.dot(w_pre_sum, l_pre_sum) / (w_resp_pre.shape[0] * l_resp_pre.shape[0])
            term2_pre = (torch.norm(w_pre_sum) ** 2) / (w_resp_pre.shape[0] ** 2)
            ln_ches_pre = term1_pre - term2_pre
            layer_ln_ches_storage_pre[i].append(ln_ches_pre.cpu())

            # C. last-token inner product (pre)
            inner_pre = torch.dot(w_resp_pre[-1], l_resp_pre[-1])
            layer_inner_storage_pre[i].append(inner_pre.cpu())

    # --- 7. Cleanup: remove hooks ---
    for h in norm_handles:
        try:
            h.remove()
        except Exception:
            pass

    # --- 8. Logging and Tensor Stacking ---
    final_ches_cols_post = []
    final_ln_ches_cols_post = []
    final_inner_cols_post = []

    final_ches_cols_pre = []
    final_ln_ches_cols_pre = []
    final_inner_cols_pre = []

    num_layers = len(layer_ches_storage_post) if layer_ches_storage_post else 0

    logger.info("\n" + "="*50 + " LAYER-WISE STATISTICS " + "="*50)

    for i in range(num_layers):
        # Convert lists to tensors for this layer (post)
        c_tensor_post = torch.tensor(layer_ches_storage_post[i])
        ln_tensor_post = torch.tensor(layer_ln_ches_storage_post[i])
        in_tensor_post = torch.tensor(layer_inner_storage_post[i])

        final_ches_cols_post.append(c_tensor_post)
        final_ln_ches_cols_post.append(ln_tensor_post)
        final_inner_cols_post.append(in_tensor_post)

        # Convert lists to tensors for this layer (pre)
        c_tensor_pre = torch.tensor(layer_ches_storage_pre[i])
        ln_tensor_pre = torch.tensor(layer_ln_ches_storage_pre[i])
        in_tensor_pre = torch.tensor(layer_inner_storage_pre[i])

        final_ches_cols_pre.append(c_tensor_pre)
        final_ln_ches_cols_pre.append(ln_tensor_pre) if 'final_ln_ches_cols_pre' in locals() else None  # ensure variable exists
        final_inner_cols_pre.append(in_tensor_pre)

        # LOGGING: The Full Statistical Block for EVERY layer (post)
        logger.info(f"\n>>> STATISTICS FOR LAYER {i} (POST) <<<")
        log_metric_stats(logger, f"Layer {i} CHES Scores (post)", c_tensor_post)
        log_metric_stats(logger, f"Layer {i} Length-Normalized CHES (post)", ln_tensor_post)
        log_metric_stats(logger, f"Layer {i} Last Hidden Inner Products (post)", in_tensor_post)

        # LOGGING: The Full Statistical Block for EVERY layer (pre)
        logger.info(f"\n>>> STATISTICS FOR LAYER {i} (PRE) <<<")
        log_metric_stats(logger, f"Layer {i} CHES Scores (pre)", c_tensor_pre)
        log_metric_stats(logger, f"Layer {i} Length-Normalized CHES (pre)", ln_tensor_pre)
        log_metric_stats(logger, f"Layer {i} Last Hidden Inner Products (pre)", in_tensor_pre)

    logger.info("="*120 + "\n")
    
    final_ches_cols_post = [t for t in final_ches_cols_post if t.numel() > 0]
    final_ln_ches_cols_post = [t for t in final_ln_ches_cols_post if t.numel() > 0]
    final_inner_cols_post = [t for t in final_inner_cols_post if t.numel() > 0]

    final_ches_cols_pre = [t for t in final_ches_cols_pre if t.numel() > 0]
    final_ln_ches_cols_pre = [t for t in final_ln_ches_cols_pre if t.numel() > 0]
    final_inner_cols_pre = [t for t in final_inner_cols_pre if t.numel() > 0]

    # --- STACKING ---
    post_ches_tensor = torch.stack(final_ches_cols_post).T if final_ches_cols_post else torch.empty(0)
    post_ln_tensor = torch.stack(final_ln_ches_cols_post).T if final_ln_ches_cols_post else torch.empty(0)
    post_inner_tensor = torch.stack(final_inner_cols_post).T if final_inner_cols_post else torch.empty(0)

    pre_ches_tensor = torch.stack(final_ches_cols_pre).T if final_ches_cols_pre else torch.empty(0)
    pre_ln_tensor = torch.stack(final_ln_ches_cols_pre).T if final_ln_ches_cols_pre else torch.empty(0)
    pre_inner_tensor = torch.stack(final_inner_cols_pre).T if final_inner_cols_pre else torch.empty(0)
   
    for h in norm_handles:
        h.remove()

    return (post_ches_tensor,
            post_ln_tensor,
            post_inner_tensor,
            pre_ches_tensor,
            pre_ln_tensor,
            pre_inner_tensor)


# --- MAIN EXECUTION ---

@torch.no_grad()
def main(config: dict):
    model_name = config["model"]
    dataset_name = config["dataset"]
    num_train_samples = config["num_train_samples"]
    train_samples_random_seed = config["train_samples_random_seed"]
    max_input_length = config["max_input_length"]
    device = torch.device(f"cuda:{config['gpu_id']}" if torch.cuda.is_available() and config["gpu_id"] >= 0 else "cpu")

    dataset_display_name = config["custom_dataset_display_name"] if config["custom_dataset_display_name"] else dataset_name.split("/")[-1]
    subdir_name = model_name.split("/")[-1] + "_" + dataset_display_name
    logger = logging_utils.create_logger(file_logging=not config["dont_save_logs"],
                                         log_dir=os.path.join(config["output_dir"], subdir_name),
                                         log_file_name_prefix=f"log_samples_{num_train_samples}")
    logger.info(f"Config: {config}")

    try:
        start_time = datetime.utcnow()

        logger.info(f"======================================================================================================")
        logger.info(f"Model: '{model_name}', Dataset: '{dataset_name}'")
        logger.info(f"======================================================================================================\n")

        # 1. Load Model & Tokenizer
        tokenizer, model = load_tokenizer_and_model(model_name, cache_dir=config["cache_dir"], device=device)

        # [CRITICAL] Enable Hidden States
        model.config.output_hidden_states = True
        model.to(device)

        # Ensure correct padding for embedding extraction
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = 'right'

        __update_tokenizer_setting_and_chat_tokens(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

        # 2. Load & Prepare Dataset
        dataset = __get_dataset(dataset_name, cache_dir=config["cache_dir"])
        sample_indices, dataset = __subsample_dataset(dataset, num_train_samples, train_samples_random_seed)
        sample_indices, tokenized_dataset = __prepare_and_tokenize_dataset(sample_indices, dataset_name, dataset, tokenizer, max_input_length,
                                                                           chat_chosen_field=config["chat_chosen_field"],
                                                                           chat_rejected_field=config["chat_rejected_field"])
        logger.info(f"Filtered out samples with empty query or outputs\n"
                    f"Original number of samples: {len(dataset)}\n"
                    f"Number of samples after filtering: {len(sample_indices)}")

        # 3. Calculate Normalized Edit Distance (Standard)
        normalized_edit_distances = get_and_log_preferred_dispreferred_normalized_edit_distance(logger, tokenized_dataset)

        # 4. Calculate Layer-wise CHES & Metrics (Ablation)
        (
            post_ches_tensor,
            post_ln_ches_tensor,
            post_inner_tensor,
            pre_ches_tensor,
            pre_ln_ches_tensor,
            pre_inner_tensor
        ) = get_and_log_layer_wise_metrics(
            logger, tokenized_dataset, model, device, tokenizer
        )


        # 5. Save Results
        results = {
            "sample_indices": sample_indices,
            "minus_normalized_edit_distances": -normalized_edit_distances,

            # ======================
            # POST-NORM METRICS
            # ======================
            "post_ches_layer_trajectories": post_ches_tensor,
            "post_ln_ches_layer_trajectories": post_ln_ches_tensor,
            "post_inner_product_layer_trajectories": post_inner_tensor,

            "post_ches_scores": post_ches_tensor[:, -1],
            "post_ln_ches_scores": post_ln_ches_tensor[:, -1],
            "post_last_hidden_embedding_inner_prods": post_inner_tensor[:, -1],

            # ======================
            # PRE-NORM METRICS
            # ======================
            "pre_ches_layer_trajectories": pre_ches_tensor,
            "pre_ln_ches_layer_trajectories": pre_ln_ches_tensor,
            "pre_inner_product_layer_trajectories": pre_inner_tensor,

            "pre_ches_scores": pre_ches_tensor[:, -1],
            "pre_ln_ches_scores": pre_ln_ches_tensor[:, -1],
            "pre_last_hidden_embedding_inner_prods": pre_inner_tensor[:, -1],
        }

        
        torch.save(results, os.path.join(config["output_dir"], subdir_name, f"results_samples.pt"))

        end_time = datetime.utcnow()
        logger.info(f"Finished script, time took: {end_time - start_time}")
    except Exception:
        logger.exception("Exception while running script.")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output_dir", type=str, default="outputs/pref_similarity", help="Directory to save log file to")
    p.add_argument("--cache_dir", type=str, default=None, help="Directory of cache for HuggingFace models and datasets")
    p.add_argument("--dont_save_logs", action="store_true", help="Only log to console, and not to a file")
    p.add_argument("--model", type=str, default="allenai/OLMo-1B-hf", help="Model to use")
    p.add_argument("--dataset", type=str, default="tatsu-lab/alpaca_farm", help="Dataset to use")
    p.add_argument("--custom_dataset_display_name", type=str, default="", help="Name of dataset to use for creating file name")
    p.add_argument("--num_train_samples", type=int, default=-1,
                   help="Number of training samples to compute preference similarity for (if < 0, all samples are used)")
    p.add_argument("--train_samples_random_seed", type=int, default=-1, help="Random seed to use for selecting train samples")
    p.add_argument("--max_input_length", type=int, default=512,
                   help="Truncate outputs to this maximal length (if < 0, does not truncate)")
    p.add_argument("--chat_chosen_field", type=str, default="chosen", help="Field name for chosen output when using models with chat template")
    p.add_argument("--chat_rejected_field", type=str, default="rejected", help="Field name for rejected output when using models with chat template")
    p.add_argument("--gpu_id", type=int, default=-1, help="GPU id to use (-1 for CPU)")
    args = p.parse_args()

    main(args.__dict__)