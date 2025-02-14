import torch
def compute_logps_with_prompt_cache(
    model: torch.nn.Module,
    prompt_inputs: dict,
    completion_ids: torch.LongTensor,
    mini_batch_size: int,
    requires_grad_for_completion: bool = True,
) -> torch.FloatTensor:
    """
    The method will compute the log probabilities of the completion tokens by using the prompt cache.
    1) Forward pass on the prompt with torch.no_grad() to get `past_key_values`.
    2) Forward pass (with or without grad) on the completion tokens using that cache.
    3) Compute per-token log probabilities for the completion.
    Args:
        model (`nn.Module`): A causal LM (transformers.AutoModelForCausalLM) or similar.
        prompt_inputs (`dict`): The dict of prompt tensors, e.g. {"input_ids", "attention_mask", ...}.
        completion_ids (`torch.LongTensor`): Shape [B*G, completion_len].
        mini_batch_size (`int`): The number of completion rows to process at once.
        requires_grad_for_completion (`bool`): Whether to enable gradient for the completion pass.
    Returns:
        per_token_logps (`torch.FloatTensor`): shape [B*G, completion_len],
        where per_token_logps[i, t] is the logprob of ith completion's t-th completion token,
        given all preceding tokens in the prompt + the partial completion up to t-1.
    """

    # Get the batch size (B), number of completions (G), and completion length (C)
    B = prompt_inputs["input_ids"].size(0)
    G = completion_ids.size(0) // B
    C = completion_ids.size(1)

    # If the user did not specify a mini_batch_size, use the full batch size (B*G)
    if mini_batch_size <= 0:
        mini_batch_size = completion_ids.size(0)
    # print(model)
    # Forward pass over prompt tokens to get 2 things with torch.no_grad:
    # 1) `past_key_values`` (KV cache)
    # 2) `prompt_last_logps` (the logprobs of the first completion token prediction)
    with torch.no_grad():
        # print(prompt_inputs)
        prompt_out = model(**prompt_inputs, use_cache=True, return_dict=True)
        # print(prompt_out)

    # Only keep the last prompt logit, immediately convert to log probabilities and expand to B*G
    prompt_last_logps = prompt_out.logits[:, -1:].log_softmax(dim=-1).repeat_interleave(G, dim=0)

    # Gather the these log probs as they relates to the first completion token
    first_completion_token_logps = torch.gather(
        prompt_last_logps, dim=-1, index=completion_ids[:, :1].unsqueeze(-1)
    ).squeeze(-1)

    # Expand the KV Cache `G` times to match the dimension of completion_ids (B -> B*G) and split into mini-batches
    repeated_kv_cache = prompt_out.past_key_values  # a DynamicCache
    repeated_kv_cache.batch_repeat_interleave(G)
    mini_batch_kv_caches = repeated_kv_cache.batch_split(full_batch_size=B * G, split_size=mini_batch_size)

    # Process completion tokens in mini-batches
    completion_token_logps = []

    for batch_idx, mini_batch_kv_cache in enumerate(mini_batch_kv_caches):
        start_idx = batch_idx * mini_batch_size
        end_idx = start_idx + mini_batch_size
        mini_batch_ids = completion_ids[start_idx:end_idx]  # (mini_batch_size, C)

        with torch.set_grad_enabled(requires_grad_for_completion):
            mini_batch_logits = model(
                input_ids=mini_batch_ids,
                past_key_values=mini_batch_kv_cache,
                # num_logits_to_keep=C,
                use_cache=False,
            ).logits[:, -C:-1, :]

        # # Original method
        # mini_batch_log_probs = mini_batch_logits.log_softmax(dim=-1)
        # del mini_batch_logits

        # mini_batch_token_log_prob = torch.gather(mini_batch_log_probs, dim=-1, index=mini_batch_index).squeeze(-1)
        # del mini_batch_log_probs

        # More optimized method (https://github.com/huggingface/trl/pull/2773)
        # Get the corresponding completion token ids and gather the logits for completion_ids w/ idx >= 1
        mini_batch_index = mini_batch_ids[:, 1:].unsqueeze(-1)  # (mini_batch_size, C-1, 1)
        mini_batch_token_logits = torch.gather(mini_batch_logits, dim=-1, index=mini_batch_index).squeeze(
            -1
        )  # (mini_batch_size, C-1)
        mini_batch_logsumexp_values = torch.stack(
            [torch.logsumexp(l, dim=-1) for l in mini_batch_logits]
        )  # (mini_batch_size, C-1)
        del mini_batch_logits
        mini_batch_token_log_prob = mini_batch_token_logits - mini_batch_logsumexp_values  # (mini_batch_size, C-1)
        completion_token_logps.append(mini_batch_token_log_prob)
        del mini_batch_token_logits, mini_batch_logsumexp_values, mini_batch_token_log_prob

    # Combine results
    all_completion_token_logps = torch.cat(completion_token_logps, dim=0)  # (B*G, C-1)
    return torch.cat([first_completion_token_logps, all_completion_token_logps], dim=1)  # (B*G, C)
