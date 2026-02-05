import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from typing import List, Optional, Tuple


def _pkv_to_legacy(past_kv):
    if isinstance(past_kv, DynamicCache):
        return past_kv.to_legacy_cache()
    return list(past_kv)


def _legacy_to_dynamic(legacy_list):
    return DynamicCache.from_legacy_cache(legacy_list)


@torch.no_grad()
def _forward_step(model, cache, embed):
    out = model(
        inputs_embeds=embed,
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )
    logits = out.logits[:, -1, :]
    next_id = torch.argmax(logits, dim=-1)
    next_embed = model.get_input_embeddings()(next_id).unsqueeze(1)
    return next_id, next_embed, out.past_key_values


@torch.no_grad()
def _generate_ids(
    model,
    tokenizer,
    cache,
    first_id: torch.Tensor,
    max_tokens: int,
) -> Tuple[List[int], DynamicCache]:
    eos = tokenizer.eos_token_id
    device = model.device

    generated = [first_id.item()]
    if eos is not None and generated[-1] == eos:
        return generated, cache

    cur_embed = model.get_input_embeddings()(first_id.to(device)).unsqueeze(1)

    for _ in range(1, max_tokens):
        next_id, next_embed, cache = _forward_step(model, cache, cur_embed)
        tid = next_id.item()
        generated.append(tid)

        if eos is not None and tid == eos:
            break

        cur_embed = next_embed

    return generated, cache


class VotingPrimitive:

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        dtype: torch.dtype = torch.float16,
        max_solver_tokens: int = 30000,
        max_selector_tokens: int = 10000,
        device: Optional[str] = None,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

        self.max_solver_tokens = max_solver_tokens
        self.max_selector_tokens = max_selector_tokens

        self._solver_system = "You are an intelligent programmer. This is your solution."
        self._selector_system = "You are a careful selector. Produce the final answer."

    @torch.no_grad()
    def _run_single_solver_kv(self, messages) -> DynamicCache:
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs, use_cache=True, return_dict=True)

        first_id = torch.argmax(out.logits[:, -1, :], dim=-1)
        _, cache = _generate_ids(
            self.model,
            self.tokenizer,
            out.past_key_values,
            first_id,
            self.max_solver_tokens,
        )
        return cache

    @torch.no_grad()
    def _build_clean_kv(self, text: str) -> DynamicCache:
        msgs = [
            {"role": "system", "content": self._solver_system},
            {"role": "assistant", "content": text},
        ]

        txt = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=False
        )

        inputs = self.tokenizer(txt, return_tensors="pt").to(self.device)
        out = self.model(**inputs, use_cache=True, return_dict=True)
        return out.past_key_values

    @torch.no_grad()
    def _concat_caches(self, caches: List[DynamicCache]) -> DynamicCache:
        legacy_concat = None

        for cache in caches:
            legacy = _pkv_to_legacy(cache)

            if legacy_concat is None:
                legacy_concat = [(k.clone(), v.clone()) for k, v in legacy]
            else:
                new = []
                for (k1, v1), (k2, v2) in zip(legacy_concat, legacy):
                    new.append((torch.cat([k1, k2], dim=2),
                                torch.cat([v1, v2], dim=2)))
                legacy_concat = new

        return _legacy_to_dynamic(legacy_concat)

    @torch.no_grad()
    def _selector_over_kv(self, merged_cache: DynamicCache) -> str:
        selector_prompt = [
            {"role": "system", "content": self._selector_system},
            {"role": "user", "content": "Output a single final solution."},
        ]

        prompt = self.tokenizer.apply_chat_template(
            selector_prompt, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs, use_cache=True, return_dict=True)

        legacy_mix = _pkv_to_legacy(merged_cache)
        legacy_sel = _pkv_to_legacy(out.past_key_values)

        merged = []
        for (k_mix, v_mix), (k_sel, v_sel) in zip(legacy_mix, legacy_sel):
            merged.append((torch.cat([k_mix, k_sel], dim=2),
                           torch.cat([v_mix, v_sel], dim=2)))

        final_cache = _legacy_to_dynamic(merged)

        first_id = torch.argmax(out.logits[:, -1, :], dim=-1)
        gen_ids, _ = _generate_ids(
            self.model,
            self.tokenizer,
            final_cache,
            first_id,
            self.max_selector_tokens,
        )

        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def run(self, solver_prompts: List[List[dict]]) -> str:
        solver_caches = []

        for prompt in solver_prompts:
            raw_cache = self._run_single_solver_kv(prompt)
            text_stub = "<solver-output-placeholder>"
            clean_cache = self._build_clean_kv(text_stub)

            solver_caches.append(clean_cache)

        merged_cache = self._concat_caches(solver_caches)
        final_answer = self._selector_over_kv(merged_cache)

        return final_answer
