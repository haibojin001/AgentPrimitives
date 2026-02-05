import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from typing import Optional, Tuple, List


def _pkv_to_legacy(past_kv):
    if isinstance(past_kv, DynamicCache):
        if hasattr(past_kv, "to_legacy_cache"):
            return past_kv.to_legacy_cache()
        raise RuntimeError("DynamicCache incompatible with current transformers.")
    if isinstance(past_kv, (tuple, list)) and len(past_kv) > 0:
        return list(past_kv)
    raise RuntimeError(f"Unsupported cache type: {type(past_kv)}")


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


class ReviewPrimitive:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        dtype: torch.dtype = torch.float16,
        max_solver_tokens: int = 30000,
        max_critic_tokens: int = 10000,
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
        self.max_critic_tokens = max_critic_tokens

        self._solver_init_template = [
            {
                "role": "system",
                "content": "You are an intelligent programmer. Complete the function provided by the user."
            }
        ]

        self._solver_refine_template = [
            {
                "role": "system",
                "content": "You are an intelligent programmer."
            },
            {
                "role": "user",
                "content": "Refine your previous solution based on the Critic feedback."
            }
        ]

        self._critic_template = [
            {
                "role": "system",
                "content": "You are an intelligent critic. Provide only feedback."
            }
        ]

    @torch.no_grad()
    def _solver_initial_kv(self, user_query: str) -> DynamicCache:
        messages = self._solver_init_template + [
            {"role": "user", "content": user_query}
        ]

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
    def _critic_kv(self, solver_cache: DynamicCache) -> DynamicCache:
        prompt = self.tokenizer.apply_chat_template(
            self._critic_template, tokenize=False, add_generation_prompt=True
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs, use_cache=True, return_dict=True)

        solver_legacy = _pkv_to_legacy(solver_cache)
        critic_legacy = _pkv_to_legacy(out.past_key_values)

        merged = []
        for (k1, v1), (k2, v2) in zip(solver_legacy, critic_legacy):
            merged.append((torch.cat([k1, k2], dim=2),
                           torch.cat([v1, v2], dim=2)))

        merged_cache = _legacy_to_dynamic(merged)

        first_id = torch.argmax(out.logits[:, -1, :], dim=-1)
        _, final_cache = _generate_ids(
            self.model,
            self.tokenizer,
            merged_cache,
            first_id,
            self.max_critic_tokens,
        )
        return final_cache

    @torch.no_grad()
    def _solver_refine_text(
        self,
        solver_cache: DynamicCache,
        critic_cache: DynamicCache,
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            self._solver_refine_template,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        out = self.model(**inputs, use_cache=True, return_dict=True)

        hist = _pkv_to_legacy(solver_cache)
        crit = _pkv_to_legacy(critic_cache)
        ref = _pkv_to_legacy(out.past_key_values)

        merged = []
        for (k0, v0), (k1, v1), (k2, v2) in zip(hist, crit, ref):
            merged.append(
                (torch.cat([k0, k1, k2], dim=2),
                 torch.cat([v0, v1, v2], dim=2))
            )

        merged_cache = _legacy_to_dynamic(merged)

        first_id = torch.argmax(out.logits[:, -1, :], dim=-1)
        gen_ids, _ = _generate_ids(
            self.model,
            self.tokenizer,
            merged_cache,
            first_id,
            self.max_solver_tokens,
        )

        return self.tokenizer.decode(gen_ids, skip_special_tokens=True)

    def run(self, user_query: str) -> str:
        solver_cache = self._solver_initial_kv(user_query)
        critic_cache = self._critic_kv(solver_cache)
        return self._solver_refine_text(solver_cache, critic_cache)
