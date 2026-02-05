import re
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


def _strip_think_ids(tokenizer, token_ids: List[int]) -> List[int]:
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    if "<think>" not in text:
        return token_ids
    if "</think>" in text:
        visible = text.split("<think>", 1)[0] + text.split("</think>", 1)[1]
    else:
        visible = text.split("<think>", 1)[1]
    return tokenizer.encode(visible, add_special_tokens=False)


@torch.no_grad()
def _chat_forward_ids(model, tokenizer, messages, max_tokens: int) -> Tuple[List[int], DynamicCache, str]:
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True, return_dict=True)
    first_id = torch.argmax(out.logits[:, -1, :], dim=-1)
    gen_ids, cache = _generate_ids(model, tokenizer, out.past_key_values, first_id, max_tokens)
    return gen_ids, cache, prompt


@torch.no_grad()
def _build_clean_kv(model, tokenizer, system_text: str, assistant_text: str) -> DynamicCache:
    msgs = [
        {"role": "system", "content": system_text},
        {"role": "assistant", "content": assistant_text},
    ]
    txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
    inputs = tokenizer(txt, return_tensors="pt").to(model.device)
    out = model(**inputs, use_cache=True, return_dict=True)
    return out.past_key_values


def _extract_exec_prompts(planner_text: str, expected_k: int = 3) -> List[str]:
    items = re.findall(r"<ExecPrompt\d+>(.*?)</ExecPrompt\d+>", planner_text, re.S)
    items = [x.strip() for x in items if x.strip()]
    if len(items) >= expected_k:
        return items[:expected_k]
    return items


class PlanExecutePrimitive:
    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-8B",
        dtype: torch.dtype = torch.float16,
        max_planner_tokens: int = 30000,
        max_executor_tokens: int = 15000,
        device: Optional[str] = None,
        num_executors: int = 3,
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        if self.device != "cuda":
            self.model = self.model.to(self.device)

        self.model.eval()

        self.max_planner_tokens = max_planner_tokens
        self.max_executor_tokens = max_executor_tokens
        self.num_executors = num_executors

        self._planner_system_kv = "You are a planning agent. This is your plan."
        self._executor_system = "You are an executor following a latent plan."
        self._executor_system_kv = "You are an executor. This is your final solution."

    def _planner_messages(self, task: str) -> List[dict]:
        return [
            {
                "role": "system",
                "content": "You are a planning agent. Break down the task and design executor prompts.",
            },
            {
                "role": "user",
                "content": (
                    f"Task:\n{task}\n\n"
                    "Rules:\n"
                    "1. First output a plan beginning with <Plan>:\n"
                    f"2. Then output EXACTLY {self.num_executors} executor prompts in tags:\n"
                    + "\n".join([f"   <ExecPrompt{i+1}>...</ExecPrompt{i+1}>" for i in range(self.num_executors)]) + "\n"
                    "3. Executors will only see their OWN prompt, not the plan."
                ),
            },
        ]

    def _executor_messages(self, exec_prompt: str) -> List[dict]:
        return [
            {"role": "system", "content": self._executor_system},
            {"role": "user", "content": exec_prompt},
        ]

    @torch.no_grad()
    def _planner_plan_and_kv(self, task: str) -> Tuple[str, DynamicCache, List[str]]:
        ids, _, _ = _chat_forward_ids(
            self.model,
            self.tokenizer,
            self._planner_messages(task),
            self.max_planner_tokens,
        )

        clean_ids = _strip_think_ids(self.tokenizer, ids)
        planner_text = self.tokenizer.decode(clean_ids, skip_special_tokens=True)

        exec_prompts = _extract_exec_prompts(planner_text, expected_k=self.num_executors)

        planner_kv = _build_clean_kv(
            self.model,
            self.tokenizer,
            self._planner_system_kv,
            planner_text,
        )

        return planner_text, planner_kv, exec_prompts

    @torch.no_grad()
    def _executor_from_latent_plan(self, planner_kv: DynamicCache, exec_prompt: str) -> str:
        ids_exec, cache_exec, _ = _chat_forward_ids(
            self.model,
            self.tokenizer,
            self._executor_messages(exec_prompt),
            max_tokens=1,
        )

        legacy_plan = _pkv_to_legacy(planner_kv)
        legacy_exec = _pkv_to_legacy(cache_exec)

        merged = []
        for (kp, vp), (ke, ve) in zip(legacy_plan, legacy_exec):
            merged.append((torch.cat([kp, ke], dim=2), torch.cat([vp, ve], dim=2)))

        merged_cache = _legacy_to_dynamic(merged)

        first_id = torch.tensor([ids_exec[0]], device=self.model.device)
        gen_ids, _ = _generate_ids(
            self.model,
            self.tokenizer,
            merged_cache,
            first_id,
            self.max_executor_tokens,
        )

        clean_ids = _strip_think_ids(self.tokenizer, gen_ids)
        return self.tokenizer.decode(clean_ids, skip_special_tokens=True)

    def run(self, task: str) -> Tuple[List[str], List[str]]:
        planner_text, planner_kv, exec_prompts = self._planner_plan_and_kv(task)

        outputs = []
        for p in exec_prompts:
            outputs.append(self._executor_from_latent_plan(planner_kv, p))

        return exec_prompts, outputs
