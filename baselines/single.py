from typing import Dict, List, Tuple, Optional

from models import ModelWrapper
from utils import (
    extract_gsm8k_answer,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
)


def build_agent_messages_single_agent(question: str, args=None):

    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert args.method in ["baseline"], "this prompt only for baseline method (single agent)"
    assert "qwen" in args.model_name.lower(), "this prompt only for qwen models"

    task = args.task

    if task in ["gsm8k", "aime2024", "aime2025"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    elif task in ["mbppplus", "humanevalplus"]:
        user_content = f"""
Target Question: {question}

You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
Now, reason step by step and output the final answer:
"""

    elif task in ["winogrande"]:
        user_content = f"""
Target Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

    else:
        user_content = f"""
Question: {question}

You are a helpful assistant.

You must reason step-by-step to solve the question without outputting other irrelevant information.
Present your reasoning, and then clearly state your final answer at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]



class SingleAgentMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        use_vllm: bool = False,
        args=None,
    ) -> None:

        self.model = model
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.use_vllm = use_vllm
        self.method_name = "baseline"
        self.args = args
        self.task = args.task

    def build_batch_messages(self, items: List[Dict]):
        return [
            build_agent_messages_single_agent(
                question=item["question"], args=self.args
            )
            for item in items
        ]

    def generate_texts(self, prompts, input_ids, attention_mask):
        if self.use_vllm:
            return self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            texts, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return texts

    def evaluate_prediction(
        self, generated_text: str, item: Dict
    ) -> Tuple[bool, Optional[str], Optional[str]]:

        if self.task in ["mbppplus", "humanevalplus"]:
            return self._eval_code_task(generated_text, item)

        if self.task in ["aime2024", "aime2025"]:
            return self._eval_aime_task(generated_text, item)

        return self._eval_gsm8k_like(generated_text, item)

    def _eval_code_task(self, generated_text: str, item: Dict):
        pred = extract_markdown_python_block(generated_text)
        gold = item.get("gold", "")

        if pred is None:
            return False, "python error: No python code block found", None

        python_code_to_exe = pred + "\n" + gold
        ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

        print("=========================================")
        print("Question")
        print(f"error_msg: {error_msg}")

        return ok, error_msg, pred

    def _eval_aime_task(self, generated_text: str, item: Dict):
        pred = normalize_answer(extract_gsm8k_answer(generated_text))
        gold = str(item.get("gold", "")).strip()

        try:
            ok = (int(pred) == int(gold))
            error_msg = None
        except ValueError:
            ok = False
            error_msg = (
                f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
            )

        return ok, error_msg, pred

    def _eval_gsm8k_like(self, generated_text: str, item: Dict):
        pred = normalize_answer(extract_gsm8k_answer(generated_text))
        gold = item.get("gold", "")
        ok = (pred == gold) if (pred and gold) else False
        return ok, None, pred


    def run_batch(self, items: List[Dict]) -> List[Dict]:

        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")


        batch_messages = self.build_batch_messages(items)

        prompts, input_ids, attention_mask, tokens_batch = \
            self.model.prepare_chat_batch(
                batch_messages, add_generation_prompt=True
            )

        generated_batch = self.generate_texts(
            prompts, input_ids, attention_mask
        )

        results: List[Dict] = []

        for idx, item in enumerate(items):

            generated_text = generated_batch[idx]

            ok, error_msg, pred = self.evaluate_prediction(
                generated_text, item
            )

            mask = attention_mask[idx].bool()
            trimmed_ids = input_ids[idx][mask].cpu().tolist()

            agent_trace = {
                "name": "SingleAgent",
                "role": "singleagent",
                "input": prompts[idx],
                "input_ids": trimmed_ids,
                "input_tokens": tokens_batch[idx],
                "output": generated_text,
            }

            results.append(
                {
                    "question": item["question"],
                    "gold": item.get("gold", ""),
                    "solution": item.get("solution", ""),
                    "prediction": pred,
                    "raw_prediction": generated_text,
                    "agents": [agent_trace],
                    "correct": ok,
                }
            )

        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
