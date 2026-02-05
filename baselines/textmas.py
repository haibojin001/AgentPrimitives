from typing import Dict, List, Tuple, Optional
import argparse
from . import default_agents
from models import ModelWrapper
from utils import (
    extract_gsm8k_answer,
    normalize_answer,
    extract_markdown_python_block,
    run_with_timeout,
)

AGENT_NAME_MAP = {
    "Planner": "Math Agent",
    "Critic": "Science Agent",
    "Refiner": "Code Agent",
    "Judger": "Task Summrizer",
    "planner": "Math Agent",
    "critic": "Science Agent",
    "refiner": "Code Agent",
    "judger": "Task Summrizer",
}


def build_agent_messages_sequential_text_mas(role: str, question: str, context: str = "", method=None, args=None):
    system_message = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

    assert method in ["text_mas"], "only for text_mas method"
    assert "qwen" in args.model_name.lower(), "only for qwen models"

    # truncate context if needed
    ctx = context[: args.text_mas_context_length]

    if role == "planner":
        user_content = f"""
You are a Planner Agent. Given an input question, design a clear, step-by-step plan for how to solve the question.

## Input Question:
{question}

Your outlined plan should be concise with a few bullet points for each step. Do not produce the final answer.

## Format your response as follows:
Planner Agent's Output:
[Your detailed plan here]

Now output your plan to solve the question below:
"""

    elif role == "critic":
        user_content = f"""
You are a Critic Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan in text format.

Your job is to carefully evaluate the correctness and completeness of the plan and provide helpful feedback.

## Input Question:
{question}

## Plan from Planner Agent:
{ctx}

## Format your response as follows:
Critic Agent's Output:
Original Plan: [Copy the provided Planner Agent's plan here]
Feedback: [Your detailed feedback to improve the plan here]

Now, output your response below:
"""

    elif role == "refiner":
        user_content = f"""
You are a Refiner Agent. You are provided with:
(1) the original question, and
(2) the Planner Agent's plan together with Critic Agent's feedback in text format.

Your job is to incorporate the feedback and produce an improved, refined step-by-step plan.

## Input Question:
{question}

## Original Plan and Critic Feedback:
{ctx}

## Format your response as follows:
Refiner Agent's Output:
[Your refined and improved plan here]

Make sure your output plan is logically correct, concise, and sufficient to guide final problem solving.
Now, output your refined plan below:
"""

    elif role == "judger":
        task = getattr(args, "task", None)

        if task in ["gsm8k", "aime2024", "aime2025"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["arc_easy", "arc_challenge", "gpqa", "medqa"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from A,B,C,D. For example \\boxed{{A}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""

        elif task in ["mbppplus", "humanevalplus"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
You must put all python code as self-contained Python function(s) in markdown code blocks. For example:
```python
import math
def add(a, b):
    return a + b
```
Do not add any other contents inside the markdown code block.
"""

        elif task in ["winogrande"]:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.
Your final answer must be selected from 1 and 2. For example \\boxed{{1}} or \\boxed{{2}}. Do not add any other contents inside the box.

Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}.
"""
        else:
            user_content = f"""
Target Question: {question}

You are the final solver agent in a sequential multi-agent system (planner -> critic -> refiner -> solver).
You are provided with the Refiner Agent's plan as reference.

Refined Plan from Previous Agents:
{ctx}

The plan might contain irrelevant or incorrect contents. Ignore them if they are not helpful for solving the target question.

You must reason step-by-step to solve the **provided Target Question** without outputting other irrelevant information.

Now, reason step by step and present your final answer clearly at the end.
"""

    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_content},
    ]


class TextMASMethod:
    def __init__(
        self,
        model: ModelWrapper,
        *,
        max_new_tokens_each: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.95,
        generate_bs: int = 1,
        args: argparse.Namespace = None,
    ) -> None:

        self.model = model
        self.max_new_tokens_each = max_new_tokens_each
        self.max_new_tokens_judger = max_new_tokens_each
        self.temperature = temperature
        self.top_p = top_p
        self.generate_bs = max(1, generate_bs)
        self.agents = default_agents()
        self.args = args
        self.method_name = "text_mas"
        self.task = args.task

    def build_batch_messages(
        self, agent, items: List[Dict], contexts: List[str]
    ) -> List[List[Dict]]:

        builder = (
            build_agent_messages_hierarchical_text_mas
            if self.args.prompt == "hierarchical"
            else build_agent_messages_sequential_text_mas
        )

        return [
            builder(
                role=agent.role,
                question=item["question"],
                context=contexts[idx],
                method=self.method_name,
                args=self.args,
            )
            for idx, item in enumerate(items)
        ]

    def generate_texts(
        self, prompts, input_ids, attention_mask
    ) -> List[str]:

        if self.model.use_vllm:
            return self.model.vllm_generate_text_batch(
                prompts,
                max_new_tokens=self.max_new_tokens_each,
                temperature=self.temperature,
                top_p=self.top_p,
            )
        else:
            texts, _ = self.model.generate_text_batch(
                input_ids,
                attention_mask,
                max_new_tokens=self.max_new_tokens_each,
                temperature=self.temperature,
                top_p=self.top_p,
            )
            return texts

    def format_agent_output(self, agent, text_out: str) -> str:
        if self.args.prompt == "hierarchical":
            name = AGENT_NAME_MAP.get(agent.name, agent.name)
        else:
            name = agent.name

        return f"[{name}]:\n{text_out}\n\n"

    def evaluate_prediction(
        self, final_text: str, item: Dict
    ) -> Tuple[bool, Optional[str], Optional[str]]:

        if self.task in ["mbppplus", "humanevalplus"]:
            return self._eval_code_task(final_text, item)

        if self.task in ["aime2024", "aime2025"]:
            return self._eval_aime_task(final_text, item)

        return self._eval_gsm8k_like(final_text, item)

    def _eval_code_task(self, final_text: str, item: Dict):
        pred = extract_markdown_python_block(final_text)
        gold = item.get("gold", "")

        if pred is None:
            return False, "python error: No python code block found", None

        python_code_to_exe = pred + "\n" + gold
        ok, error_msg = run_with_timeout(python_code_to_exe, timeout=10)

        print("=========================================")
        print("Question")
        print(f"error_msg: {error_msg}")

        return ok, error_msg, pred

    def _eval_aime_task(self, final_text: str, item: Dict):
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = str(item.get("gold", "")).strip()

        try:
            pred_int = int(pred)
            gold_int = int(gold)
            ok = (pred_int == gold_int)
            error_msg = None
        except ValueError:
            ok = False
            error_msg = (
                f"Value error in parsing answer. Pred: {pred}, Gold: {gold}"
            )

        return ok, error_msg, pred

    def _eval_gsm8k_like(self, final_text: str, item: Dict):
        pred = normalize_answer(extract_gsm8k_answer(final_text))
        gold = item.get("gold", "")
        ok = (pred == gold) if (pred and gold) else False
        return ok, None, pred

    def run_batch(self, items: List[Dict]) -> List[Dict]:

        if len(items) > self.generate_bs:
            raise ValueError("Batch size exceeds configured generate_bs")

        batch_size = len(items)

        contexts = ["" for _ in range(batch_size)]
        history_contexts = ["" for _ in range(batch_size)]
        agent_traces: List[List[Dict]] = [[] for _ in range(batch_size)]
        final_texts = ["" for _ in range(batch_size)]

        for agent in self.agents:

            batch_messages = self.build_batch_messages(
                agent, items, contexts
            )

            prompts, input_ids, attention_mask, tokens_batch = \
                self.model.prepare_chat_batch(
                    batch_messages,
                    add_generation_prompt=True,
                )

            generated_texts = self.generate_texts(
                prompts, input_ids, attention_mask
            )

            for idx in range(batch_size):
                text_out = generated_texts[idx].strip()
                formatted_output = self.format_agent_output(agent, text_out)

                if agent.role != "judger":
                    contexts[idx] += formatted_output
                    history_contexts[idx] += formatted_output
                else:
                    final_texts[idx] = text_out

                mask = attention_mask[idx].bool()
                trimmed_ids = input_ids[idx][mask].cpu().tolist()

                agent_traces[idx].append(
                    {
                        "name": agent.name,
                        "role": agent.role,
                        "input": prompts[idx],
                        "input_ids": trimmed_ids,
                        "input_tokens": tokens_batch[idx],
                        "output": text_out,
                    }
                )

        results: List[Dict] = []

        for idx, item in enumerate(items):

            ok, error_msg, pred = self.evaluate_prediction(
                final_texts[idx], item
            )

            results.append(
                {
                    "question": item["question"],
                    "gold": item.get("gold", ""),
                    "solution": item.get("solution", ""),
                    "context": history_contexts[idx],
                    "prediction": pred,
                    "raw_prediction": final_texts[idx],
                    "agents": agent_traces[idx],
                    "correct": ok,
                }
            )

        return results

    def run_item(self, item: Dict) -> Dict:
        return self.run_batch([item])[0]
