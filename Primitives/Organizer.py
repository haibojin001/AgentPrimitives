import json
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from primitives.ReviewPrimitive import ReviewPrimitive
from primitives.VotingPrimitive import VotingPrimitive
from primitives.PlanExecutePrimitive import PlanExecutePrimitive


class Organizer:
    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        knowledge_pool: Optional[Any] = None,
    ):
        self.cfg = config or {}
        self.knowledge_pool = knowledge_pool

        org_model_name = self.cfg.get("organizer_model_name", self.cfg.get("model_name", "Qwen/Qwen3-8B"))
        org_dtype = self.cfg.get("organizer_dtype", torch.float16)
        self.org_device = self.cfg.get("organizer_device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

        self.org_tokenizer = AutoTokenizer.from_pretrained(org_model_name, trust_remote_code=True)
        self.org_model = AutoModelForCausalLM.from_pretrained(
            org_model_name,
            torch_dtype=org_dtype,
            device_map="auto" if self.org_device == "cuda" else None,
            trust_remote_code=True,
        )
        if self.org_device != "cuda":
            self.org_model = self.org_model.to(self.org_device)
        self.org_model.eval()

        self.review = ReviewPrimitive(
            model_name=self.cfg.get("model_name", "Qwen/Qwen3-8B"),
            max_solver_tokens=self.cfg.get("review_max_solver_tokens", 30000),
            max_critic_tokens=self.cfg.get("review_max_critic_tokens", 10000),
        )

        self.vote = VotingPrimitive(
            model_name=self.cfg.get("model_name", "Qwen/Qwen3-8B"),
            max_solver_tokens=self.cfg.get("vote_max_solver_tokens", 30000),
            max_selector_tokens=self.cfg.get("vote_max_selector_tokens", 10000),
        )

        self.plan_execute = PlanExecutePrimitive(
            model_name=self.cfg.get("model_name", "Qwen/Qwen3-8B"),
            max_planner_tokens=self.cfg.get("planner_max_tokens", 30000),
            max_executor_tokens=self.cfg.get("executor_max_tokens", 15000),
            num_executors=self.cfg.get("num_executors", 3),
        )

        self.max_plan_tokens = self.cfg.get("organizer_max_plan_tokens", 2048)

    def _kp_retrieve(self, query: str) -> List[Dict[str, Any]]:
        if self.knowledge_pool is None:
            return []
        try:
            out = self.knowledge_pool.retrieve(query)
            if out is None:
                return []
            if isinstance(out, list):
                return out
            if isinstance(out, dict):
                return [out]
            return []
        except Exception:
            return []

    def _organizer_prompt(self, query: str, kp_examples: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        kp_text = ""
        if kp_examples:
            packed = []
            for ex in kp_examples[:5]:
                packed.append(json.dumps(ex, ensure_ascii=False))
            kp_text = "\n".join(packed)

        schema = {
            "plan_version": "v1",
            "nodes": [
                {"id": "n1", "type": "plan_execute", "params": {"num_executors": 3}},
                {"id": "n2", "type": "vote", "params": {"num_candidates": 3}},
                {"id": "n3", "type": "review", "params": {"iterations": 1}},
            ],
            "pipeline": ["n1", "n2", "n3"],
            "output": "n3",
        }

        sys = (
            "You are the Organizer. Your job is to design a primitive-based multi-agent system for the input query.\n"
            "You must NOT solve the task.\n"
            "You must output a JSON plan only, following the required schema.\n"
            "Use only these primitive types: review, vote, plan_execute.\n"
            "Return JSON only. No extra text."
        )

        user = (
            "Input Query:\n"
            f"{query}\n\n"
            "Available Primitives:\n"
            "- review: iterative self-critique refinement\n"
            "- vote: latent voting/selection over multiple candidates\n"
            "- plan_execute: planner produces executor prompts; executors solve conditioned on latent plan\n\n"
            "Knowledge Pool Examples (structural guidance only):\n"
            f"{kp_text if kp_text else '[NONE]'}\n\n"
            "Required JSON schema:\n"
            f"{json.dumps(schema, ensure_ascii=False)}\n\n"
            "Constraints:\n"
            "- nodes: list of node objects {id,type,params}\n"
            "- pipeline: ordered list of node ids\n"
            "- output: id of final node\n"
            "- params must be minimal and task-agnostic\n"
            "- Do not include any task solution content\n"
        )

        return [{"role": "system", "content": sys}, {"role": "user", "content": user}]

    @torch.no_grad()
    def _llm_plan(self, query: str) -> Dict[str, Any]:
        kp = self._kp_retrieve(query)
        msgs = self._organizer_prompt(query, kp)
        prompt = self.org_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs = self.org_tokenizer(prompt, return_tensors="pt").to(self.org_device)
        out = self.org_model(**inputs, use_cache=False, return_dict=True)
        first_id = torch.argmax(out.logits[:, -1, :], dim=-1)

        gen = [first_id.item()]
        eos = self.org_tokenizer.eos_token_id
        cur = self.org_model.get_input_embeddings()(first_id.to(self.org_model.device)).unsqueeze(1)

        for _ in range(1, self.max_plan_tokens):
            o = self.org_model(inputs_embeds=cur, use_cache=False, return_dict=True)
            nid = torch.argmax(o.logits[:, -1, :], dim=-1)
            tid = nid.item()
            gen.append(tid)
            if eos is not None and tid == eos:
                break
            cur = self.org_model.get_input_embeddings()(nid).unsqueeze(1)

        txt = self.org_tokenizer.decode(gen, skip_special_tokens=True).strip()
        plan = self._parse_plan_json(txt)
        plan = self._validate_or_fallback(plan, query)
        return plan

    def _parse_plan_json(self, text: str) -> Dict[str, Any]:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            return {}
        blob = m.group(0)
        try:
            return json.loads(blob)
        except Exception:
            blob2 = re.sub(r",\s*([}\]])", r"\1", blob)
            try:
                return json.loads(blob2)
            except Exception:
                return {}

    def _validate_or_fallback(self, plan: Dict[str, Any], query: str) -> Dict[str, Any]:
        def ok_id(x):
            return isinstance(x, str) and len(x) > 0

        if not isinstance(plan, dict):
            plan = {}

        nodes = plan.get("nodes", None)
        pipeline = plan.get("pipeline", None)
        output = plan.get("output", None)

        if not isinstance(nodes, list) or not isinstance(pipeline, list) or not ok_id(output):
            return self._fallback_plan(query)

        node_map = {}
        for n in nodes:
            if not isinstance(n, dict):
                continue
            nid = n.get("id", None)
            ntype = n.get("type", None)
            params = n.get("params", {})
            if ok_id(nid) and ntype in {"review", "vote", "plan_execute"} and isinstance(params, dict):
                node_map[nid] = {"id": nid, "type": ntype, "params": params}

        if not node_map:
            return self._fallback_plan(query)

        if any(x not in node_map for x in pipeline) or output not in node_map:
            return self._fallback_plan(query)

        plan["nodes"] = [node_map[n["id"]] for n in nodes if isinstance(n, dict) and n.get("id") in node_map]
        return plan

    def _fallback_plan(self, query: str) -> Dict[str, Any]:
        is_long = len(query) > 600
        looks_like_coding = ("def " in query) or ("class " in query) or ("```" in query)
        needs_decompose = any(k in query.lower() for k in ["plan", "steps", "subtask", "decompose", "workflow"])

        nodes = []
        pipeline = []

        if needs_decompose or is_long:
            nodes.append({"id": "n1", "type": "plan_execute", "params": {"num_executors": int(self.cfg.get("num_executors", 3))}})
            pipeline.append("n1")

        if is_long:
            nodes.append({"id": "n2", "type": "vote", "params": {"num_candidates": 3}})
            pipeline.append("n2")

        if looks_like_coding or not pipeline:
            nodes.append({"id": "n3", "type": "review", "params": {"iterations": 1}})
            pipeline.append("n3")

        return {"plan_version": "v1", "nodes": nodes, "pipeline": pipeline, "output": pipeline[-1]}

    def plan_only(self, query: str) -> Dict[str, Any]:
        return self._llm_plan(query)

    def solve(self, query: str) -> str:
        plan = self._llm_plan(query)
        node_map = {n["id"]: n for n in plan["nodes"]}

        state: Dict[str, Any] = {"input": query}
        last_out: Any = query

        for nid in plan["pipeline"]:
            node = node_map[nid]
            t = node["type"]
            p = node.get("params", {})

            if t == "review":
                last_out = self.review.run(str(last_out))

            elif t == "vote":
                k = int(p.get("num_candidates", 3))
                base = str(last_out)
                solver_prompts = []
                styles = [
                    "You are a concise solver.",
                    "You are a careful solver focusing on correctness.",
                    "You produce readable code and robust edge handling.",
                    "You are a solver that double-checks corner cases.",
                ]
                for i in range(k):
                    solver_prompts.append(
                        [
                            {"role": "system", "content": styles[i % len(styles)]},
                            {"role": "user", "content": base},
                        ]
                    )
                last_out = self.vote.run(solver_prompts)

            elif t == "plan_execute":
                _, outs = self.plan_execute.run(str(last_out))
                last_out = outs[0] if outs else str(last_out)

            else:
                raise ValueError(f"Unknown primitive type: {t}")

            state[nid] = last_out

        return str(last_out)
