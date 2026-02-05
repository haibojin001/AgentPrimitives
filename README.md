# Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems

This repository provides a reference implementation of **Agent Primitives** — reusable latent building blocks for constructing LLM-based Multi-Agent Systems (MAS).
Instead of designing task-specific agent teams with handcrafted roles and verbose natural-language communication, primitives-based MAS composes a small set of recurring computation patterns (e.g., review, voting, planning) and enables **latent communication via KV-cache** (when supported by the backbone).

---

## 🛠️ Environment Setup

### 1️⃣ Clone the Repository
```bash
git clone AgentPrimitives.git
cd AgentPrimitives
```

### 2️⃣ Install Dependencies

```bash
conda create -n agentprims python=3.10 -y
conda activate agentprims
pip install -r requirements.txt
```
Using the HF cache directory to avoid repeated downloads:

```bash
export HF_HOME=/path/to/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export HF_DATASETS_CACHE=$HF_HOME
```

## 3️⃣ Agent Primitives

The core implementation of this repository is in the `primitives/` directory, which contains four modules:

```
primitives/
├── ReviewPrimitive.py       # Review Primitive: iterative latent self-critique
├── VotingPrimitive.py         # Voting & Selection Primitive: latent aggregation
├── PlanExecutePrimitive.py  # Planning & Execution Primitive: latent plan + execution
└── Organizer.py     # Organizer: dynamically composes primitives into a MAS
```

Each primitive communicates via KV-cache (when supported by the backbone) and exposes a standard LLM-like interface, so they can be composed and reused across tasks.

---

## 4️⃣ How to Run (Current Usage)

At the moment, you can run the system through the Organizer:

```python
from primitives.organizer import Organizer

org = Organizer()
```

This will dynamically design a primitive-based MAS for the given query.

---

## 🎯 Run Baselines

We provide the following baselines in our experiments:

- **Single Agent**
- **TextMAS**
- **LatentMAS**

To install the **LatentMAS** baseline, run:

```bash
bash baselines/latentmas.sh
```

## 🔜 Next Step: Demo (Coming Soon)

We will update the repository with a clean end-to-end demo script (e.g., `run_demo.py`) that shows how to:

* load a task,
* let the Organizer design a MAS, and
* execute the primitives step by step.

More detailed experiment pipelines and examples will be added gradually.




