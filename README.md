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
