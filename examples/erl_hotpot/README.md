# Experiential RL for HotpotQA

This tutorial trains the ERL HotpotQA workflow (`examples.erl_hotpot`) with a retrieval tool server and HotpotQA data.

## Overview

By the end, you will have:

1. Prepared HotpotQA train/validation datasets in `DatasetRegistry`
2. Started a retrieval server for tool calls
3. Launched ERL (or GRPO baseline) training from `train_scripts/hotpot`

---

## Setup

From repo root, make sure you have:

- rLLM + verl installed (see root `README.md`)
- LangGraph dependencies installed

Install LangGraph dependencies:

```bash
pip install langchain-openai langgraph
```

Install retrieval server dependencies (recommend a fresh env):

```bash
conda create -n rag-server python=3.10 pip -y
conda activate rag-server
pip install faiss-gpu==1.7.2 Flask numpy==1.26.4 sentence-transformers torch
```

---

## 1. Prepare Dataset and Retrieval Data (Required)

ERL HotpotQA training requires these registry entries:

- `hotpotqa` / `train`
- `hotpotqa-small` / `test`

Prepare data with the LangGraph data scripts:

```bash
cd examples/sdk/langgraph
python data/prepare_hotpotqa_data.py
python data/download_search_data.py --data_dir ./search_data
cat search_data/prebuilt_indices/part_aa search_data/prebuilt_indices/part_ab > search_data/prebuilt_indices/e5_Flat.index
mv search_data/wikipedia/wiki-18.jsonl search_data/prebuilt_indices/corpus.json
```

---

## 2. Start Retrieval Server

Start retrieval server on port `9002`:

```bash
cd examples/sdk/langgraph
bash launch_server.sh ./search_data/prebuilt_indices 9002
```

---

## 3. Train

Run from repo root.

### 3.1 ERL (Qwen)

```bash
bash train_scripts/hotpot/train_erl_hotpot.sh
```

### 3.2 ERL (Olmo)

Before running Olmo training, apply the local vLLM patch to enable Olmo tool calling:

```bash
cd scripts/olmo_patch
bash patch_vllm.sh
```

Then run:

```bash
bash train_scripts/hotpot/train_erl_hotpot_olmo.sh
```

### 3.3 GRPO baselines

```bash
bash train_scripts/hotpot/train_grpo_hotpot.sh
bash train_scripts/hotpot/train_grpo_hotpot_olmo.sh
```

These scripts launch `examples.erl_hotpot.train_erl_hotpot_flow` and configure workflow flags for ERL vs GRPO behavior.

---

## 4. Workflow Flag Meanings

ERL uses an experience -> reflection -> internalization loop. These flags control which parts of that loop are optimized:

- `train_first_attempt`: optimize the policy on the initial attempt (pre-reflection behavior).
- `train_second_attempt_raw`: optimize the reflection-guided second attempt in its raw form.
- `train_second_attempt_distilled`: optimize distilled second-attempt data for internalization (context distillation), so improvements transfer into the base policy.
- `train_updater`: optimize the reflection/updater stage itself (the component that produces improvement guidance).

Each `*_adv_estimator` field chooses how advantages are computed for that trajectory type:

- `grpo`: GRPO.
- `raft`: Reward-ranked Fine-Tuning ([RAFT](https://arxiv.org/abs/2504.11343)); in practice, REINFORCE on positive-reward trajectories with additional stabilizing techniques (for example importance sampling and KL constraints).

In our current implementation, we apply GRPO to the first-attempt step, second-attempt step, and reflection/updater step. We apply RAFT to the internalization (context distillation) step. You can also switch to GAE or other advantage estimators supported by verl/rLLM.

For the GRPO baseline scripts, the configured behavior is:

- optimize first attempts only (`train_first_attempt=True`, `train_first_attempt_adv_estimator=grpo`).
- disable second-attempt optimization (`train_second_attempt_raw=False`, `train_second_attempt_distilled=False`).
- disable updater/reflection-policy optimization (`train_updater=False`).
- keep other `*_adv_estimator` fields set as defaults in case those branches are enabled later.

---

## Notes

- `ERL_HOTPOT_RETRIEVER_URL` must point to a reachable retrieval server for all workers.
