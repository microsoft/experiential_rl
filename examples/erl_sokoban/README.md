# Experiential RL for Sokoban

This example trains a Sokoban language agent with the ERL workflow.

## Dataset Preparation

Sokoban training requires dataset registry entries for `sokoban` train/test splits.  
The dataset used in the paper is already preprocessed under `rllm_data/datasets`, so you can train directly without running preparation scripts.  
Only prepare a new dataset if you want custom task configurations (for example different map sizes or split sizes).

From repo root:

```bash
python3 -m examples.erl_sokoban.prepare_sokoban_data
```

Optional custom sizes (from repo root):

```bash
python3 -c "from examples.erl_sokoban.prepare_sokoban_data import prepare_sokoban_data; prepare_sokoban_data(train_size=20000, test_size=500, size_range=(6,8), max_solution_steps=8)"
```

## Training

Run from repo root:

```bash
bash train_scripts/train_erl_sokoban.sh
```

GRPO baseline script:

```bash
bash train_scripts/train_grpo_sokoban.sh
```

Both scripts call `examples.erl_sokoban.train_erl_sokoban_flow` and include their own training hyperparameters.

## Workflow Flag Meanings

ERL uses an experience -> reflection -> internalization loop. These flags control which parts of that loop are optimized:

- `train_first_attempt`: optimize the policy on the initial attempt (pre-reflection behavior).
- `train_second_attempt_raw`: optimize the reflection-guided second attempt in its raw form.
- `train_second_attempt_distilled`: optimize distilled second-attempt data for internalization (context distillation), so improvements transfer into the base policy.
- `train_updater`: optimize the reflection/updater stage itself (the component that produces improvement guidance).

Each `*_adv_estimator` field chooses how advantages are computed for that trajectory type:

- `grpo`: Group Relative Policy Optimization.
- `raft`: Reward-ranked Finetuning ([RAFT](https://arxiv.org/abs/2504.11343)); in practice, REINFORCE on positive-reward trajectories with additional stabilizing techniques (for example importance sampling and KL constraints).

In our current implementation, we apply GRPO to the first-attempt step, second-attempt step, and reflection/updater step. We apply RAFT to the internalization (context distillation) step. You can also switch to GAE or other advantage estimators supported by verl/rLLM.

For `train_scripts/train_grpo_sokoban.sh`, the configured behavior is:

- optimize first attempts only (`train_first_attempt=True`, `train_first_attempt_adv_estimator=grpo`).
- disable second-attempt optimization (`train_second_attempt_raw=False`, `train_second_attempt_distilled=False`).
- disable updater/reflection-policy optimization (`train_updater=False`).
- keep other `*_adv_estimator` fields set as defaults in case those branches are enabled later.
