# RoboMemArena Task Evaluation Code Guide

- Project: RoboMemArena evaluation benchmark
- Repo: RoboMemArena
- Type: implementation guide

## 1. Scope

This document explains how the RoboMemArena `task1` and `task2..26` evaluation benchmark is organized.
It focuses on:

- code structure
- task entrypoints
- stage success logic
- goal success logic
- how to connect a user-trained policy

This is **not** a training document and **not** tied to any one policy implementation.
Users only need to provide a policy adapter that maps environment observations plus prompt into an action chunk.

## 2. Core paths inside this package

### 2.1 Evaluation entrypoints

- `scripts/eval_task1_only.py`
- `scripts/eval_tasks2_26.py`
- `scripts/eval_common.py`
- `scripts/run_all_tasks1_26_until_stage_nonzero.py`
- `reference_evaluation/task1_nomap_reference/eval_task1_nomap_reference.py`
- `reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py`

### 2.2 Benchmark dependencies included here

- task prompt text is derived from the task id / BDDL file stem inside `scripts/eval_common.py`
- BDDL: `bddl/`
- LIBERO fork used by this benchmark: `libero_fork/`

### 2.3 Policy integration point

- adapter interface: `scripts/policy_adapter.py`
- adapter template: `scripts/example_policy_adapter_template.py`

## 3. High-level evaluation structure

The benchmark has two layers.

### 3.1 Environment layer

`eval_common.py` is responsible for:

- resolving `task_id -> bddl path`
- creating `OffScreenRenderEnv`
- checking final BDDL goal success
- saving rollout videos

### 3.2 Policy layer

The benchmark does not assume OpenPI or any specific model stack.
Instead, it calls a user-provided adapter:

- input: raw environment observation `obs`, task prompt `prompt`, target `resize_size`
- output: action chunk with shape `[horizon, action_dim]`

This means users can connect:

- VLA models
- imitation learning policies
- diffusion policies
- transformer policies
- any custom controller that can output an action chunk

## 4. task1 versus task2..26

### 4.1 task1

`task1` uses:

- `scripts/eval_task1_only.py`
- `reference_evaluation/task1_nomap_reference/eval_task1_nomap_reference.py` for the Task 1 evaluation code

Its default visual preprocessing path is:

- environment render resolution: `640x480`
- policy resize target: `256`

This is now aligned with the legacy task1 old-run setup and no longer depends on the batch runner to inject the environment resolution patch indirectly.

Its two stage checks are:

1. `cookies_in_basket`
2. `tomato_in_basket`

### 4.2 task2..26

`task2..26` uses:

- `scripts/eval_tasks2_26.py`
- `reference_evaluation/tasks2_26_vlm5_reference/eval_tasks2_26_vlm_vla.py` for the VLM5 Tasks 2-26 evaluation code

Its stage definitions are centralized in:

- `_task_specs(task_id)`

Each stage is a `StageSpec(name, check_fn)`.

## 5. Where prompt and goal come from

### 5.1 Prompt

Prompts are defined in:

- task prompt text derivation in `scripts/eval_common.py`

The evaluation code maps `task_id -> task_key -> prompt`.

### 5.2 Goal

The final task goal comes from the corresponding BDDL file in:

- `bddl/`

`eval_common.py` parses the BDDL goal expression and checks whether the final environment state satisfies it.

## 6. Stage success versus goal success

These are different metrics.

### 6.1 Stage success

Stage success measures process completion.
Each task defines a sequence of stage checks. A stage is marked complete once its check function becomes true.

At the end of an episode:

```text
stage score = completed stages / total stages * 100
```

### 6.2 Goal success

Goal success measures whether the final state satisfies the BDDL goal.

So:

- `stage` = process completion
- `goal` = final outcome completion

They can disagree.

## 7. Drawer task logic

Drawer-related tasks use region motion relative to the drawer's own initial position.
The main checks are:

- `_drawer_open_abs(...)`
- `_drawer_closed_abs(...)`

Current thresholds:

- `open`: absolute region y displacement `> 0.10`
- `close`: absolute region y displacement `< 0.08`

This avoids hard-coded global constants and instead uses the recorded initial site position when available.

## 8. Pour task logic

Pour stages do not directly inspect liquid.
They check whether the robot performs a sustained pouring motion.

Core function:

- `_pour_stage(...)`

Typical signals:

- tilt angle range
- minimum duration
- optional high-angle hold frames

So pour stage is mainly a process metric, while goal success is the final state metric.

## 9. Batch runner behavior

The batch runner is:

- `scripts/run_all_tasks1_26_until_stage_nonzero.py`

Its logic is:

1. iterate task2..26 in order
2. start from `seed=100`
3. if `stage=0`, try the next seed
4. stop the current task when `stage > 0` is first observed
5. record results into `attempts.tsv` and `task_summary.tsv`

## 10. What users need to change for their own model

Users usually only need to implement their own adapter.

### 10.1 Required change

Implement:

- `scripts/example_policy_adapter_template.py`

or provide another module that exposes:

```python
def build_adapter(**kwargs) -> BasePolicyAdapter:
    ...
```

### 10.2 Usually no need to change

In most cases users do **not** need to rewrite:

- stage definitions
- BDDL parsing
- goal success logic
- batch evaluation logic

## 11. Output files

Typical output artifacts are:

- rollout videos
- `attempts.tsv`
- `task_summary.tsv`
- optional JSONL attempt log from the batch runner

These files are the evaluation outputs. They are not part of the benchmark code itself.
