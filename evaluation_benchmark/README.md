# RoboMemArena Evaluation Benchmark

- Project: RoboMemArena evaluation benchmark
- Repo: RoboMemArena
- Type: evaluation package

## What this directory contains

This directory contains a model-agnostic evaluation benchmark for RoboMemArena tasks 1-26.
It is meant for users who already trained their own policy and now want to evaluate it on the complete benchmark.

This package is organized around a small adapter interface, so users can plug in their own model and evaluate it on the 1-26 task setting without rewriting the RoboMemArena environment, BDDL loading, rollout loop, video recording, or metric aggregation.

It focuses on:

- evaluation entry scripts
- benchmark task definitions and prompts
- adapter-based policy integration
- VLM/VLA reference evaluation code
- official CSR/TSR reporting for the 1-26 task benchmark

## Directory layout

```text
evaluation_benchmark/
  README.md
  docs/
    evaluate_your_model.md
    task_evaluation_code_guide.md
  scripts/
    policy_adapter.py
    example_policy_adapter_template.py
    eval_common.py
    run_all_tasks1_26_until_stage_nonzero.py
  reference_evaluation/
    README.md
  async_vlm26_reference/
    README.md
    eval_fullvlm26_async_vlm_vla.py
    run_fullvlm26_async_vlm_vla_csr_tsr.sh
    fullvlm_v2_26_memory_tasks.json
  bddl/
  libero_fork/
```

Some source files keep historical task-specific names for compatibility, but the public benchmark setting is the full 1-26 task evaluation.

## Quick start

For a focused guide on plugging in your own checkpoint or policy, see [Evaluate Your Model on RoboMemArena](docs/evaluate_your_model.md).

1. Make sure your environment can import the local LIBERO fork and its dependencies.
   You typically need a working `mujoco + robosuite + OpenGL/EGL` environment before running actual evaluation.
2. Implement your own adapter by following `scripts/example_policy_adapter_template.py`.
3. Run the 1-26 task sweep:

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26_until_stage_nonzero.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/ckpt"}' \
  --out-root outputs/tasks1_26_until_stage_nonzero
```

The sweep uses the benchmark reference stage/goal checkers, so external model evaluation follows the same 1-26 scoring setting as the reference evaluation path.

## Adapter contract

Your adapter must return a numpy array with shape `[horizon, action_dim]`.
Each row is one action to send to the environment. The benchmark code will reuse up to `replan_steps` actions before querying the adapter again.

Required methods:

- `reset()`
- `infer_actions(obs, prompt, resize_size)`

See `scripts/policy_adapter.py` and `scripts/example_policy_adapter_template.py`.

## 26-task reference evaluation

If you want to inspect or reuse our VLM/VLA reference evaluation logic, see:

```text
evaluation_benchmark/async_vlm26_reference/
evaluation_benchmark/reference_evaluation/
```

Use the 26-task reference runner for full benchmark reporting. Internal helper files may keep task-specific names for compatibility with earlier experiments, but users should treat this as one 1-26 benchmark setting.

Metric names:

- `CSR`: final BDDL goal success rate
- `TSR`: stage/process completion score
