# RoboMemArena Evaluation Benchmark

- Project: RoboMemArena evaluation benchmark
- Repo: RoboMemArena
- Type: evaluation package

## What this directory contains

This directory contains a model-agnostic evaluation benchmark for RoboMemArena task1 and task2..26.
It is meant for users who already trained their own policy and now want to evaluate it on the benchmark tasks.

This package is organized around a small adapter interface, so users can plug in their own model and evaluate it on the benchmark tasks.

It focuses on:

- evaluation entry scripts
- benchmark task definitions and prompts
- adapter-based policy integration
- VLM/VLA reference evaluation code

## Directory layout

```text
evaluation_benchmark/
  README.md
  docs/
    task_evaluation_code_guide.md
  scripts/
    policy_adapter.py
    example_policy_adapter_template.py
    eval_common.py
    eval_task1_only.py
    eval_tasks2_26.py
    run_all_tasks1_26_until_stage_nonzero.py
  reference_evaluation/
    README.md
    task1_nomap_reference/
      eval_task1_nomap_reference.py
    tasks2_26_vlm5_reference/
      eval_tasks2_26_vlm_vla.py
      run_tasks2_26_vlm_vla_csr_tsr.sh
      fullvlm_v2_26_memory_tasks.json
  async_vlm26_reference/
    README.md
    eval_fullvlm26_async_vlm_vla.py
    run_fullvlm26_async_vlm_vla_csr_tsr.sh
    fullvlm_v2_26_memory_tasks.json
  bddl/
  libero_fork/
```

## Quick start

1. Make sure your environment can import the local LIBERO fork and its dependencies.
   You typically need a working `mujoco + robosuite + OpenGL/EGL` environment before running actual evaluation.
2. Implement your own adapter by following `scripts/example_policy_adapter_template.py`.
3. Run single-task evaluation:

```bash
cd evaluation_benchmark
python scripts/eval_task1_only.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/ckpt"}' \
  --video-out-path outputs/task1
```

By default, `task1` now matches the legacy evaluation path used in our old runs:

- environment render resolution: `640x480`
- policy resize target: `256`

4. Run task2..26 evaluation:

```bash
cd evaluation_benchmark
python scripts/eval_tasks2_26.py \
  --task-id 4 \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/ckpt"}' \
  --video-out-path outputs/task4
```

5. Run the batch sweep until each task from 2 to 26 first reaches `stage > 0`:

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26_until_stage_nonzero.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/ckpt"}' \
  --out-root outputs/tasks2_26_until_stage_nonzero
```

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
evaluation_benchmark/reference_evaluation/
```

That reference evaluation folder contains, in order, the Task 1 evaluation code and the VLM5 Tasks 2-26 evaluation code for the remaining 25 tasks.
Task 1 is intentionally separated from the Tasks 2-26 runner.
Async reference code is kept separately in `evaluation_benchmark/async_vlm26_reference/`.
Users may adjust prompts for their own models, but the rewritten prompts should remain semantically aligned with the official 26-task table and BDDL goals.
It uses the same metric names as our current reports:

- `CSR`: average stage/process completion percentage
- `TSR`: final BDDL goal success rate
