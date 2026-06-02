# Evaluate Your Model on RoboMemArena

This page explains how to plug your own policy into the RoboMemArena evaluation benchmark.
The benchmark code handles the RoboMemArena task definitions, BDDL goals, rollouts, videos,
and CSR/TSR metrics. Your model only needs to provide actions for the current observation and task prompt.

## Option 1: Generic Policy Adapter

Use this path if your model is a single policy that takes an observation and prompt, then outputs an action chunk.

Implement the adapter interface in:

```text
evaluation_benchmark/scripts/example_policy_adapter_template.py
```

The required method is:

```python
def infer_actions(self, obs, prompt, resize_size):
    """Return a numpy action chunk with shape [horizon, action_dim]."""
```

The benchmark will call your adapter during rollout. It will create the RoboMemArena/LIBERO environment,
load the task BDDL, reset the environment, pass the current observation and prompt to your adapter, execute
the returned actions, save videos, and compute CSR/TSR.

Run Task 1:

```bash
cd evaluation_benchmark
python scripts/eval_task1_only.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/your_checkpoint"}' \
  --video-out-path outputs/task1
```

Run one task from Task 2-26:

```bash
cd evaluation_benchmark
python scripts/eval_tasks2_26.py \
  --task-id 4 \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/your_checkpoint"}' \
  --video-out-path outputs/task4
```

Run the full Task 1-26 sweep wrapper:

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26_until_stage_nonzero.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/your_checkpoint"}' \
  --out-root outputs/your_model_eval_1_26
```

In this adapter path, Task 1 uses the Task 1 reference stage checks, and Task 2-26 uses the same reference stage/goal checker as the VLM/VLA reference path:

```text
evaluation_benchmark/scripts/task2_26_reference_stage.py
```

## Option 2: VLM/VLA Reference Evaluation

Use this path if your system follows the reference VLM planner + VLA policy-server structure.
The VLM5 Tasks 2-26 evaluation code currently lives in:

```text
evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference/
```

Task 1 is intentionally separated and should use:

```text
evaluation_benchmark/reference_evaluation/task1_nomap_reference/
```

OpenPI source interface:

```text
Default: use the bundled minimal runtime at third_party/openpi_minimal if available.
Optional: point OPENPI_ROOT to your own OpenPI checkout. No code changes are needed.
```

Required local inputs:

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional if using a bundled minimal OpenPI runtime
export OPENPI_INFERENCE_ROOT=/abs/path/to/openpi_inference
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm_task1
export VLA_CKPT=/abs/path/to/vla_alltask/params
export VLA_CONFIG=<your_vla_config_name>  # optional; default runner value is pi05_robomemarena
```

Run Tasks 2-26 with the reference runner:

```bash
cd evaluation_benchmark/reference_evaluation/tasks2_26_vlm5_reference
bash run_tasks2_26_vlm_vla_csr_tsr.sh
```

Run a subset:

```bash
export TASKS_JSON='[2,3,4]'
bash run_tasks2_26_vlm_vla_csr_tsr.sh
```

## Metrics

- `CSR`: average stage/process completion percentage.
- `TSR`: final BDDL goal success rate.

For official reporting, use one evaluation path consistently. The generic adapter path is intended for external
models. The VLM/VLA reference path is intended for reproducing or adapting the provided reference integration.
