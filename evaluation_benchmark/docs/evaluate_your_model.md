# Evaluate Your Model on RoboMemArena

This page explains how to plug your own policy into the RoboMemArena 1-26 task evaluation benchmark.
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

Run the full 1-26 task sweep:

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26_until_stage_nonzero.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/your_checkpoint"}' \
  --out-root outputs/your_model_eval_1_26
```

The adapter sweep uses the benchmark reference stage/goal checkers, so the scoring path stays aligned with the 1-26 reference evaluation.

## Option 2: VLM/VLA Reference Evaluation

Use this path if your system follows the reference VLM planner + VLA policy-server structure.
For the full 1-26 task reference integration, use:

```text
evaluation_benchmark/async_vlm26_reference/
```

The source tree may keep task-specific helper files internally for compatibility with earlier experiments, but the user-facing evaluation setting is the complete 1-26 benchmark.

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
export VLM_CKPT=/abs/path/to/vlm_checkpoint
export VLA_CKPT=/abs/path/to/vla_checkpoint
export VLA_CONFIG=<your_vla_config_name>  # optional; default runner value is pi05_robomemarena
```

Run the 1-26 reference runner:

```bash
cd evaluation_benchmark/async_vlm26_reference
export TASKS_JSON='[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26]'
export NUM_TRIALS=1
bash run_fullvlm26_async_vlm_vla_csr_tsr.sh
```

## Metrics

- `CSR`: final BDDL goal success rate.
- `TSR`: stage/process completion score.

For official reporting, use one evaluation path consistently. The generic adapter path is intended for external
models. The VLM/VLA reference path is intended for reproducing or adapting the provided reference integration.
