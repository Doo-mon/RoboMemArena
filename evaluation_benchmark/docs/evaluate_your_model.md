# Evaluate Your Model on RoboMemArena

This page explains the single integration interface for evaluating your model on the RoboMemArena 1-26 task benchmark.

The benchmark code handles the evaluation loop:

1. create the RoboMemArena/LIBERO environment
2. load the task BDDL and task prompt
3. reset the environment
4. collect the current observation
5. pass `obs + prompt` to your adapter
6. execute the returned action chunk
7. record videos and compute CSR/TSR

Your model only needs to provide an adapter that returns actions.

## Adapter Interface

Implement the adapter interface in:

```text
evaluation_benchmark/scripts/example_policy_adapter_template.py
```

The required method is:

```python
def infer_actions(self, obs, prompt, resize_size):
    """Return a numpy action chunk with shape [horizon, action_dim]."""
```

The adapter can contain any model architecture. For example:

- a single policy that directly maps `obs + prompt` to an action chunk
- a VLM planner that chooses a subtask, followed by a VLA policy that outputs actions
- a policy server client that forwards `obs + prompt` to a remote model
- custom preprocessing for images, states, prompts, or action dimensions

All of them use the same external contract: `obs + prompt -> action chunk`.

## Run the 1-26 Evaluation

```bash
cd evaluation_benchmark
python scripts/run_all_tasks1_26_until_stage_nonzero.py \
  --adapter-spec /abs/path/to/your_adapter.py:build_adapter \
  --adapter-kwargs '{"checkpoint_dir": "/abs/path/to/your_checkpoint"}' \
  --out-root outputs/your_model_eval_1_26
```

`--adapter-spec` points to a Python file and factory function:

```text
/abs/path/to/your_adapter.py:build_adapter
```

The factory should return an object with `reset()` and `infer_actions(obs, prompt, resize_size)`.

## VLM/VLA Models

A VLM/VLA system should still be exposed through the same adapter interface. Inside the adapter, you can run the VLM planner, maintain a subtask buffer, call a VLA policy server, convert observations, and return the final action chunk.

For a reference implementation of that kind of stack, inspect:

```text
evaluation_benchmark/async_vlm26_reference/
evaluation_benchmark/reference_evaluation/
```

Those folders show one way to connect a VLM planner and VLA policy to RoboMemArena. They are examples of the same benchmark setting, not a separate scoring definition.

Useful environment variables for the reference VLM/VLA stack include:

```bash
export OPENPI_ROOT=/abs/path/to/openpi  # optional if using a bundled minimal OpenPI runtime
export OPENPI_INFERENCE_ROOT=/abs/path/to/openpi_inference
export TARGET_LIBERO_PATH=/abs/path/to/LIBERO/libero
export VLM_CKPT=/abs/path/to/vlm_checkpoint
export VLA_CKPT=/abs/path/to/vla_checkpoint
export VLA_CONFIG=<your_vla_config_name>  # optional; default runner value is pi05_robomemarena
```

## Metrics

- `CSR`: final BDDL goal success rate.
- `TSR`: stage/process completion score.

For official reporting, use the same adapter and scoring path consistently across all 1-26 tasks.
