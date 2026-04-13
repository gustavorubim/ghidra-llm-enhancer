# GRPO Autoresearch — Research Constitution

This file is the source of truth for the autoresearch loop. Edit it to steer
the agent. The agent reads it at the start of every iteration.

---

## Goal

Maximize the composite eval score on the `val` split (50 samples) after each
GRPO training run:

```
score = json_valid_rate * 0.5 + readability_score * 0.3 + compile_success_rate * 0.2
```

Secondary objectives (break ties, used for logging only):
- `behavior_success_rate` should not regress
- `naming_score` should not regress

---

## Ratchet rule

A change is **KEPT** if:
- `score >= baseline_score + 0.02`

Otherwise it is **REVERTED** and the config is restored exactly to the
baseline snapshot in `research/baseline.json`.

If `research/baseline.json` has `score: 0.0`, the first successful run that
produces any non-zero json_valid_rate establishes the baseline automatically.

---

## Editable files

The agent may only modify these files:

| File | What can change |
|------|----------------|
| `configs/training/grpo_qwen35_2b_12gb.yaml` | Any key listed in the search space below |

No other files may be modified during the loop. The agent must not touch
dataset files, SFT configs, Python source, or any key marked off-limits.

---

## Search space

Change **one variable per experiment**. Do not combine changes.

### `training.learning_rate`
Candidates: `1.0e-6`, `2.0e-6` (current), `5.0e-6`, `1.0e-5`

### `training.max_completion_length`
Candidates: `384` (current), `512`, `640`
Note: if you increase this, verify `max_prompt_length + max_completion_length <= max_seq_length`.

### `training.max_prompt_length`
Candidates: `768`, `896` (current), `1024`
Note: same constraint as above.

### `training.generations_per_prompt`
Candidates: `2`, `4` (current), `6`
Note: increasing this raises VRAM. Do not exceed `6` on 12 GB.

### `training.warmup_ratio`
Candidates: `0.05`, `0.1` (current), `0.2`

### `training.behavior_similarity_threshold`
Candidates: `0.2`, `0.35` (current), `0.5`

### `training.max_grad_norm`
Candidates: `0.05`, `0.1` (current), `0.2`

### `training.reward_weights.compile`
Candidates: `2.0`, `3.0` (current), `4.0`, `5.0`

### `training.reward_weights.behavior`
Candidates: `1.0`, `2.0` (current), `3.0`

### `training.reward_weights.format`
Candidates: `0.5`, `1.0` (current), `2.0`

### `training.reward_weights.cleanup`
Candidates: `1.0`, `1.5` (current), `2.0`

### `training.reward_weights.hallucination_penalty`
Candidates: `1.0`, `2.0` (current), `3.0`

### `training.reward_weights.readability`
Candidates: `0.5`, `1.0` (current), `2.0`

---

## Off limits — never change these

```
model.sft_checkpoint_dir      # SFT checkpoint is fixed
model.base_model_id           # base model is fixed
model.loader_variant          # Unsloth only
training.max_seq_length       # memory constraint
training.max_steps            # keeps experiments comparable
training.batch_size           # memory constraint
training.grad_accum_steps     # memory constraint
training.lora_rank            # architecture decision
training.load_in_4bit         # memory constraint
training.optim                # adamw_8bit is fixed
training.save_steps           # always equal to max_steps
training.precision            # bf16 only
training.epochs               # always 1
```

---

## Experiment protocol — follow exactly

### Step 1 — Read history

Read `research/experiment_log.jsonl`. Each line is a completed experiment.

- Do not retry any (`variable`, `value`) pair that already appears in the log
  with result `REVERTED` — it already failed.
- Do not retry any pair that appears with result `KEPT` — it is already the
  baseline.
- Prefer variables that have not been explored yet over ones with partial data.
- When choosing a value for an unexplored variable, start with the candidate
  that is most different from the current config value (not an adjacent step).

### Step 2 — Propose one change

State your hypothesis clearly before making any file edits. Example:
> "Hypothesis: increasing compile reward weight from 3.0 to 4.0 will improve
> compile_success_rate and therefore the composite score."

Read the current value from the config file. Apply exactly one change.
Record the old value so you can revert precisely.

### Step 3 — Run training

```
decomp-clarifier train-grpo --profile grpo_qwen35_2b_12gb
```

Training takes approximately 60 minutes. After starting the command,
**use ScheduleWakeup with delaySeconds=3900** and resume on wake.

Do not poll. Do not check intermediate state. Wake once and proceed.

### Step 4 — Run evaluation

After waking, confirm training completed by checking for the latest
`artifacts/runs/train-grpo-*/model/grpo_training_manifest.json`.

Then run:
```
decomp-clarifier eval-grpo-checkpoint --split val --sample-limit 50
```

Evaluation takes approximately 10 minutes. After starting,
**use ScheduleWakeup with delaySeconds=700** and resume on wake.

### Step 5 — Read results

Find the latest manifest:
```
artifacts/runs/eval-grpo-checkpoint-*/checkpoint_eval_manifest.json
```
(sort by directory name descending — timestamp is embedded in the name)

Extract:
- `metrics.json_valid_rate`
- `metrics.readability_score`
- `metrics.compile_success_rate`
- `metrics.behavior_success_rate`
- `metrics.naming_score`

Compute: `score = json_valid_rate * 0.5 + readability_score * 0.3 + compile_success_rate * 0.2`

### Step 6 — Keep or revert

Compare `score` to `baseline_score` in `research/baseline.json`.

**If KEPT** (`score >= baseline_score + 0.02`):
1. Update `research/baseline.json` with the new score and a snapshot of all
   changed config keys.
2. `git add configs/training/grpo_qwen35_2b_12gb.yaml research/baseline.json`
3. `git commit -m "research: keep <variable>=<value> score=<score:.3f>"`

**If REVERTED** (`score < baseline_score + 0.02`):
1. Restore the config: `git checkout configs/training/grpo_qwen35_2b_12gb.yaml`
2. Do not update `baseline.json`.

### Step 7 — Log the experiment

Append one JSON line to `research/experiment_log.jsonl`:

```json
{
  "iteration": <N>,
  "variable": "<yaml.key.path>",
  "old_value": <old>,
  "new_value": <new>,
  "hypothesis": "<one sentence>",
  "score": <float>,
  "baseline_score": <float>,
  "json_valid_rate": <float>,
  "readability_score": <float>,
  "compile_success_rate": <float>,
  "behavior_success_rate": <float>,
  "naming_score": <float>,
  "result": "KEPT" | "REVERTED",
  "train_run_id": "<run_id>",
  "eval_run_id": "<run_id>",
  "timestamp": "<ISO8601>"
}
```

### Step 8 — Continue immediately

After logging, proceed directly to the next iteration (Step 1).
Use ScheduleWakeup with delaySeconds=60 to resume if the loop context needs
a clean tick.

---

## NEVER STOP

Do not pause to ask for permission between iterations.
Do not stop if a run produces a lower score — log it as REVERTED and continue.
The only reasons to stop are:
- An unrecoverable error that requires human inspection (OOM crash, corrupt
  checkpoint, missing CLI command)
- The human edits this file and adds a `## STOP` section at the top

---

## Current research focus

Start by exploring reward weights (compile, behavior, format) since those are
the most direct levers on the metrics we are optimizing. Move to
hyperparameters (learning_rate, warmup_ratio) once reward weights stabilize.
