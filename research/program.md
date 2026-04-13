# GRPO Autoresearch - Research Constitution

This file is the source of truth for the autonomous GRPO research loop.
The agent reads it at the start of every iteration and follows it exactly.

## STOP

The loop only stops when this section is explicitly set to `Status: stop`.
The human controls the loop by editing this file.

Status: continue
Reason: active

## Mission

Continuously improve the post-SFT GRPO system for binary-grounded decompiler
clarification. The loop must run autonomously:

- inspect prior evidence
- invent the next hypothesis
- implement the experiment
- run training and evaluation
- keep or discard the change
- log the result
- continue immediately

This is adapted from `karpathy/autoresearch`, but applied to this repository's
GRPO stage instead of a single-file trainer.

## Core idea

This is not a fixed search-space sweep.

The agent is expected to generate hypotheses from evidence, including:

- reward telemetry under `artifacts/runs/train-grpo-*/model/logs/`
- GRPO training manifests
- checkpoint eval manifests
- comparison reports
- inspection samples
- the history in `research/experiment_log.jsonl`
- the current champion state on the autoresearch branch

The human edits this file to steer strategy, not to enumerate every candidate.

## Platform and prerequisites

Before the loop begins, all of these must be true:

1. The machine is Windows with NVIDIA CUDA available for training.
2. A usable SFT checkpoint already exists and `train-grpo` resolves it.
3. The RL dataset exists at `data/processed/rl/rl_records.jsonl`.
4. The commands below work in the active project `.venv`:

```powershell
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_2b_12gb
python -m decomp_clarifier.cli eval-grpo-checkpoint --training-profile grpo_qwen35_2b_12gb --split val --sample-limit 1
```

If any prerequisite is missing, stop and report the blocker.

## Branch model

The loop runs on a dedicated champion branch:

```text
autoresearch/<tag>
```

The current `HEAD` of that branch is the champion.

Each experiment runs on a temporary candidate branch created from the champion:

```text
autoresearch-tmp/<tag>-<iteration>
```

This avoids destructive resets. The loop should not rely on `git reset --hard`
to discard failed ideas.

## Ground truth and read-only surfaces

Do not modify the evaluation harness or any code that defines the metric being
used for comparison. These files are read-only during the loop:

- `src/decomp_clarifier/evaluation/`
- `src/decomp_clarifier/evaluation/checkpoint_eval.py`
- `src/decomp_clarifier/evaluation/report_builder.py`
- `src/decomp_clarifier/evaluation/metrics.py`
- `src/decomp_clarifier/evaluation/compile_eval.py`
- `src/decomp_clarifier/evaluation/behavior_eval.py`
- dataset artifacts under `data/`
- SFT training code outside the GRPO-related prompt/data packing path

The loop may inspect these files for context, but must not edit them.

## Editable experiment surface

The loop may modify only the bounded GRPO experiment surface:

- `configs/training/grpo_qwen35_2b_12gb.yaml`
- `src/decomp_clarifier/training/grpo/rewards.py`
- `src/decomp_clarifier/training/grpo/train.py`
- `src/decomp_clarifier/training/grpo/data.py`
- `src/decomp_clarifier/dataset/prompt_formatter.py`
- `src/decomp_clarifier/dataset/packers.py`

The goal is to let the agent change the real GRPO system, not only a static
config, while keeping the research surface small and auditable.

## Off limits

Do not change these unless the human edits this file and explicitly expands
scope:

- `model.base_model_id`
- `model.loader_variant`
- `training.precision`
- `training.load_in_4bit`
- `training.max_seq_length`
- `training.batch_size`
- `training.grad_accum_steps`
- `training.max_steps`
- `training.save_steps`
- `training.epochs`
- dependencies, package versions, or environment setup
- dataset generation, dataset splits, or evaluation prompts outside the bounded
  experiment surface

The current repo does not have a fixed wall-clock scout trainer. Therefore
`training.max_steps` remains fixed so experiments stay comparable.

## What hypotheses are allowed

Any small, coherent idea that could improve GRPO is allowed, including:

- reward weighting changes
- new reward terms or penalties
- stricter or softer reward gates
- prompt packing changes for RL
- changes to what binary-grounded evidence is presented in the RL prompt
- GRPO rollout configuration within the allowed profile
- cleanup, readability, naming, hallucination, and signature shaping logic

Do not enumerate a finite search grid. Invent the next best experiment from the
evidence you have.

## Simplicity criterion

All else equal, prefer simpler code and smaller diffs.

Keep a change only if the measured improvement justifies the extra complexity.
A tiny score gain with ugly or fragile logic is not a win.
A simplification with equal or slightly better metrics is a win.

## Experiment metrics

The loop uses one scalar score plus hard keep gates.

Scout and confirm score:

```text
score =
  0.30 * behavior_success_rate +
  0.25 * compile_success_rate +
  0.20 * json_valid_rate +
  0.15 * readability_score +
  0.10 * naming_score
```

Hard keep gates:

- `compile_success_rate` must not drop by more than `0.01` vs champion
- `behavior_success_rate` must not drop by more than `0.01` vs champion
- `json_valid_rate` must not drop by more than `0.02` vs champion

Use `field_complete_rate` and inspection samples as diagnostics, not as primary
keep criteria.

## Baseline and champion state

There is no manual baseline bootstrapping.

The first completed run on a fresh autoresearch branch is always the baseline.
After that:

- champion state = `HEAD` of `autoresearch/<tag>`
- `research/baseline.json` is a convenience mirror of the current champion
- `research/experiment_log.jsonl` is the run ledger

`baseline.json` is not the code source of truth. Git branch state is.

Do not include `research/baseline.json` or `research/experiment_log.jsonl` in
candidate experiment commits. They are control-plane logs, not part of the
experiment patch.

## Loop protocol

### Step 0 - Setup on a fresh run

If the champion branch does not exist:

1. Propose a tag based on the date, for example `apr13-grpo`.
2. Create `autoresearch/<tag>` from the current branch.
3. Ensure `research/experiment_log.jsonl` exists.
4. Run one baseline experiment with no code changes.
5. Record that run as `status: "baseline"` and mirror it into
   `research/baseline.json`.

If the champion branch already exists, resume from it.

### Step 1 - Read evidence

At the start of every iteration, inspect:

- the latest entries in `research/experiment_log.jsonl`
- the latest champion metrics in `research/baseline.json`
- the latest train/eval manifests
- reward component telemetry for the latest kept and latest discarded runs
- comparison and inspection reports if the last run regressed

The next experiment must come from this evidence, not from a fixed candidate
table.

### Step 2 - Form one hypothesis

Write a one-sentence hypothesis before editing.

A single experiment may touch multiple files, but it must represent one
coherent idea. Example categories:

- "Make broken outputs score clearly negative instead of merely discounted."
- "Pass more binary-grounded call evidence into the RL prompt."
- "Reduce reward weight on cleanup and increase signature fidelity."

### Step 3 - Create a candidate branch

Create a new temporary candidate branch from the champion branch:

```powershell
git checkout autoresearch/<tag>
git checkout -b autoresearch-tmp/<tag>-<iteration>
```

All experiment code changes happen on this candidate branch.

### Step 4 - Implement the experiment

Apply the minimal code/config diff needed for the hypothesis.

Before training:

1. Review the diff.
2. Record the touched files.
3. Commit the candidate change with a short message:

```text
research: try <short description>
```

### Step 5 - Run training

Run the full GRPO training experiment:

```powershell
python -m decomp_clarifier.cli train-grpo --training-profile grpo_qwen35_2b_12gb
```

Training typically takes about 60 minutes.
After starting it, sleep instead of polling when the agent runtime supports
that.

If training does not produce a new
`artifacts/runs/train-grpo-*/model/grpo_training_manifest.json`, treat the run
as a crash.

### Step 6 - Run scout evaluation

Run a cheap scout eval first:

```powershell
python -m decomp_clarifier.cli eval-grpo-checkpoint --training-profile grpo_qwen35_2b_12gb --split val --sample-limit 50
```

Read the latest
`artifacts/runs/eval-grpo-checkpoint-*/checkpoint_eval_manifest.json` and
extract:

- `json_valid_rate`
- `readability_score`
- `compile_success_rate`
- `behavior_success_rate`
- `naming_score`
- `field_complete_rate`

Compute the scout score using the formula above.

### Step 7 - Decide whether to confirm

Let `champion_score` be the score stored in `research/baseline.json`.

Discard immediately if any hard keep gate fails against the champion.

Otherwise:

- if this is the first baseline run, continue to confirm
- if `scout_score < champion_score + 0.005`, discard without confirm
- if `scout_score >= champion_score + 0.005`, run confirm eval

### Step 8 - Run confirm evaluation

Confirm with the full validation split:

```powershell
python -m decomp_clarifier.cli eval-grpo-checkpoint --training-profile grpo_qwen35_2b_12gb --split val
```

Read the latest confirm manifest and compute `confirm_score` with the same
formula and hard gates.

### Step 9 - Keep or discard

Keep the candidate only if:

- this is the baseline run, or
- all hard keep gates pass on confirm, and
- `confirm_score >= champion_score + 0.01`

If kept:

1. Update `research/baseline.json` to mirror the new champion metrics and
   config summary.
2. Append a ledger entry with `status: "keep"`.
3. Checkout the champion branch.
4. Fast-forward it to the candidate commit:

```powershell
git checkout autoresearch/<tag>
git merge --ff-only autoresearch-tmp/<tag>-<iteration>
```

5. Delete the temporary candidate branch.

If discarded:

1. Append a ledger entry with `status: "discard"` or `status: "crash"`.
2. Checkout the champion branch.
3. Delete the temporary candidate branch.
4. Leave the champion branch unchanged.

### Step 10 - Continue immediately

Start the next iteration from the champion branch without asking for
permission.

## Crash policy

If the experiment crashes:

- inspect the failure artifact or tail of the log
- if it is a trivial implementation mistake in the current candidate, fix it
  and retry once on the same candidate branch
- if the idea itself appears broken, log `status: "crash"`, discard it, and
  move on

Do not get stuck repairing one bad idea for many retries.

## Ledger format

Append one JSON object per experiment to `research/experiment_log.jsonl`.
Each line should include at least:

```json
{
  "iteration": 0,
  "timestamp": "ISO8601",
  "tag": "apr13-grpo",
  "champion_branch": "autoresearch/apr13-grpo",
  "candidate_branch": "autoresearch-tmp/apr13-grpo-0001",
  "parent_commit": "abc1234",
  "candidate_commit": "def5678",
  "status": "baseline|keep|discard|crash",
  "hypothesis": "one sentence",
  "files_touched": ["path1", "path2"],
  "train_run_id": "train-grpo-...",
  "scout_eval_run_id": "eval-grpo-checkpoint-...",
  "confirm_eval_run_id": "eval-grpo-checkpoint-... or null",
  "scout_score": 0.0,
  "confirm_score": 0.0,
  "metrics": {
    "json_valid_rate": 0.0,
    "field_complete_rate": 0.0,
    "readability_score": 0.0,
    "compile_success_rate": 0.0,
    "behavior_success_rate": 0.0,
    "naming_score": 0.0
  },
  "notes": "short summary of why it was kept or discarded"
}
```

## Current focus

Prioritize experiments in this order unless the evidence strongly suggests
otherwise:

1. reward logic and gates in `rewards.py`
2. reward weights and rollout settings in the GRPO training profile
3. RL prompt evidence packing and prompt compactness
4. data passed from packed RL records into the reward function

Avoid broad refactors. Small, evidence-driven experiments compound fastest.

## NEVER STOP

Once setup is complete, do not pause to ask whether to continue.
Do not stop because an experiment lost.
Do not wait for the human between iterations.

Only stop when:

- this file sets `Status: stop` under `## STOP`
- a prerequisite is missing
- the environment is corrupted in a way the agent cannot recover from
