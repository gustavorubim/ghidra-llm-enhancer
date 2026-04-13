# Autoresearch Loop — Instructions

The research loop runs GRPO experiments autonomously, keeping only changes
that improve the composite eval score. It follows the ratchet pattern from
karpathy/autoresearch adapted for this project's training cycle.

---

## Prerequisites

Before starting the loop, all three conditions must be true:

1. **SFT checkpoint is working.** A recent SFT run exists at
   `artifacts/runs/train-sft-*/model/` with `json_valid_rate > 0.7` on eval.

2. **GRPO code is wired.** `configs/training/grpo_qwen35_2b_12gb.yaml` has
   `model.sft_checkpoint_dir` pointing to the SFT checkpoint directory.

3. **Baseline is populated.** Run one clean GRPO train + eval manually,
   then update `research/baseline.json` with the real score and run IDs.
   See "Initializing the baseline" below.

---

## Initializing the baseline

Run once manually before starting the loop:

```bash
# 1. Train
decomp-clarifier train-grpo --profile grpo_qwen35_2b_12gb

# 2. Eval
decomp-clarifier eval-grpo-checkpoint --split val --sample-limit 50

# 3. Read the manifest
#    artifacts/runs/eval-grpo-checkpoint-<latest>/checkpoint_eval_manifest.json
#    Compute: score = json_valid_rate*0.5 + readability_score*0.3 + compile_success_rate*0.2
```

Then edit `research/baseline.json`:
- Set `score` to the computed value
- Set `json_valid_rate`, `readability_score`, `compile_success_rate` from the manifest
- Set `train_run_id` and `eval_run_id` to the run IDs
- Set `timestamp` to now (ISO 8601)
- Remove the `note` field

The loop will not keep any experiment that does not beat this number by 0.02.

---

## Starting the loop

Open a Claude Code session in this repository and run:

```
/loop Follow the research protocol in research/program.md to run the next
iteration of the GRPO autoresearch loop. Self-pace using ScheduleWakeup:
3900s during training, 700s during eval, 60s between iterations.
```

The agent will:
1. Read `research/program.md` for the protocol
2. Read `research/experiment_log.jsonl` for history
3. Propose and apply one config change
4. Run training (~60 min), sleep via ScheduleWakeup
5. Run eval (~10 min), sleep via ScheduleWakeup
6. Keep or revert the change
7. Log to `experiment_log.jsonl`
8. Repeat from step 1

---

## Steering the loop

Edit `research/program.md` between iterations to change direction. Changes
are picked up at the next iteration start (step 1 above). Common edits:

**Focus on a specific variable:**
Add a note to the "Current research focus" section at the bottom of
`program.md`. Example: "Prioritize exploring learning_rate values next."

**Add a new candidate value:**
Add it to the relevant candidates list in the Search space section.

**Remove a candidate from consideration:**
Delete it from the list. Candidates already logged in experiment_log.jsonl
as REVERTED are automatically skipped regardless.

**Change the metric:**
Edit the score formula in the Goal section. This takes effect on the next
eval comparison — it does not retroactively reweight past experiments.

**Change the ratchet threshold:**
Edit the `0.02` in the Ratchet rule section.

---

## Stopping the loop

Add a `## STOP` section at the very top of `research/program.md`:

```markdown
## STOP

Reason: <your reason here>
```

The agent checks for this at the start of every iteration and halts cleanly
after logging the current state.

Alternatively, close the Claude Code session. The loop does not run
independently — it only executes when the agent is active.

---

## Reading results

**Quick summary:** read `research/experiment_log.jsonl`. Each line is one
experiment. `result: KEPT` means the config improved; `result: REVERTED`
means it did not.

**Best config so far:** `research/baseline.json` always reflects the
current best. The `config_snapshot` inside it shows what values produced
that score.

**Detailed eval reports:** each `eval-grpo-checkpoint` run writes an HTML
report to `artifacts/runs/eval-grpo-checkpoint-*/reports/`.

**Git history:** every KEPT experiment is a commit with the message
`research: keep <variable>=<value> score=<X.XXX>`. Run `git log --oneline`
to see the improvement trajectory.

---

## Expected throughput

| GPU | Time per experiment | Experiments per night (8 h) |
|-----|--------------------|-----------------------------|
| 12 GB (current) | ~70 min | ~6 |
| 24 GB | ~45 min | ~10 |

The search space has roughly 40 candidate (variable, value) pairs. A full
sweep takes 5–7 nights. In practice the loop will find improvements early
and converge before exhausting all candidates.

---

## File reference

| File | Written by | Purpose |
|------|-----------|---------|
| `research/program.md` | Human | Research constitution — steer here |
| `research/baseline.json` | Human + agent | Current best score and config snapshot |
| `research/experiment_log.jsonl` | Agent | One line per completed experiment |
| `research/instructions.md` | Human | This file |
| `configs/training/grpo_qwen35_2b_12gb.yaml` | Agent (reverted if no improvement) | Active GRPO config |
