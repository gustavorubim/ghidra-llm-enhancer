# Autoresearch Loop - Instructions

This repository now uses an `autoresearch`-style GRPO loop.
The goal is not to pre-enumerate hypotheses. The goal is to let the agent:

- study the latest evidence
- invent the next experiment
- train and evaluate it
- keep winners
- discard losers
- continue indefinitely

The source of truth for that behavior is `research/program.md`.

## What changed

The old setup behaved like a config sweep.
The new setup behaves like `karpathy/autoresearch`:

- the agent creates hypotheses on its own
- the agent may edit a bounded set of real GRPO system files
- the first run establishes the baseline automatically
- the champion is the current `autoresearch/<tag>` branch head
- each experiment runs on a temporary candidate branch
- keep/discard decisions happen after training plus scout/confirm eval

## Before you start

Make sure all of these are true:

1. You are on a Windows CUDA machine that can run the GRPO path.
2. A working SFT checkpoint already exists.
3. `data/processed/rl/rl_records.jsonl` exists.
4. The project `.venv` is active.

Optional sanity checks:

```powershell
python -m decomp_clarifier.cli train-grpo --help
python -m decomp_clarifier.cli eval-grpo-checkpoint --help
```

## Starting the loop

Open the agent in this repository and point it at `research/program.md`.

Example kickoff prompt:

```text
Read research/program.md and start the GRPO autoresearch loop.
Do the setup if needed, establish the baseline automatically if none exists,
and then continue running experiments without asking for permission.
```

If your agent runtime supports sleeping or wakeups, let it sleep during long
training and evaluation runs instead of polling.

## Branch model

The loop uses two branch types:

- champion branch: `autoresearch/<tag>`
- temporary branch per experiment: `autoresearch-tmp/<tag>-<iteration>`

The champion branch is the current best system.
Each candidate branch contains one experimental idea.

When a candidate wins, the champion branch fast-forwards to it.
When a candidate loses, the temporary branch is deleted and the champion stays
where it is.

## Files to watch

These files matter during the loop:

| File | Purpose |
|------|---------|
| `research/program.md` | Agent constitution and experiment rules |
| `research/instructions.md` | Human quick-start and control notes |
| `research/experiment_log.jsonl` | Ledger of every baseline, keep, discard, and crash |
| `research/baseline.json` | Convenience mirror of current champion metrics |
| `artifacts/runs/train-grpo-*/model/grpo_training_manifest.json` | Training run outcome |
| `artifacts/runs/eval-grpo-checkpoint-*/checkpoint_eval_manifest.json` | Eval metrics |
| `artifacts/runs/eval-grpo-checkpoint-*/comparison.md` | Side-by-side summary |
| `artifacts/runs/eval-grpo-checkpoint-*/inspection_samples.*` | Concrete examples |

## What the agent is allowed to change

The loop may edit only this bounded experiment surface:

- `configs/training/grpo_qwen35_2b_12gb.yaml`
- `src/decomp_clarifier/training/grpo/rewards.py`
- `src/decomp_clarifier/training/grpo/train.py`
- `src/decomp_clarifier/training/grpo/data.py`
- `src/decomp_clarifier/dataset/prompt_formatter.py`
- `src/decomp_clarifier/dataset/packers.py`

This is the local equivalent of `train.py` in Karpathy's repo: small, real,
and sufficient to change GRPO behavior meaningfully.

## What the agent must not change

The loop must not edit:

- evaluation code and report code
- dataset artifacts
- environment or dependency setup
- SFT code outside the bounded RL prompt/data path
- hardware safety settings that would make experiments incomparable

## Stopping the loop

To stop cleanly, edit the top of `research/program.md` so the `## STOP`
section says `Status: stop` and includes a real stop reason.

Example:

```markdown
## STOP

Status: stop
Reason: pause experiments and review the current champion manually
```

The agent should stop at the start of the next iteration after logging its
state.

Closing the agent session also stops the loop.

## Steering the loop

Do not add giant search grids.
Instead, steer strategy by editing the high-level guidance in
`research/program.md`.

Good steering edits:

- "Prioritize reward-gate experiments before prompt-packing experiments."
- "Allow more aggressive prompt compaction this evening."
- "Focus on reducing compile regressions from invented helper calls."
- "Stop exploring readability-only wins unless compile and behavior stay flat."

Bad steering edits:

- long candidate tables for every scalar
- hand-authored fixed experiment order
- instructions that prevent the agent from inventing its own next step

## Reading results

Use these views:

- `research/experiment_log.jsonl`: one line per experiment
- `research/baseline.json`: current champion summary
- latest `comparison.md`: compact metric table
- latest `inspection_samples.md` or `.jsonl`: concrete wins and failures

The git branch also matters:

- `autoresearch/<tag>` is the winning line of descent
- temporary branches are disposable

## Throughput expectations

This repo is slower than Karpathy's 5-minute trainer.
On the current GRPO path, a full experiment is much more expensive.

Approximate cadence on the current 12 GB profile:

- training: about 60 minutes
- scout eval: about 10 minutes
- confirm eval: additional time only for promising candidates

So the loop is still autonomous, but it is lower-throughput and more
expensive per idea. That is why the constitution uses:

- a bounded edit surface
- evidence-driven hypotheses
- a scout/confirm eval flow
- fast discard of weak candidates

## Notes on legacy files

`research/baseline.json` and `research/experiment_log.jsonl` already exist in
the repo. The new loop still uses them, but only as ledgers.

- `baseline.json` is not the code source of truth
- the champion branch head is the true baseline
- the first run overwrites the old zero baseline automatically

## Reference

This setup is intentionally modeled after `karpathy/autoresearch`, but adapted
to this repository's GRPO pipeline and current CLI:

- upstream repo: https://github.com/karpathy/autoresearch
- upstream constitution: https://raw.githubusercontent.com/karpathy/autoresearch/master/program.md
