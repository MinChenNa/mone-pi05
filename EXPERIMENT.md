# MoNe × π0.5 mechanism-validation experiment

## Material Passport

- Material: `09_MoNe.pdf` (10 pages, supplied locally by the user)
- Backbone interface: Physical Intelligence `openpi`, revision `15a9616`
- Status: mechanism-validation prototype; not a reproduction of reported MoNe numbers
- Data exposure: local only

## Hypothesis

A query-conditioned, fixed-size fast-weight memory can carry long-horizon task
history into π0.5 without growing the token count used by each action query.

## Integration

`Pi05MoNeContextAdapter` consumes historical π0.5 prefix embeddings in fixed
segments, builds a constant-size fast-weight state, and generates memory tokens
from the current query embeddings. The tokens are prepended before π0.5 creates
its reusable prefix KV cache; the original π0.5 weights are untouched.

## What this test establishes

1. Segment-wise history writes change subsequent query reads.
2. Fast-weight state size is invariant to history length.
3. Query memory-token count is invariant to history length.
4. The augmented tensors satisfy π0.5's prefix embedding/mask contract.

## What remains for a model-quality claim

- Replace the stable delta rule with the paper's per-layer meta-trained SwiGLU
  update (learned rate, momentum/decay, local RoPE, and LoRA projections).
- Attach memory inside every Gemma attention layer, not only at the prefix API.
- Train the adapters on robot trajectories with long-horizon distractors.
- Evaluate checkpointed π0.5 on LIBERO/ALOHA long-horizon variants, reporting
  task success, action MSE, latency, peak memory, and history-length scaling.

## Real-checkpoint validation (2026-08-24)

Checkpoint: official `pi05_libero`, converted to PyTorch bfloat16. Hardware:
RTX 5070 Ti Laptop GPU, PyTorch 2.7.1 + CUDA 12.8.

- Real checkpoint interface smoke test: baseline and MoNe actions were finite.
- Fast-weight state: `[1, 4, 512, 512]`, 2 MiB after 968 history tokens.
- Fixed query read: 8 memory tokens.
- Peak allocated GPU memory: 7.68 GB.
- On three frames from official LIBERO episode 0 (frames 20/80/140), using
  matching seeds and two flow steps, mean 10-step action MSE changed from
  `0.0002510274` to `0.0002369458` (`-5.61%`). Every sampled frame improved.

This is an offline diagnostic on three correlated frames from a training
episode, not a held-out benchmark or evidence of higher rollout success. A
credible performance claim still requires multiple held-out episodes, standard
flow-step settings, trained memory projections/meta-parameters, and simulator
rollouts with confidence intervals.

## Multi-task go/no-go diagnostic

Six official LIBERO demonstration episodes from six task selections were used.
Each current observation was evaluated with the same noise under three
conditions: baseline pi0.5, relevant history, and history from another task.

### Two flow steps, 24 paired samples

- Relevant-history mean MSE: -3.70% vs baseline, but only 50% pairwise wins.
- Bootstrap 95% CI for absolute improvement crossed zero; one-sided Wilcoxon
  `p=0.322`.
- Irrelevant history produced a similar -3.10% mean change.
- Relevant history beat irrelevant history in only 45.8% of samples
  (`p=0.668` for selectivity).

### Ten flow steps, 12 paired samples

- Relevant history improved 10/12 samples, but mean MSE changed only -0.106%.
- Irrelevant history also changed mean MSE by -0.085%.
- Relevant-vs-irrelevant selectivity was not significant (`p=0.285`), and the
  two conditions' paired effects were highly correlated (`r≈0.84`).

### Decision

The adapter has a reproducible effect on pi0.5 outputs, but the current
untrained zero-W0/identity-projection implementation does not demonstrate that
it uses the *content* of relevant history. Do not invest in full simulator
rollouts for this adapter as-is. Continue only as a bounded training experiment:
learn the memory projections/gate (and preferably the MoNe meta-update) and
require relevant-history selectivity on a held-out offline gate before running
expensive closed-loop evaluation.
# Bounded adapter-training decision experiment (2026-08-25)

The official `pi05_libero` backbone was frozen and only a 256-dimensional
MoNe-style delta-memory adapter (including learned W0, Q/K/V/output projections,
and an injection gate) was trained.  Sixteen proxy pairs from episodes
0, 1, 2, and 10 were used for 200 contrastive steps.  Episodes 20 and 30 were
held out from adapter training and evaluated with four frames each and 10 flow
steps.

## Results

| Test | Relevant-history result vs pi0.5 | Relevant vs irrelevant |
|---|---:|---:|
| Contrastive training proxy | 50% pair accuracy; loss 0.7707 -> 0.8118 | no learned separation |
| Trained adapter, learned gate logit -2.203 | MSE +2965%; 0/8 wins | 37.5% wins; p=0.527 |
| Weak injection, gate logit -6 | MSE +3153%; 0/8 wins | 25.0% wins; p=0.902 |
| Near-zero injection, gate logit -100 | MSE -0.434%; 2/8 wins; CI crosses zero | histories identical by construction |

The near-zero control stays close to the original backbone, so merely adding
eight prefix positions is not the main failure.  Nonzero learned memory tokens
destabilize the action head even at a weak gate, while the proxy objective does
not learn history selectivity.  The preregistered continuation criteria
(>=70% wins, >=3% MSE improvement, positive bootstrap CI, and relevant history
better than irrelevant history) are all missed.

## Decision

Stop scaling this adapter design.  Further work should be treated as a redesign:
normalize injected token energy to the native pi0.5 prefix distribution, train
with frozen-backbone action distillation/behavior-cloning loss rather than a
cosine proxy, and use an insertion mechanism that preserves the pretrained
prefix interface.  No real-robot or LIBERO simulator success claim is supported
by these offline action-MSE diagnostics.
