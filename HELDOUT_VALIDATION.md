# Independent-episode validation

The four 100-step adapters were trained on episodes 0, 1, 2, 10. This validation
excludes those episodes from both evaluation samples and donor histories.
It does not establish independence from the original pi0.5 pretraining or from
the dataset-level normalization statistics.

## Run on the offline GPU node

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/run_mone_heldout.sh
```

Runs layers 4 and 9 sequentially on the first visible allocated GPU, preserving
CUDA_VISIBLE_DEVICES. Does not train. Each layer uses 8 tasks × 2 episodes ×
3 frames × 3 fixed paired noise/time draws. Report paths and log paths are printed
at launch. The shell returns nonzero on failure. Output JSON is only written when
the complete run finishes; JSONL records are flushed during evaluation.

The same current observation, action target, noise and time are used for all
conditions: baseline, relevant history, different-task history, same-task other
episode history, empty state, old circular roll, and reversed valid-token order.
Wrong-history donors are separate from the evaluation episodes, and selected at
similar relative progress. The current task prompt is kept identical in all history
conditions, so different-task results test visual/state content rather than a
changed prompt. Correct history is the frame 10 steps before the current frame,
matching the original training protocol. This is one historical frame, not a
ten-frame temporal sequence.

Reports include full 32-dimensional flow loss and the first 7 real action dimensions
separately. Noise/time draws and frames are first averaged within an episode.
Confidence intervals use 5000 paired episode bootstrap samples, with win fractions
and per-episode means. These intervals are descriptive: episodes within a task can
remain correlated, there is one adapter-training seed, and layers were selected
on training results. Flow loss is not task success rate.

Evidence to look for: relevant history reduces held-out loss relative to baseline
and both wrong-history controls, preferably in both the full and real-action
metrics. A training loss reduction alone does not establish this. If controls
improve equally, the adapter may be providing a generic correction.

## Old shuffled control

The training script used history.roll(1) with mask.roll(1). When the final token
is padding, the effective sequence of valid memory writes is unchanged. The new
CPU audit checks actual tokenizer masks for the selected evaluation prompts and
all four adapter-training prompts. It also compares memory states using the real
trained adapter and synthetic features. Full GPU evaluation records actual
backbone-feature state differences for roll, reverse and wrong-history controls.

CPU checks and preparation reports do not mean the GPU evaluation completed.
Look for status=complete and summary/summary_action7 in the final JSON reports.

## Follow-up controls after the first 8-task result

The first evaluation completed on 2026-09-26. Layer 4 reduced real-action
loss from 0.332867 to 0.325814 over 16 episodes; different-task history gave
0.327178, while same-task other-episode history gave 0.326131. The same-task
difference remained uncertain. Layer 9 was weaker. These results selected L4
for a larger, separate task set.

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/run_mone_controls.sh
```

This evaluates the existing L4 adapter on 20 tasks excluded from the first
evaluation report, two episodes per task, with three frames and three paired
noise/time draws. It adds two controls: current observation passed through the
history encoder, and one fixed memory state from a held-out donor reused for
every evaluation sample. The fixed state is encoded once with its original
task prompt; it does not vary with the evaluation prompt. Both controls test
whether the adapter mainly provides a generic correction. No training occurs.

The report adds `summary_task` and `summary_action7_task`. These bootstrap
task means, so the two episodes from one task remain together. The previous
episode-level summaries remain available. As before, this measures offline
flow loss, not LIBERO closed-loop success.

## Result and selective-training pilot

The 20-task L4 control run completed. On the seven real action dimensions,
baseline loss was 0.347526; relevant history was 0.342936, different-task
history 0.342960, same-task other-episode history 0.342859, current frame
0.342806, and fixed memory 0.342947. Task-level paired intervals for relevant
against every nonempty control crossed zero. The old action-only objective
learned a useful adapter correction but did not establish a history-specific
benefit. The four original training episodes represented only three closely
related bowl tasks.

The next pilot starts from the existing L4 adapter and trains on the 20 tasks
in the completed control report. For each current frame it caches the correct
frame from ten steps earlier, another episode of the same task, a different
task's visual/state history with the current prompt, and the current frame.
One negative type is paired with each positive training step. Across three
epochs, each example sees all three negative types and noise/time draws.
The training loss adds a smooth paired ranking term to the seven-dimensional
action loss. The wrong-history loss is capped at the frozen-backbone baseline
inside the ranking term; this removes any incentive to make it arbitrarily
worse. Training metrics alone do not establish a useful memory mechanism.

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/run_mone_selective.sh
```

The runner first does a two-step GPU smoke, then trains 360 steps on one visible
GPU, then evaluates both the old and new adapters on the 12 tasks absent from
both previous evaluation reports. It writes separate logs, checkpoints, and
paired evaluation reports. The final decision should compare the new adapter
with the old one on those same 12 tasks: relevant history should beat the
fixed, current-frame, and wrong-history controls without losing its advantage
against the no-memory baseline. If it does not, do not scale this objective.

## Automatic offline gate and paired LIBERO rollout pilot

The selective run tagged `20260926-124724-16706` finished. Its 12-task action7
loss was 0.326581 with relevant history versus 0.334673 for the old adapter and
0.338321 for the frozen backbone. The paired task-level offline gate is
`inconclusive`: relevant is better than fixed memory and current-frame memory,
but the intervals against same-task and different-task wrong histories cross
zero. This permits a small rollout screen; it is not offline proof of a
history-specific gain.

The new rollout evaluator uses `LayerMemory.activate(model, state, layer=4)`
around the actual `sample_actions` call, matching adapter training. The older
`sample_actions(mone_state=...)` path prepends tokens and is a different
architecture; do not use that path for this L4 checkpoint. Historical
observations are taken ten simulator steps before each planning observation.
Baseline, relevant and fixed-memory arms use the same LIBERO task, initial
state, action sampler seed and five-step replanning interval. The fixed memory
is encoded once from the source report and reused for all trials.

On an H200 node, run after the current selective run finishes:

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/run_mone_gates_after_train.sh 20260926-124724-16706
```

This writes or reuses the offline gate report, runs one LIBERO rollout smoke,
then evaluates one held-out task each from LIBERO-10, Goal and Spatial on three
visible GPUs in parallel. Each task has five paired initial states and three
arms (45 rollouts total). Per-arm logs and JSONL progress are preserved; the
merged JSON reports success rates and paired trial wins/losses. This is a
small screen, not an official benchmark estimate. The runner skips rollout
only if the offline gate says `fail_current_adapter`; its current result is
`inconclusive`, so it proceeds.

The Mone checkpoint was converted from the shared **pi0.5 base** weights and
combined with LIBERO normalization statistics; it is not the official
LIBERO-finetuned pi0.5 checkpoint. If all three rollout arms have near-zero
success, this test cannot decide whether historical memory is useful. It first
establishes whether this backbone can execute the selected LIBERO tasks at all.

LIBERO simulator source is pinned at commit
`8f1084e3132a39270c3a13ebe37270a43ece2a01` in Mone's `vendor/libero`.
Its config is in Mone's `assets/libero_config`, and the simulator dependencies
are installed in the Mone `.venv`. ACOT code and environments are untouched.

The first H200 rollout smoke hit the known GPU-image missing-`libEGL.so.1`
case before reaching the simulator. Mone now has its own private GLVND library
copies in `assets/glstub/lib`; the runner appends that directory to
`LD_LIBRARY_PATH` and probes a real EGL context before loading the model.
The source files were read only; no ACOT file was changed.

## Completed paired rollout and interpretation

The H200 run tagged `20260926-124724-16706` completed after the EGL fix.
All 15 paired initial states were evaluated in three arms, for 45 rollouts.
Baseline, relevant history, and fixed history each succeeded in 0/15 trials.
All runs reached their suite's full time limit (LIBERO-10: 520; Goal: 300;
Spatial: 220 steps), with no evaluator exception. The paired comparisons are
therefore all ties. See `outputs/heldout/L4-rollout-paired-20260926-124724-16706.json`.

This is an inconclusive *mechanism* test, not evidence that memory harms or
helps success: the no-memory backbone also failed every selected task. The
current checkpoint is converted pi0.5 base, not a LIBERO-finetuned policy.
Do not scale Mone training based on the offline loss improvement alone. The
next meaningful success-rate test needs a competent π0.5 LIBERO backbone,
followed by retraining the Mone adapter on that exact backbone and repeating
the paired baseline/relevant/fixed-history rollouts. The existing adapter is
not portable to different backbone features without validation and retraining.

## Frozen history-content screen prepared

`scripts/run_mone_history_screen.sh` provides `prepare`, `screen`, and
`confirm` stages. Preparation completed for tag `20260926-124724-16706` and
froze a hashed plan at `outputs/heldout/L4-history-plan-20260926-124724-16706.json`.
It chooses 12 adapter-training-disjoint tasks, two new trajectories per task
for exploration, two for confirmation, and a separate same-task history
donor. The fixed-memory donor is also separate. Five evaluation frames per
trajectory are predetermined; no new training occurs.

For each current frame the evaluator compares clean vision, a gray wrist
camera, and fixed central gray occlusions in both cameras. Each condition
uses the same current observation, action target, noise and flow time across
no memory, correct t-10 history, same-task wrong history, fixed history,
and current-frame history. Historical donor inputs remain unoccluded. Results
use true 7D-action flow loss and paired bootstrap over task means.

The preregistered primary screen uses central occlusion: correct history
must beat all four controls with positive task-bootstrap lower bounds, beat
same-task wrong history by at least 0.001 on average and on at least 9 of
12 tasks, and occlusion must raise baseline loss by at least 0.001. Clean
memory cannot worsen baseline mean loss by over 1%. These thresholds are
screening rules, not proof of closed-loop success. Confirmation is enabled
only if the screen passes.

The login node has no CUDA devices. CPU planning, static checks and unit
tests are done; GPU smoke and evaluation remain for an H200 node:

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/run_mone_history_screen.sh screen
# Only if the screen report says screen_positive:
bash scripts/run_mone_history_screen.sh confirm
```

Each full stage starts with one smoke case, then shards the 12 tasks across
four GPUs. Logs and JSONL progress files are written within Mone.

The H200 exploratory screen completed for tag `20260926-124724-16706`:
12 tasks, 24 new trajectories, 1,080 paired condition/noise samples, with
all four shards complete. The verdict is `inconclusive`; the confirmation
split remains untouched by the preregistered stop rule. In the primary
central-occlusion condition, correct-history action7 loss was 0.341232 versus
0.343582 without memory, 0.341451 with same-task wrong history, 0.341403
with fixed history, and 0.341839 with the current frame as history. The
correct-versus-same-task gain was 0.000219 (task-bootstrap 95% descriptive
interval [0.000081, 0.000366]), below the preregistered 0.001 minimum.
Correct versus fixed history was 0.000171 with an interval crossing zero.

The central occlusion also failed its sensitivity check: the no-memory
baseline loss was 0.345341 with clean vision and 0.343582 with the central
occlusion, a mean change of -0.001760. The wrist-camera ablation raised mean
baseline loss by 0.006478, but the task-bootstrap interval crossed zero and
correct-versus-same-task history was only 0.000076. Clean-vision results
showed a much larger generic adapter gain (0.011453 versus baseline) than
the correct-versus-same-task history gain (0.000201). Thus the current pilot
does not establish a practically useful history-specific mechanism.
See `outputs/heldout/L4-history-screen-20260926-124724-16706.json`.

## Official pi0.5 LIBERO baseline assets

The shared ACoT asset is `pi05_base`, not the official LIBERO-finetuned
pi0.5. The `ravenking/checkpoints/LIBERO-*` safetensors are OpenVLA models.
The 16 upstream `pi05_libero` objects are now downloaded into Mone's private
`checkpoints/pi05_libero_official_jax`, with its own LIBERO norm stats.
Orbax metadata reads successfully. The login container has a 16 GiB cgroup
limit and the full converter was OOM-killed; it now checks the limit before
retrying. On an H200 node with at least 32 GiB container RAM, run:

```bash
cd /path/to/workspace/mone_pi05_server/openpi
bash scripts/prepare_mone_official_libero.sh convert &&
bash scripts/run_pi05_libero_baseline.sh
```

Conversion writes `checkpoints/pi05_libero_official_pytorch`; the old
`pi05_libero_pytorch` (actually converted base) is preserved. The baseline
runner uses the official norm stats and tests 5 official initial states each
for Spatial, Goal and LIBERO-10, saving per-trial action ranges and robot
displacement. If baseline success is still zero, inspect the action pipeline
before training another Mone adapter.
