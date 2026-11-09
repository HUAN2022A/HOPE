# Local reference validation — 2026-11-09

These are local Python/ONNX Runtime/MuJoCo measurements using the published
model21800 export, based on upstream commit
`7990f18c2c1ae560c9e0708dca10228b9f2cc610`. They are not native Gate3 or hardware
qualification results. No policy training or weight changes were performed.

## Environment and identity

| Component | Observed version |
| --- | --- |
| Platform | macOS arm64, Apple M2 |
| Python | 3.12.16 |
| NumPy / PyYAML | 2.5.4 / 6.0.3 |
| ONNX Runtime | 1.31.1, CPUExecutionProvider |
| MuJoCo | 4.0.0 |
| Actor contract | 110-D observation plus reference clock, 31-D action, 50 Hz |
| Actor SHA-256 | `6bf1a2418f8538e23577a0153f2fe6a1e78dee91f41650a232259432a84a4dc8` |

The original policy, deploy parameters, robot source MJCF, PD gains, action
scales, and shared ball-physics configuration files were retained. Rendering
dependencies and generated recordings are not deployment inputs.

## Standalone evaluator regressions

The original seed-0 first incoming ball produced a predicted racket height of
approximately -0.767 m because prediction omitted the physical table bounce.
The source robot MJCF also supplied a Gate3 arena in a different frame, which
intersected the standalone scene's robot at initialization. Finally, rejecting
an invalid intercept without cancelling its pending swing could leave the
controller following an obsolete target.

The reference changes add table-bounce prediction, exclude the inherited
Gate3 table/net from the standalone scene, and recover from cancelled
pre-strike tasks without resetting robot state or applied-action history.

The deterministic one-ball 1.7-second probe changed from a maximum pelvis-axis
tilt of 60.62 degrees to 12.51 degrees. A separate corrected single-robot run
completed 50 balls over 127.26 seconds without falling, with maximum tilt
17.88 degrees and minimum pelvis height 0.902 m. It recorded 14 balls with real
racket contacts and 2/50 scored returns. The single-robot legacy success metric
includes a proximity fallback; do not compare it directly to raw-contact-only
dual-robot measurements.

## Two-robot measurements

Configuration:

```bash
python apps/a3_mujoco_duel/duel.py \
  --points 50 --seed 0 --max-point-seconds 6 \
  --strike-plane-x 0.2363 --impact-model --landing-y 0.44 --station-tracking \
  --output /tmp/hope-duel-50-new
```

Use `--points 16` for the shorter reference run. The two controllers have
independent histories and actuator sets, while one MuJoCo plant advances the
only physical ball. Ball launches occur between points; there is no within-point
ball assistance, proximity scoring substitute, or robot reset between points.

| Measurement | 16 points | 50 points |
| --- | ---: | ---: |
| Simulated duration, including recovery | 86.38 s | 245.52 s |
| Points with at least two valid alternating contacts | 5/16 | 8/50 |
| Longest valid alternating-contact prefix | 6 | 6 |
| P1 / P2 raw racket contacts | 12 / 10 | 23 / 28 |
| Both robots standing through the complete run | Yes | **No** |

The shorter run had maximum pelvis tilts 19.77/20.05 degrees and minimum
heights 0.932/0.958 m. A repeat produced bitwise-identical qpos for all
4,319 frames and 83 position coordinates. The first 16 points of the 50-point
run also reproduced that state sequence exactly.

The longer run exposed unresolved instability:

| Robot | First pelvis tilt >60 degrees | First pelvis height <0.4 m |
| --- | --- | --- |
| P1 | Point 30, 158.80 s | Point 30, 159.42 s |
| P2 | Point 45, 222.68 s | Recovery after point 45, 224.40 s |

Both robots remained down at the end. All recorded qpos stayed finite and
simulation time advanced monotonically; process completion is not a stability
pass. Later points were retained after falling. The cause of this long-run
instability has not been established by the earlier standalone fixes.

The lightweight contact qualifier requires alternating real paddle contacts,
net clearance before reception, and exactly one bounce on the receiver's half.
It is not a complete competition referee. Each point has a six-second observation
cap; points 10, 15, and 24 reached it in the 50-point run. Do not treat capped
points as natural misses or six contacts as a demonstrated skill limit.

## Checks and retained evidence

The following groups passed locally: seven standalone evaluator/lifecycle
regressions, five existing success-metric tests, and three two-robot checks.

```bash
python hope_training/whole_body_tracking/tests/test_reference_evaluator_physics.py
python hope_training/whole_body_tracking/tests/test_success_metric.py
python apps/a3_mujoco_duel/test_duel.py
```

Raw logs, source hashes, qpos arrays, per-step/per-point reports and videos were
retained locally. They are excluded from this source contribution. The run's
`states.npz` stores qpos for visualization, not a full physics/controller restart
snapshot; exact continuation additionally requires velocities and controller
and physics state.

Next validation should reproduce the point-30/45 failures with full-state
snapshots, fix their cause, and repeat the same 50-point configuration before
claiming long-run stability. Multi-seed and longer-point tests remain future work.
