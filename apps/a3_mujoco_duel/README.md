# Two-A3 physical-ball reference experiment

Measured results and unresolved long-run failures are recorded in
[VALIDATION.md](VALIDATION.md).

Two independent model21800 ONNX controllers drive two floating-base Agibot A3
robots in one MuJoCo model. Both controllers run at 50 Hz, each with 110-D
observations, 31 joint targets, its own lifecycle and applied-action history.
The second player's positions, orientations and world vectors are transformed
through a 180-degree table-centred frame. Joint state and body-frame gyro remain
in each robot's own convention.

The scene contains one physical ball, one table/net, and 62 controlled joints.
Only the initial incoming ball of each point is launched by the harness. During
a point, gravity, drag, table bounces and paddle contacts drive the ball; there
is no assisted return, proximity contact substitute or per-hit ball reset.
Robots continue through recovery between points and are initialized only once.

## Run the verified configuration

Use the existing reference environment (validated on macOS arm64 with Python
3.12.16, MuJoCo 4.0.0 and ONNX Runtime 1.31.1), or install the reference runner's
NumPy/PyYAML/MuJoCo/ONNX Runtime dependencies. Run from the repository root:

```bash
python apps/a3_mujoco_duel/duel.py \
  --points 16 --seed 0 --strike-plane-x 0.2363 \
  --impact-model --landing-y 0.44 --station-tracking \
  --output /tmp/hope-duel-demo
```

Use a new output directory for every run; existing evidence is never overwritten.
The flags select the evaluated profile:

- The local hit plane is at x=0.5363 m (0.2363 m past the near edge), approximately
  0.58 m ahead of the initial pelvis, matching model21800's fixed-plane recipe.
- `--impact-model` converts desired outgoing ball velocity into paddle velocity
  using the no-spin normal-impact equation from the reference Planner.
- `--landing-y 0.44` aims toward the opponent's forehand lane in each player's frame.
- `--station-tracking` uses exported per-clip reach offsets and the policy's
  trained lateral station feedback; it does not prescribe a gait or move the root.

Each point is capped at 6 simulated seconds by default, followed by 2 seconds
of recovery. `--max-point-seconds` changes that observation window. A time-capped
point is not evidence that the ball naturally went out. These are reference
experiments, not a complete competition referee or a hardware deployment path.

The measured seed-0 run lasted 86.38 seconds over 16 points. Its longest point
contained six consecutive alternating real racket contacts (three per robot),
each preceded by net clearance and one own-half table bounce. Both robots stayed
upright; maximum pelvis tilts were 19.77° / 20.05° and minimum heights were
0.932 / 0.958 m. The result is a working short-rally demonstration, not a claim
of robust indefinite rallying or a policy training result.

Extending that same seed-0 profile to 50 points ran for 245.52 simulated seconds
but did not maintain continuous standing: P1 fell during point 30 and P2 during
point 45. Eight of 50 points had at least two valid alternating contacts, and
the longest chain remained six. The first 16 points reproduced the shorter
run's qpos exactly. Keep the short-rally result separate from long-run stability.

Without these profile flags, the CLI retains the simple fixed-station/legacy
target baseline for comparison. No changes are made to the existing single-robot
evaluation path by this application.

## Evidence and video

Outputs are `report.json` (contacts, table bounces, net crossings, posture and
model identity), `states.npz` (all qpos), and `trace.jsonl` (per-tick evidence).
Only raw MuJoCo paddle contacts count. The lightweight referee separately checks
the valid alternating-contact prefix of each point; a contact alone is not a
legal return. Its scope excludes a full service/let/scoring rules implementation.

With ImageIO, imageio-ffmpeg and Pillow installed, render the full record or one
uninterrupted point, at the original simulation speed:

```bash
python apps/a3_mujoco_duel/render.py /tmp/hope-duel-demo \
  --output /tmp/hope-duel-full.mp4
python apps/a3_mujoco_duel/render.py /tmp/hope-duel-demo \
  --point 10 --output /tmp/hope-duel-point10.mp4
```

Rendering replays saved qpos with `mj_forward`; it does not run a new policy or
change the recorded physics. Point excerpts exclude the subsequent ball-parking
interval; full videos label the intervals between points. Add `--auto-frame` to
keep both robots and the table visible if they move toward the edge of the arena.

## Checks

```bash
python apps/a3_mujoco_duel/test_duel.py
```

The tests check mirrored local observations, disjoint actuator ownership,
independent policy histories, one physical step per control tick, true ball
contact with both rackets, and the moving-paddle impact equation.
