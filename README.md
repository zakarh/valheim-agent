# Valheim Player

Pure-pixel, goal-conditioned reinforcement learning from human feedback for Valheim.

The project is built around this loop:

```text
human or AI plays Valheim with a named goal
-> episodes are recorded as pixels, actions, and metadata
-> you compare same-goal A/B rollouts
-> a goal-conditioned reward model learns your preferences
-> RL trains a policy against that learned reward
-> the improved AI is recorded and compared again
```

The goal is to avoid object labels and hand-written Valheim reward heuristics. The reward comes from your preferences over recorded behavior. Goal labels such as `explore`, `gather_wood`, or `combat` tell one model which behavior you want.

## Setup

```powershell
python -m pip install -r requirements.txt
```

Use a safe single-player world while testing. Live-control scripts send keyboard and mouse input to the focused game window.

## Project Files

```text
valheim_capture.py          Window capture foundation
valheim_actions.py          Shared keyboard/mouse action definitions
valheim_control.py          Shared live input helpers
valheim_goals.py            Shared goal parsing, metadata, and id helpers
valheim_rl_actions.py       Discrete action set for RL policies

record_gameplay.py          Optional human gameplay recorder with goal labels
train_behavior_clone.py     Goal-conditioned behavior-cloning trainer
run_behavior_clone.py       Behavior-clone live runner
behavior_model.py           Behavior-cloning CNN

record_ai_rollout.py        Record AI episodes as video/actions/metadata
compare_rollouts.py         Make same-goal A/B comparison videos and save preferences
preference_reward_model.py  Goal-conditioned pixel reward model and clip helpers
train_reward_model.py       Train reward model from A/B preferences
score_rollouts.py           Score recorded episodes with the reward model
valheim_rl_env.py           Live Gymnasium env using learned reward
train_rlhf_policy.py        Train DQN against the learned reward
rollout_recorder.py         Shared rollout video/action recorder
```

## Generated Folders

```text
captures/      Latest capture snapshots
datasets/      Optional human gameplay recordings
models/        Behavior, reward, and RL policy models
rollouts/      AI episode recordings
comparisons/   Side-by-side A/B comparison videos
preferences/   Human preference labels and rollout scores
```

These folders are ignored by git.

## Goals

A goal is a behavior label stored in recording metadata. Goal names are normalized to lowercase with underscores, so `Gather Wood` becomes `gather_wood`.

Examples:

```text
general
explore
gather_wood
combat
build
```

Older recordings and checkpoints without goal metadata are treated as `general`.

## 1. Confirm Window Capture

Launch Valheim, or run this first and then launch Valheim:

```powershell
python .\valheim_capture.py
```

The script waits for `valheim.exe`, finds the visible Valheim game window, captures the client area, logs FPS, and refreshes:

```text
captures/latest.png
```

This is closer to OBS Window Capture than OBS Game Capture. If the window is minimized or fully covered, the script may not see the game.

## 2. Optional: Record Human Demonstrations

Behavior cloning is useful for bootstrapping a policy before RLHF. Record separate sessions for each behavior you care about:

```powershell
python .\record_gameplay.py --goal explore --fps 10 --duration 600
python .\record_gameplay.py --goal gather_wood --fps 10 --duration 600
python .\record_gameplay.py --goal combat --fps 10 --duration 600
```

Each session writes `meta.json` with the selected goal plus frame/action data.

Train one goal-conditioned behavior model:

```powershell
python .\train_behavior_clone.py --goals explore,gather_wood,combat --epochs 10
```

Run a specific behavior from the same checkpoint:

```powershell
python .\run_behavior_clone.py --goal gather_wood --dry-run
python .\run_behavior_clone.py --goal gather_wood
```

## 3. Record AI Rollouts

A rollout is one AI episode:

```text
rollouts/<policy_name>/<episode_id>/
  episode.mp4
  actions.csv
  meta.json
  thumbnail.jpg
```

Start with random same-goal baselines:

```powershell
python .\record_ai_rollout.py --policy random --goal explore --policy-name random_explore_a --duration 60
python .\record_ai_rollout.py --policy random --goal explore --policy-name random_explore_b --duration 60 --seed 11
```

Record a behavior clone for a selected goal:

```powershell
python .\record_ai_rollout.py --policy behavior_clone --model-path .\models\behavior_clone.pt --goal gather_wood --policy-name bc_gather --duration 60
```

Record an RLHF DQN policy:

```powershell
python .\record_ai_rollout.py --policy dqn --model-path .\models\rlhf_dqn.zip --goal explore --policy-name rlhf_explore --duration 60
```

Dry-run mode records video and predictions without sending controls:

```powershell
python .\record_ai_rollout.py --policy random --goal explore --policy-name random_dry --dry-run --duration 30
```

`record_ai_rollout.py` waits 5 seconds before recording/sending inputs so you can prep the scene. Override it with `--start-delay 0`.

## 4. Compare Rollouts And Save Preferences

Create side-by-side A/B videos and choose which same-goal episode is better:

```powershell
python .\compare_rollouts.py --goal explore --policy-a random_explore_a --policy-b random_explore_b --pairs 5
```

If `--goal` is omitted, the comparer still avoids cross-goal pairs. Preference labels go to:

```text
preferences/preferences.csv
```

Choices:

- `a`: AI A did better
- `b`: AI B did better
- `t`: tie
- `s`: skip
- `q`: quit

## 5. Train The Learned Reward Model

Train a goal-conditioned reward model from your A/B choices:

```powershell
python .\train_reward_model.py --epochs 20
```

The model samples short clips from each preferred/non-preferred episode and trains with a pairwise preference loss:

```text
score(preferred clip, goal) > score(non-preferred clip, goal)
```

You can restrict or order goals explicitly:

```powershell
python .\train_reward_model.py --goals explore,gather_wood,combat --epochs 20
```

Output:

```text
models/reward_model.pt
```

You need enough preference labels for this to become meaningful. A few labels prove the pipeline; dozens to hundreds are better for actual learning.

## 6. Score Rollouts With The Reward Model

Before using the learned reward for RL, check whether it ranks existing rollouts sensibly:

```powershell
python .\score_rollouts.py
```

By default each rollout is scored using the goal in its `meta.json`. Override that with:

```powershell
python .\score_rollouts.py --goal combat
```

Output:

```text
preferences/rollout_scores.csv
```

If the ranking looks backwards, collect more comparisons or fix inconsistent labels before training the RL policy.

## 7. Train RL Against The Learned Reward

Load into a safe Valheim world and keep the game focused:

```powershell
python .\train_rlhf_policy.py --goal explore --timesteps 10000
```

This trains a DQN policy in a live Gymnasium environment:

```text
Valheim pixels + optional goal planes -> DQN action -> keyboard/mouse input
rolling pixel clip + goal -> reward model score -> RL reward
```

Train one conditioned DQN across multiple goals:

```powershell
python .\train_rlhf_policy.py --goals explore,gather_wood,combat --timesteps 50000
```

When multiple goals are supplied, the environment appends one-hot goal planes to the pixel observation and randomly selects a goal on reset. A sidecar metadata file is saved next to the DQN zip so rollout recording can rebuild the same observation shape.

Useful options:

```powershell
python .\train_rlhf_policy.py --goal explore --timesteps 50000 --reward-scale 2.0
python .\train_rlhf_policy.py --goal explore --no-input
python .\train_rlhf_policy.py --goal explore --reward-mode score
```

The default reward mode is `delta`, meaning each step is rewarded by the change in learned reward score. This avoids relying too heavily on the reward model's arbitrary absolute score offset.

## 8. Record The RLHF Policy And Repeat

Record the newly trained policy:

```powershell
python .\record_ai_rollout.py --policy dqn --model-path .\models\rlhf_dqn.zip --goal explore --policy-name rlhf_explore --duration 60
```

Compare it against an older same-goal policy:

```powershell
python .\compare_rollouts.py --goal explore --policy-a rlhf_explore --policy-b random_explore_a --pairs 5
```

Then retrain the reward model with the expanded preference file:

```powershell
python .\train_reward_model.py --epochs 20
```

Repeat:

```text
record -> compare -> train reward -> train RL -> record
```

## Minimal First Full Loop

```powershell
python .\record_ai_rollout.py --policy random --goal explore --policy-name random_explore_a --duration 30
python .\record_ai_rollout.py --policy random --goal explore --policy-name random_explore_b --duration 30 --seed 11
python .\compare_rollouts.py --goal explore --policy-a random_explore_a --policy-b random_explore_b --pairs 3
python .\train_reward_model.py --epochs 5
python .\score_rollouts.py
python .\train_rlhf_policy.py --goal explore --timesteps 1000
python .\record_ai_rollout.py --policy dqn --model-path .\models\rlhf_dqn.zip --goal explore --policy-name rlhf_test --duration 30
```

This proves the whole path works. It will not produce a strong Valheim player yet.

## Notes And Limitations

- The reward model learns from your preferences; it is not guaranteed to match your intent until it has enough labels.
- Early reward models can be exploitable. Always inspect rollout videos and score rankings.
- Preferences should compare rollouts for the same goal.
- Live RL is slow because Valheim is the simulator.
- DQN starts with a small discrete action set in `valheim_rl_actions.py`.
- Behavior cloning only learns behaviors present in your dataset.
- Mouse capture/control can be imperfect in games that lock or recenter the cursor.
- Keep Valheim focused when running live-control scripts.
