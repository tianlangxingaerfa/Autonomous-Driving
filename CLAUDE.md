# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A modular autonomous driving (AD) stack implemented in Python, running against a live CARLA simulator instance. The single entry-point script `ad_stack.py` connects to CARLA, spawns an ego vehicle, and drives a synchronous control loop at 20 Hz.

## Running the Stack

CARLA must be running locally before starting (`localhost:2000`). Start CARLA first, then:

```bash
python ad_stack.py
```

No build step is required. Dependencies: `carla` (matching your CARLA server version) and `numpy`.

## Architecture

All modules live in `ad_stack.py`. Data flows strictly in one direction per tick:

```
Perception → STM → WorldModel → Cost → Actor → carla.VehicleControl
```

| Module | Class | Role |
|--------|-------|------|
| Configurator | `Configurator` | Central config: CARLA host/port, hyperparameters, cost weights |
| Short-Term Memory | `ShortTermMemory` | Rolling deque (N frames) of `EgoFrame` + `ActorFrame` lists |
| Perception | `Perception` | Pulls CARLA telemetry + nearby actors each tick; writes to STM |
| World Model | `WorldModel` | Constant-velocity linear extrapolation of actor positions for T steps |
| Intrinsic Cost | `IntrinsicCost` | Lane-center error, speed tracking, jerk/comfort |
| Critic Cost | `CriticCost` | Collision risk from world-model predictions; traffic penalties |
| Policy | `Policy` | MPPI-style: samples N candidate (throttle, steer) sequences |
| Optimizer | `Optimizer` | Weights candidates by MPPI cost, returns `carla.VehicleControl` |

`CostModule` aggregates `IntrinsicCost + CriticCost`. `Actor` owns `Policy + Optimizer` and exposes a single `act()` call.

## Key Design Decisions

- **Synchronous mode** (`world.tick()` called manually) — ensures deterministic 20 Hz steps. Disable by setting `Configurator.SYNC_MODE = False` and removing the `world.tick()` call.
- **Placeholder cost math** — `CriticCost.compute` and `Policy.sample` contain stub logic. The CARLA data pipeline (state retrieval, control application) is fully functional; only the RL/optimization internals need replacement.
- **MPPI in Optimizer** — current implementation applies cost only to the current state (not per-trajectory rollout). True MPPI requires simulating each candidate forward through `WorldModel`.
- **Cleanup guarantee** — `ego.destroy()` and async-mode restoration are in a `try/finally`, so they run even on `KeyboardInterrupt`.

## Extending the Stack

- **Add a sensor** (camera, LiDAR): spawn it attached to `ego` in `main()`, collect data in `Perception.tick()`, store in a new `STM` field.
- **Replace Policy/Optimizer**: implement your RL policy by subclassing or replacing `Policy.sample` and `Optimizer.optimize`. The rest of the pipeline is unchanged.
- **Multi-file refactor**: each class maps cleanly to its own module (`configurator.py`, `memory.py`, `perception.py`, etc.) when the file grows too large.
- **Tuning costs**: all weights are in `Configurator` (`W_LANE_CENTER`, `W_SPEED_TRACK`, `W_COMFORT_JERK`, `W_COLLISION`, `W_TRAFFIC_RULE`).
