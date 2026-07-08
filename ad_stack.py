"""
Modular Autonomous Driving Stack for CARLA Simulator.

Architecture:
  Configurator -> Perception -> STM -> WorldModel
                                           |
                                     Cost (Intrinsic + Critic)
                                           |
                                    Actor (Policy + Optimizer)
                                           |
                                   carla.VehicleControl
"""

import math
import random
import collections
import numpy as np
import carla


# ---------------------------------------------------------------------------
# 1. Configurator
# ---------------------------------------------------------------------------

class Configurator:
    # CARLA server
    HOST = "localhost"
    PORT = 2000
    TIMEOUT = 30.0
    SYNC_MODE = True
    FIXED_DELTA_SECONDS = 0.05  # 20 Hz

    # Ego vehicle
    VEHICLE_BLUEPRINT = "vehicle.tesla.model3"
    TARGET_SPEED      = 5.0   # m/s (~18 km/h)
    SPEED_KP          = 0.5

    # STM
    STM_BUFFER_LEN = 20

    # World model
    PREDICTION_HORIZON = 15   # steps to predict forward (~0.75 s)

    # Cost weights
    W_LANE_CENTER  = 1.0
    W_SPEED_TRACK  = 2.0    # strong pull toward target speed
    W_COMFORT_JERK = 0.3
    W_COLLISION    = 20.0   # avoidance priority, but not overwhelming
    W_TRAFFIC_RULE = 5.0

    # Collision geometry
    COLLISION_RADIUS = 2.0     # m — soft margin around ego centroid
    DANGER_RADIUS    = 4.0     # m — within this → exponential cost

    # Camera view mode: "third_person" or "bird"
    CAMERA_VIEW = "third_person"

    # Traffic
    NPC_VEHICLES = 100
    NPC_WALKERS  = 50

    # Optimizer (MPPI)
    MPPI_SAMPLES    = 128      # more samples → better coverage
    MPPI_LAMBDA     = 0.5      # lower temperature → sharper selection
    MPPI_NOISE_STD  = 0.15     # perturbation on throttle/steer

    # Emergency reactive layer
    EMERGENCY_DIST  = 6.0      # m — trigger hard brake + steer
    EMERGENCY_CONE  = 40.0     # half-angle (deg) of forward threat cone


# ---------------------------------------------------------------------------
# 2. Short-Term Memory (STM)
# ---------------------------------------------------------------------------

class EgoFrame:
    def __init__(self, location, velocity, acceleration, transform):
        self.location     = location
        self.velocity     = velocity
        self.acceleration = acceleration
        self.transform    = transform


class ActorFrame:
    def __init__(self, actor_id, location, velocity, bounding_box):
        self.actor_id     = actor_id
        self.location     = location
        self.velocity     = velocity
        self.bounding_box = bounding_box


class ShortTermMemory:
    def __init__(self, maxlen=Configurator.STM_BUFFER_LEN):
        self.ego_frames   = collections.deque(maxlen=maxlen)
        self.actor_frames = collections.deque(maxlen=maxlen)

    def push(self, ego_frame: EgoFrame, actor_list: list):
        self.ego_frames.append(ego_frame)
        self.actor_frames.append(actor_list)

    def latest_ego(self) -> "EgoFrame | None":
        return self.ego_frames[-1] if self.ego_frames else None

    def latest_actors(self) -> list:
        return self.actor_frames[-1] if self.actor_frames else []

    def ego_velocity_history(self) -> np.ndarray:
        return np.array([[f.velocity.x, f.velocity.y, f.velocity.z]
                         for f in self.ego_frames])


# ---------------------------------------------------------------------------
# 3. Perception
# ---------------------------------------------------------------------------

class Perception:
    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle, stm: ShortTermMemory):
        self.world       = world
        self.ego_vehicle = ego_vehicle
        self.stm         = stm
        self.carla_map   = world.get_map()
        self.current_waypoint = None

    def tick(self):
        ego = self.ego_vehicle
        tf  = ego.get_transform()
        loc = tf.location
        vel = ego.get_velocity()
        acc = ego.get_acceleration()

        ego_frame = EgoFrame(loc, vel, acc, tf)

        actor_list = []
        for actor in self.world.get_actors():
            if actor.id == ego.id:
                continue
            if not isinstance(actor, (carla.Vehicle, carla.Walker)):
                continue
            a_loc = actor.get_location()
            if loc.distance(a_loc) > 50.0:
                continue
            a_vel = actor.get_velocity()
            a_bb  = actor.bounding_box
            actor_list.append(ActorFrame(actor.id, a_loc, a_vel, a_bb))

        self.current_waypoint = self.carla_map.get_waypoint(loc)
        self.stm.push(ego_frame, actor_list)


# ---------------------------------------------------------------------------
# 4. World Model
# ---------------------------------------------------------------------------

class WorldModel:
    """Constant-velocity linear extrapolation of surrounding actors."""
    def __init__(self, stm: ShortTermMemory,
                 horizon=Configurator.PREDICTION_HORIZON,
                 dt=Configurator.FIXED_DELTA_SECONDS):
        self.stm     = stm
        self.horizon = horizon
        self.dt      = dt

    def predict(self) -> dict:
        """Returns {actor_id: (horizon, 3) np.ndarray} of predicted XY positions."""
        predictions = {}
        for frame in self.stm.latest_actors():
            pos = np.array([frame.location.x, frame.location.y, frame.location.z])
            vel = np.array([frame.velocity.x, frame.velocity.y, frame.velocity.z])
            traj = np.array([pos + vel * self.dt * t for t in range(1, self.horizon + 1)])
            predictions[frame.actor_id] = traj
        return predictions


# ---------------------------------------------------------------------------
# 5. Ego Kinematic Simulator
# ---------------------------------------------------------------------------

def _simulate_ego_trajectory(
    ego_frame: EgoFrame,
    actions: np.ndarray,          # (H, 3): [throttle, steer, brake]
    dt: float,
    max_speed: float = 20.0,
) -> np.ndarray:
    """
    Simple bicycle-model forward simulation.
    Returns (H+1, 3) array of [x, y, yaw] for each timestep (step 0 = current).

    Uses a kinematic bicycle model:
      x'   = x + v * cos(yaw) * dt
      y'   = y + v * sin(yaw) * dt
      yaw' = yaw + (v / L) * tan(steer * max_steer) * dt
      v'   = v + (throttle - brake) * accel_scale * dt
    """
    L = 2.875          # Tesla Model 3 wheelbase (m)
    max_steer_rad = math.radians(40.0)
    accel_scale   = 4.0   # m/s² per unit throttle/brake

    yaw = math.radians(ego_frame.transform.rotation.yaw)
    x   = ego_frame.location.x
    y   = ego_frame.location.y
    v   = max(0.0, math.hypot(ego_frame.velocity.x, ego_frame.velocity.y))

    traj = np.empty((actions.shape[0] + 1, 3))
    traj[0] = [x, y, yaw]

    for i, (throttle, steer, brake) in enumerate(actions):
        delta = steer * max_steer_rad
        v += (throttle - brake) * accel_scale * dt
        v = float(np.clip(v, 0.0, max_speed))
        x += v * math.cos(yaw) * dt
        y += v * math.sin(yaw) * dt
        if v > 0.1:
            yaw += (v / L) * math.tan(delta) * dt
        traj[i + 1] = [x, y, yaw]

    return traj


# ---------------------------------------------------------------------------
# 6. Cost Module
# ---------------------------------------------------------------------------

class IntrinsicCost:
    def __init__(self, cfg: type = Configurator):
        self.cfg = cfg

    def compute_trajectory(
        self,
        ego_traj: np.ndarray,       # (H+1, 3): [x, y, yaw] per step
        actions: np.ndarray,         # (H, 3): [throttle, steer, brake]
        waypoint,
        current_speed: float,
    ) -> float:
        cfg = self.cfg

        # Lane-center deviation: lateral distance only (project onto waypoint normal)
        # This ignores along-lane progress so forward motion is not penalised.
        if waypoint is not None:
            wp_loc = waypoint.transform.location
            wp_yaw = math.radians(waypoint.transform.rotation.yaw)
            # Lane forward direction
            fwd_x, fwd_y = math.cos(wp_yaw), math.sin(wp_yaw)
            dx = ego_traj[-1, 0] - wp_loc.x
            dy = ego_traj[-1, 1] - wp_loc.y
            # Lateral component = cross product magnitude
            lat_err = abs(dx * (-fwd_y) + dy * fwd_x)
        else:
            lat_err = 0.0

        # Speed tracking: use throttle-vs-brake net to estimate end speed,
        # penalise deviation from target regardless of direction of approach
        mean_throttle = float(np.mean(actions[:, 0]))
        mean_brake    = float(np.mean(actions[:, 2]))
        # Rough final speed estimate from kinematics (accel_scale=4 m/s², same as simulator)
        estimated_speed = max(0.0, current_speed + (mean_throttle - mean_brake) * 4.0 * actions.shape[0] * 0.05)
        speed_err = abs(estimated_speed - cfg.TARGET_SPEED)

        # Jerk proxy: variance in throttle/brake actions across horizon
        jerk = float(np.std(actions[:, 0]) + np.std(actions[:, 2]))

        return (
            cfg.W_LANE_CENTER  * lat_err +
            cfg.W_SPEED_TRACK  * speed_err +
            cfg.W_COMFORT_JERK * jerk
        )


class CriticCost:
    def __init__(self, cfg: type = Configurator):
        self.cfg = cfg

    def compute_trajectory(
        self,
        ego_traj: np.ndarray,   # (H+1, 3): [x, y, yaw]
        predictions: dict,      # {actor_id: (H, 3) xyz}
    ) -> float:
        """
        For each timestep, compute minimum distance between ego and every actor.
        Apply exponential penalty for distances below DANGER_RADIUS.
        """
        cfg   = self.cfg
        total = 0.0

        ego_xy = ego_traj[1:, :2]   # (H, 2) — skip step 0 (current)

        for traj in predictions.values():
            # traj: (H, 3) — actor predicted positions
            H = min(ego_xy.shape[0], traj.shape[0])
            dists = np.linalg.norm(ego_xy[:H] - traj[:H, :2], axis=1)  # (H,)
            min_d = dists.min()

            if min_d < cfg.DANGER_RADIUS:
                # Exponential cost: peaks at 0 m, ~1.0 at COLLISION_RADIUS
                risk = math.exp(-min_d / max(cfg.COLLISION_RADIUS, 0.1))
                total += cfg.W_COLLISION * risk

        return total


class CostModule:
    def __init__(self, cfg: type = Configurator):
        self.intrinsic = IntrinsicCost(cfg)
        self.critic    = CriticCost(cfg)

    def trajectory_cost(
        self,
        ego_traj: np.ndarray,
        actions: np.ndarray,
        waypoint,
        current_speed: float,
        predictions: dict,
    ) -> float:
        return (
            self.intrinsic.compute_trajectory(ego_traj, actions, waypoint, current_speed) +
            self.critic.compute_trajectory(ego_traj, predictions)
        )


# ---------------------------------------------------------------------------
# 7. Actor Module
# ---------------------------------------------------------------------------

class Policy:
    """
    Samples N candidate action sequences (throttle, steer, brake).
    Nominal steer is derived from the next waypoint heading.
    """
    def __init__(self, cfg: type = Configurator):
        self.cfg = cfg

    def _nominal_steer(self, ego_frame: EgoFrame, waypoint) -> float:
        """Stanley-style heading error to waypoint."""
        if waypoint is None:
            return 0.0
        wp_yaw  = math.radians(waypoint.transform.rotation.yaw)
        ego_yaw = math.radians(ego_frame.transform.rotation.yaw)
        err = wp_yaw - ego_yaw
        # Normalise to [-pi, pi]
        err = (err + math.pi) % (2 * math.pi) - math.pi
        # Simple P-gain — clamp to [-1, 1]
        return float(np.clip(err * 0.6, -1.0, 1.0))

    def sample(self, ego_frame: EgoFrame, waypoint=None) -> np.ndarray:
        N = self.cfg.MPPI_SAMPLES
        H = self.cfg.PREDICTION_HORIZON
        σ = self.cfg.MPPI_NOISE_STD

        speed = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)
        error = self.cfg.TARGET_SPEED - speed
        raw   = self.cfg.SPEED_KP * error
        nominal_throttle = float(np.clip( raw, 0.0, 1.0))
        nominal_brake    = float(np.clip(-raw, 0.0, 1.0))
        nominal_steer    = self._nominal_steer(ego_frame, waypoint)

        base  = np.tile([nominal_throttle, nominal_steer, nominal_brake], (N, H, 1))

        # Wider steer noise to properly explore lateral avoidance manoeuvres
        noise = np.zeros((N, H, 3))
        noise[:, :, 0] = np.random.randn(N, H) * σ          # throttle
        noise[:, :, 1] = np.random.randn(N, H) * (σ * 3.0)  # steer — wider range
        noise[:, :, 2] = np.random.randn(N, H) * σ          # brake

        candidates = base + noise
        candidates[:, :, 0] = np.clip(candidates[:, :, 0], 0.0,  1.0)
        candidates[:, :, 1] = np.clip(candidates[:, :, 1], -1.0, 1.0)
        candidates[:, :, 2] = np.clip(candidates[:, :, 2], 0.0,  1.0)

        # Mutual exclusion: if brake > 0.1 zero throttle (and vice versa)
        heavy_brake = candidates[:, :, 2] > 0.1
        candidates[:, :, 0][heavy_brake] = 0.0
        return candidates


class Optimizer:
    """MPPI: evaluate each candidate's own simulated trajectory, then weight-average."""
    def __init__(self, cost_module: CostModule, cfg: type = Configurator):
        self.cost = cost_module
        self.cfg  = cfg

    def optimize(
        self,
        candidates: np.ndarray,    # (N, H, 3)
        ego_frame: EgoFrame,
        waypoint,
        predictions: dict,
    ) -> carla.VehicleControl:
        cfg   = self.cfg
        lam   = cfg.MPPI_LAMBDA
        dt    = cfg.FIXED_DELTA_SECONDS
        speed = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)

        costs = np.empty(len(candidates))
        for i, actions in enumerate(candidates):
            ego_traj = _simulate_ego_trajectory(ego_frame, actions, dt)
            costs[i] = self.cost.trajectory_cost(
                ego_traj, actions, waypoint, speed, predictions
            )

        beta    = costs.min()
        weights = np.exp(-(costs - beta) / lam)
        weights /= weights.sum()

        first_actions = candidates[:, 0, :]          # (N, 3)
        best_action   = (weights[:, None] * first_actions).sum(axis=0)

        throttle = float(np.clip(best_action[0], 0.0,  1.0))
        steer    = float(np.clip(best_action[1], -1.0, 1.0))
        brake    = float(np.clip(best_action[2], 0.0,  1.0))

        # Mutual exclusion at output: MPPI averaging always blends both non-zero.
        # CARLA physics treats any brake > 0 as holding the vehicle from rest.
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0

        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False)


# ---------------------------------------------------------------------------
# 8. Emergency Reactive Layer
# ---------------------------------------------------------------------------

def emergency_override(
    control: carla.VehicleControl,
    ego_frame: EgoFrame,
    actor_frames: list,
    cfg: type = Configurator,
) -> carla.VehicleControl:
    """
    If an obstacle is within EMERGENCY_DIST in the forward cone, override
    with hard brake and steer away. Bypasses MPPI for immediate response.
    """
    ego_yaw = math.radians(ego_frame.transform.rotation.yaw)
    ego_x   = ego_frame.location.x
    ego_y   = ego_frame.location.y
    half_cone = math.radians(cfg.EMERGENCY_CONE)

    closest_dist  = float("inf")
    steer_away    = 0.0

    for actor in actor_frames:
        dx = actor.location.x - ego_x
        dy = actor.location.y - ego_y
        dist = math.hypot(dx, dy)
        if dist > cfg.EMERGENCY_DIST or dist < 0.5:
            continue

        # Angle of obstacle relative to ego heading
        angle_to = math.atan2(dy, dx)
        rel_angle = (angle_to - ego_yaw + math.pi) % (2 * math.pi) - math.pi

        if abs(rel_angle) < half_cone:
            if dist < closest_dist:
                closest_dist = dist
                # Steer away from the obstacle's lateral side
                steer_away = -math.copysign(1.0, rel_angle) * min(1.0, (cfg.EMERGENCY_DIST - dist) / cfg.EMERGENCY_DIST * 2.0)

    if closest_dist < cfg.EMERGENCY_DIST:
        brake_strength = float(np.clip(1.0 - closest_dist / cfg.EMERGENCY_DIST, 0.3, 1.0))
        return carla.VehicleControl(
            throttle=0.0,
            steer=float(np.clip(steer_away * 0.7 + control.steer * 0.3, -1.0, 1.0)),
            brake=brake_strength,
            hand_brake=False,
        )

    return control


# ---------------------------------------------------------------------------
# 9. Actor
# ---------------------------------------------------------------------------

class Actor:
    def __init__(self, cost_module: CostModule, cfg: type = Configurator):
        self.policy    = Policy(cfg)
        self.optimizer = Optimizer(cost_module, cfg)
        self.cfg       = cfg

    def act(
        self,
        ego_frame: EgoFrame,
        waypoint,
        predictions: dict,
        actor_frames: list,
    ) -> carla.VehicleControl:
        candidates = self.policy.sample(ego_frame, waypoint)
        control    = self.optimizer.optimize(candidates, ego_frame, waypoint, predictions)
        control    = emergency_override(control, ego_frame, actor_frames, self.cfg)
        return control


# ---------------------------------------------------------------------------
# Helper: spawn NPC traffic
# ---------------------------------------------------------------------------

def spawn_npc_vehicles(client: carla.Client, world: carla.World, n: int) -> list:
    bp_lib    = world.get_blueprint_library()
    spawn_pts = world.get_map().get_spawn_points()
    random.shuffle(spawn_pts)

    vehicle_bps = bp_lib.filter("vehicle.*")
    batch = []
    for sp in spawn_pts[:n]:
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        batch.append(carla.command.SpawnActor(bp, sp)
                     .then(carla.command.SetAutopilot(carla.command.FutureActor, True)))

    spawned = []
    for resp in client.apply_batch_sync(batch, True):
        if not resp.error:
            spawned.append(resp.actor_id)
    print(f"[AD] Spawned {len(spawned)} NPC vehicles")
    return spawned


def spawn_npc_walkers(client: carla.Client, world: carla.World, n: int) -> tuple:
    bp_lib     = world.get_blueprint_library()
    walker_bps = bp_lib.filter("walker.pedestrian.*")
    ctrl_bp    = bp_lib.find("controller.ai.walker")

    spawn_locs = [world.get_random_location_from_navigation() for _ in range(n)]
    spawn_locs = [loc for loc in spawn_locs if loc is not None]

    batch      = [carla.command.SpawnActor(random.choice(walker_bps),
                                           carla.Transform(loc)) for loc in spawn_locs]
    walker_ids = [r.actor_id for r in client.apply_batch_sync(batch, True) if not r.error]

    walkers    = world.get_actors(walker_ids)
    ctrl_batch = [carla.command.SpawnActor(ctrl_bp, carla.Transform(), w) for w in walkers]
    ctrl_ids   = [r.actor_id for r in client.apply_batch_sync(ctrl_batch, True) if not r.error]

    world.tick()
    for ctrl in world.get_actors(ctrl_ids):
        ctrl.start()
        ctrl.go_to_location(world.get_random_location_from_navigation())
        ctrl.set_max_speed(1.0 + random.random())

    print(f"[AD] Spawned {len(walker_ids)} walkers")
    return walker_ids, ctrl_ids


# ---------------------------------------------------------------------------
# Helper: update spectator camera
# ---------------------------------------------------------------------------

def update_spectator(spectator: carla.Actor, ego_tf: carla.Transform, mode: str):
    loc = ego_tf.location
    yaw = ego_tf.rotation.yaw

    if mode == "bird":
        cam_tf = carla.Transform(
            carla.Location(x=loc.x, y=loc.y, z=loc.z + 30.0),
            carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
        )
    else:
        rad      = math.radians(yaw)
        offset_x = -8.0 * math.cos(rad)
        offset_y = -8.0 * math.sin(rad)
        cam_tf   = carla.Transform(
            carla.Location(x=loc.x + offset_x, y=loc.y + offset_y, z=loc.z + 4.0),
            carla.Rotation(pitch=-15.0, yaw=yaw, roll=0.0),
        )

    spectator.set_transform(cam_tf)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    cfg = Configurator

    client = carla.Client(cfg.HOST, cfg.PORT)
    client.set_timeout(cfg.TIMEOUT)
    world  = client.get_world()

    settings = world.get_settings()
    settings.synchronous_mode    = True
    settings.fixed_delta_seconds = cfg.FIXED_DELTA_SECONDS
    world.apply_settings(settings)

    tm = client.get_trafficmanager()
    tm.set_synchronous_mode(True)
    tm.set_global_distance_to_leading_vehicle(2.5)
    tm.global_percentage_speed_difference(-20)

    npc_vehicle_ids          = spawn_npc_vehicles(client, world, cfg.NPC_VEHICLES)
    npc_walker_ids, ctrl_ids = spawn_npc_walkers(client, world, cfg.NPC_WALKERS)

    bp_lib    = world.get_blueprint_library()
    ego_bp    = bp_lib.find(cfg.VEHICLE_BLUEPRINT)
    spawn_pts = world.get_map().get_spawn_points()
    random.shuffle(spawn_pts)

    ego = None
    for sp in spawn_pts:
        ego = world.try_spawn_actor(ego_bp, sp)
        if ego is not None:
            break
    if ego is None:
        raise RuntimeError("Could not spawn ego vehicle — all spawn points occupied.")
    print(f"[AD] Spawned ego vehicle id={ego.id} at {ego.get_location()}")

    spectator   = world.get_spectator()
    stm         = ShortTermMemory()
    perception  = Perception(world, ego, stm)
    world_model = WorldModel(stm)
    cost_module = CostModule(cfg)
    actor       = Actor(cost_module, cfg)

    try:
        while True:
            world.tick()

            perception.tick()
            ego_frame = stm.latest_ego()
            if ego_frame is None:
                continue

            update_spectator(spectator, ego_frame.transform, cfg.CAMERA_VIEW)

            predictions  = world_model.predict()
            waypoint     = perception.current_waypoint
            actor_frames = stm.latest_actors()
            control      = actor.act(ego_frame, waypoint, predictions, actor_frames)
            ego.apply_control(control)

            speed = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)
            print(
                f"[AD] speed={speed:.1f} m/s ({speed*3.6:.1f} km/h)  "
                f"throttle={control.throttle:.2f}  "
                f"brake={control.brake:.2f}  "
                f"steer={control.steer:.3f}  "
                f"actors_nearby={len(actor_frames)}"
            )

    finally:
        print("[AD] Cleaning up...")
        for ctrl in world.get_actors(ctrl_ids):
            ctrl.stop()

        all_ids = [ego.id] + npc_vehicle_ids + npc_walker_ids + ctrl_ids
        client.apply_batch_sync([carla.command.DestroyActor(aid) for aid in all_ids], True)

        settings.synchronous_mode = False
        world.apply_settings(settings)
        tm.set_synchronous_mode(False)
        print("[AD] Done.")


if __name__ == "__main__":
    main()
