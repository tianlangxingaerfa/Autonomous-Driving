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
    VEHICLE_BLUEPRINT  = "vehicle.tesla.model3"
    FREE_CRUISE_SPEED  = 20.0   # m/s — fallback ceiling when no pace controller is active
    MIN_FOLLOW_SPEED   = 0.0    # m/s — allow full stop behind a stopped actor
    CLEAR_HORIZON      = 30.0   # m — beyond this, ahead is considered clear
    CLEAR_CONE         = 35.0   # half-angle (deg) of forward clearance cone — wider catches side-clipping actors
    LANE_WIDTH_HALF    = 2.0    # m — actors beyond this lateral offset are treated as other-lane
    CLOSING_SPEED_GAIN = 1.0    # seconds of closing-speed lookahead
    SAFE_STOP_GAP      = 15.0   # m — start tapering speed toward 0 when stopped actor is within this range
    STOP_DISTANCE      = 8.0    # m — centroid-to-centroid target gap (~4.7 m vehicle + 3.3 m clear space)
    STOP_ACTOR_SPEED   = 1.0    # m/s — actor speed below which we treat it as stopped

    # Longitudinal control: throttle is capped at CRUISE_THROTTLE_MAX so the car
    # never lurches to full throttle regardless of how large the speed error is.
    # The rate limiter (MAX_ACCEL_RATE) then smooths even this capped value.
    CRUISE_THROTTLE_MAX = 0.45  # maximum allowed throttle during normal cruise
    SPEED_KP_ACCEL      = 0.08  # throttle ∝ (speed_ceiling - speed); capped above
    SPEED_KP_BRAKE      = 0.45  # brake ∝ (speed - speed_ceiling); higher = tighter stop

    # Control smoothing (EMA) — higher alpha = faster response (less lag)
    SMOOTH_ACCEL_ALPHA = 0.20
    SMOOTH_STEER_ALPHA = 0.45   # raised: EMA must keep up with the higher rate limit

    # Hard rate limits per tick (dt = 0.05 s, 20 Hz).
    #   MAX_ACCEL_RATE 0.020/tick × 20 Hz × 4 m/s² = 1.6 m/s³  (gentle jerk)
    #   MAX_BRAKE_RATE 0.08 /tick × 20 Hz            = much faster braking response
    #   MAX_STEER_RATE 0.120/tick × 20 Hz            = 2.4 steer-unit/s  (responsive for sharp turns)
    MAX_ACCEL_RATE = 0.020
    MAX_BRAKE_RATE = 0.080   # braking is asymmetrically faster than acceleration
    MAX_STEER_RATE = 0.120

    # Steering geometry
    STEER_K_CURVE   = 0.40
    STEER_K_YAW     = 0.50
    STEER_K_CTE     = 1.20   # stronger centering — was 0.80; rate limiter above allows it
    STEER_SPEED_EPS = 2.0
    STANLEY_LOOKAHEAD = 10.0

    # Curve speed limit: when the road-curvature yaw-change over the lookahead
    # distance exceeds CURVE_YAW_THRESH (rad), cap speed to CURVE_SPEED_MAX.
    # This reduces lateral inertia so the CTE correction can hold the lane on turns.
    CURVE_YAW_THRESH = 0.12   # rad (~7°) over 10 m lookahead — any noticeable bend
    CURVE_SPEED_MAX  = 7.0    # m/s (~25 km/h) on curves

    # STM
    STM_BUFFER_LEN = 20

    # World model
    PREDICTION_HORIZON = 15   # steps to predict forward (~0.75 s)

    # Cost weights
    W_LANE_CENTER  = 5.0   # raised: penalise lateral drift throughout the trajectory
    W_SPEED_TRACK  = 2.0   # strong pull toward target speed
    W_COMFORT_JERK = 1.0
    W_COLLISION    = 20.0  # avoidance priority, but not overwhelming
    W_TRAFFIC_RULE = 5.0

    # Collision geometry
    COLLISION_RADIUS = 2.0     # m — soft margin around ego centroid
    DANGER_RADIUS    = 4.0     # m — within this → exponential cost

    # Camera view mode: "third_person" or "bird"
    CAMERA_VIEW = "third_person"

    # Traffic
    NPC_VEHICLES = 40
    NPC_WALKERS  = 50

    # Optimizer (MPPI)
    MPPI_LAMBDA          = 0.5   # temperature: lower = sharper selection

    # Structured candidate grid (replaces random noise).
    # On a clear road: single deterministic action (no grid needed).
    # Near obstacles: throttle/brake varied over a grid of N_THROTTLE × N_BRAKE
    # values and steer held at the nominal value.  This removes tick-to-tick
    # randomness — same state always produces the same candidate set.
    MPPI_N_THROTTLE = 9    # throttle levels: linspace(0, CRUISE_THROTTLE_MAX, N)
    MPPI_N_BRAKE    = 9    # brake levels:    linspace(0, 1, N)
    # Total candidates = N_THROTTLE × N_BRAKE = 81 (fast, deterministic)

    # Emergency reactive layer
    EMERGENCY_DIST  = 6.0      # m — trigger hard brake + steer
    EMERGENCY_CONE  = 35.0     # half-angle (deg) — matches CLEAR_CONE so side-clip actors are caught

    # Clear-road threshold: if no actor is within this distance in the forward cone,
    # skip MPPI entirely and apply the deterministic nominal action directly.
    # This eliminates all stochastic shiver on an unobstructed road.
    CLEAR_ROAD_DIST = 20.0     # m

    # Traffic-light compliance
    TL_STOP_LINE_DIST    = 60.0   # m — how far ahead to look for a controlling light
    TL_YELLOW_SPEED_CAP  = 3.0    # m/s — final creep speed on yellow
    TL_TAPER_START       = 50.0   # m from stop point — begin smooth taper
    TL_STOP_BUFFER       = 1.0    # m — centroid rests this far before the stop-waypoint
    TL_RIGHT_TURN_YAW    = 0.40   # rad (~23°) signed yaw change → treat as right turn, skip TL

    # Lap-pace controller
    # The pace controller computes a desired cruise speed = remaining_dist / remaining_time
    # so the car arrives at the lap end within LAP_TARGET_TIME seconds regardless of
    # how long it was held behind traffic or at red lights.
    # Hard bounds keep the target inside a safe range; safety layers (obstacles,
    # traffic lights, curves, emergency) are always applied on top of this.
    LAP_DISTANCE     = 700.0   # m — estimated circuit length (tune per map)
    LAP_TARGET_TIME  = 45.0   # s — desired lap duration
    PACE_SPEED_MIN   = 6.0     # m/s — never command below this on a clear road
    PACE_SPEED_MAX   = 22.0    # m/s — absolute speed ceiling the pace controller may request
    LAP_CLOSE_RADIUS = 5.0    # m — within this radius of spawn → lap is closed

    # Lane change
    LC_BEHIND_CLEAR  = 6.0   # m — min gap behind in target lane before changing
    LC_AHEAD_CLEAR   = 6.0   # m — min gap ahead  in target lane before changing
    LC_SLOW_THRESH   = 0.70   # fraction of pace target — lane-ahead actor below this triggers LC desire
    LC_COMMIT_TICKS  = 60     # ticks (~3 s) to hold the new lane before re-evaluating
    LC_BLOCKED_TICKS = 40     # ticks to wait after a blocked attempt before retrying


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
    def __init__(self, actor_id, location, velocity, bounding_box, is_vehicle: bool = True):
        self.actor_id     = actor_id
        self.location     = location
        self.velocity     = velocity
        self.bounding_box = bounding_box
        self.is_vehicle   = is_vehicle


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
    """
    Perception tick.

    Waypoint tracking strategy (fixes passive lane-change bug):
      carla.Map.get_waypoint(location) re-snaps to the geometrically nearest lane
      every tick.  When the ego drifts toward a lane boundary, the snap jumps to the
      adjacent lane, inverts the CTE sign, and drives the car further into that lane
      — a positive-feedback oscillation that eventually produces an unintended lane
      change.

      Instead we maintain a *tracked* waypoint that only advances *forward* along the
      current lane's road graph.  We advance it whenever the ego passes beyond its
      position (dot-product test), and we never jump laterally.  A lateral re-anchor
      is only allowed on a hard reset (vehicle teleport / first tick).
    """

    # How far ahead of the current waypoint the lookahead target should sit.
    _LOOKAHEAD_DIST = Configurator.STANLEY_LOOKAHEAD
    # Advance the tracked waypoint in small steps along the graph.
    _ADVANCE_STEP   = 1.5   # m per graph walk step — finer for smoother curves
    # How far behind the ego the tracked waypoint is allowed to lag.
    # Keep small so CTE is measured against the correct lane direction after a turn.
    _MAX_LAG        = 2.0   # m

    def __init__(self, world: carla.World, ego_vehicle: carla.Vehicle, stm: ShortTermMemory):
        self.world            = world
        self.ego_vehicle      = ego_vehicle
        self.stm              = stm
        self.carla_map        = world.get_map()
        self.current_waypoint  = None
        self.lookahead_waypoint = None
        self._tracked_wp       = None   # forward-only tracked waypoint
        self._approach_wp      = None   # last non-junction waypoint — used for TL matching
        self._pinned_stop_loc  = None   # committed stop-line XY (set once, cleared on green)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _advance_tracker(self, ego_x: float, ego_y: float):
        """
        Walk the tracked waypoint forward along the lane graph until it is
        at most _MAX_LAG metres behind the ego.

        Also maintains _approach_wp: the last waypoint that was NOT inside a
        junction.  Its road_id / lane_id are used for traffic-light matching so
        that the correct stop-waypoints are found even after the ego has already
        entered the intersection.
        """
        wp = self._tracked_wp
        if wp is None:
            return

        for _ in range(80):
            wp_x  = wp.transform.location.x
            wp_y  = wp.transform.location.y
            fwd_x = math.cos(math.radians(wp.transform.rotation.yaw))
            fwd_y = math.sin(math.radians(wp.transform.rotation.yaw))
            along = (ego_x - wp_x) * fwd_x + (ego_y - wp_y) * fwd_y
            if along < self._MAX_LAG:
                break
            nexts = wp.next(self._ADVANCE_STEP)
            if not nexts:
                break
            if not wp.is_junction:
                same = [n for n in nexts
                        if n.road_id == wp.road_id and n.lane_id == wp.lane_id]
                wp = same[0] if same else nexts[0]
            else:
                wp = nexts[0]

        self._tracked_wp = wp
        # Keep _approach_wp pointing at the last non-junction waypoint.
        if not wp.is_junction:
            self._approach_wp = wp
        elif self._approach_wp is None:
            self._approach_wp = wp

    def _build_lookahead(self) -> "carla.Waypoint":
        """
        Walk from the tracked waypoint forward _LOOKAHEAD_DIST m.
        Follows the road graph freely through junctions so the heading reference
        curves along the turn arc rather than freezing at the junction entry.
        """
        wp = self._tracked_wp
        if wp is None:
            return None
        remaining = self._LOOKAHEAD_DIST
        while remaining > 0.0:
            nexts = wp.next(min(remaining, self._ADVANCE_STEP))
            if not nexts:
                break
            # Same logic: stay on lane on normal road, free walk inside junction
            if not wp.is_junction:
                same = [n for n in nexts
                        if n.road_id == wp.road_id and n.lane_id == wp.lane_id]
                wp = same[0] if same else nexts[0]
            else:
                wp = nexts[0]
            remaining -= self._ADVANCE_STEP
        return wp

    # ------------------------------------------------------------------
    # Public tick
    # ------------------------------------------------------------------

    def tick(self):
        ego = self.ego_vehicle
        tf  = ego.get_transform()
        loc = tf.location
        vel = ego.get_velocity()
        acc = ego.get_acceleration()

        ego_frame = EgoFrame(loc, vel, acc, tf)

        # ---- Traffic-light state ----
        tl_state = carla.TrafficLightState.Unknown
        tl_dist  = float("inf")   # distance to the relevant stop-line (m)

        # Helper: find the stop-waypoint distance on the ego's lane for a given TL.
        def _stop_dist_for_tl(tl, road_id, lane_id):
            best = float("inf")
            best_loc = None
            for swp in tl.get_stop_waypoints():
                if swp.road_id != road_id or swp.lane_id != lane_id:
                    continue
                d = math.hypot(swp.transform.location.x - loc.x,
                               swp.transform.location.y - loc.y)
                if d < best:
                    best = d
                    best_loc = swp.transform.location
            return best, best_loc

        if ego.is_at_traffic_light():
            tl = ego.get_traffic_light()
            if tl is not None:
                tl_state = tl.get_state()
                ref_wp   = self._approach_wp or self._tracked_wp
                ego_road = ref_wp.road_id if ref_wp else -1
                ego_lane = ref_wp.lane_id if ref_wp else 0

                if tl_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                    # Pin the stop location on the first red/yellow tick.
                    if self._pinned_stop_loc is None:
                        raw_d, raw_loc = _stop_dist_for_tl(tl, ego_road, ego_lane)
                        if raw_loc is not None:
                            self._pinned_stop_loc = raw_loc
                else:
                    self._pinned_stop_loc = None

                if self._pinned_stop_loc is not None:
                    tl_dist = math.hypot(self._pinned_stop_loc.x - loc.x,
                                         self._pinned_stop_loc.y - loc.y)
                else:
                    raw_d, _ = _stop_dist_for_tl(tl, ego_road, ego_lane)
                    tl_dist = raw_d if raw_d < float("inf") else 0.0
        elif self._approach_wp is not None:
            # Wide look-ahead scan: use the approach road+lane for matching so
            # left-turn pre-junction detection works correctly.
            ref_wp    = self._approach_wp
            ego_road  = ref_wp.road_id
            ego_lane  = ref_wp.lane_id
            ego_yaw   = math.radians(tf.rotation.yaw)
            fwd_x     = math.cos(ego_yaw)
            fwd_y     = math.sin(ego_yaw)

            best_dist      = Configurator.TL_STOP_LINE_DIST
            best_state     = carla.TrafficLightState.Unknown
            best_stop_dist = float("inf")
            best_stop_loc  = None

            for tl in self.world.get_actors().filter("traffic.traffic_light*"):
                tl_loc = tl.get_location()
                dx = tl_loc.x - loc.x
                dy = tl_loc.y - loc.y
                dist = math.hypot(dx, dy)
                if dist >= best_dist:
                    continue
                if dx * fwd_x + dy * fwd_y <= 0:
                    continue
                raw_d, raw_loc = _stop_dist_for_tl(tl, ego_road, ego_lane)
                if raw_loc is None:
                    continue
                best_dist      = dist
                best_state     = tl.get_state()
                best_stop_dist = raw_d
                best_stop_loc  = raw_loc

            tl_state = best_state
            if best_state in (carla.TrafficLightState.Red, carla.TrafficLightState.Yellow):
                # Pin as soon as we detect a red/yellow ahead.
                if self._pinned_stop_loc is None and best_stop_loc is not None:
                    self._pinned_stop_loc = best_stop_loc
                if self._pinned_stop_loc is not None:
                    tl_dist = math.hypot(self._pinned_stop_loc.x - loc.x,
                                         self._pinned_stop_loc.y - loc.y)
                else:
                    tl_dist = best_stop_dist
            else:
                self._pinned_stop_loc = None
                tl_dist = float("inf")

        ego_frame.traffic_light_state = tl_state
        ego_frame.traffic_light_dist  = tl_dist

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
            actor_list.append(ActorFrame(actor.id, a_loc, a_vel, a_bb,
                                         is_vehicle=isinstance(actor, carla.Vehicle)))

        # ---- Waypoint tracking ----
        if self._tracked_wp is None:
            # First tick or hard reset: anchor to the actual nearest waypoint.
            self._tracked_wp = self.carla_map.get_waypoint(loc)

        self._advance_tracker(loc.x, loc.y)
        self.current_waypoint   = self._tracked_wp
        self.lookahead_waypoint = self._build_lookahead()

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
        desired_speed: float,
        tl_must_stop: bool = False,  # True when ego is obligated to stop (red/yellow)
    ) -> float:
        cfg = self.cfg

        # Lane-center deviation: average lateral error across every predicted step.
        # Using only the endpoint allowed mid-trajectory drift at zero cost.
        if waypoint is not None:
            wp_loc = waypoint.transform.location
            wp_yaw = math.radians(waypoint.transform.rotation.yaw)
            fwd_x, fwd_y   = math.cos(wp_yaw), math.sin(wp_yaw)
            lat_norm_x = -fwd_y
            lat_norm_y =  fwd_x
            # Accumulate lateral error for every step after the initial one
            total_lat = 0.0
            for step in range(1, ego_traj.shape[0]):
                dx = ego_traj[step, 0] - wp_loc.x
                dy = ego_traj[step, 1] - wp_loc.y
                total_lat += abs(dx * lat_norm_x + dy * lat_norm_y)
            lat_err = total_lat / max(ego_traj.shape[0] - 1, 1)
        else:
            lat_err = 0.0

        # Speed tracking against dynamic desired_speed
        mean_throttle = float(np.mean(actions[:, 0]))
        mean_brake    = float(np.mean(actions[:, 2]))
        estimated_speed = max(0.0, current_speed + (mean_throttle - mean_brake) * 4.0 * actions.shape[0] * 0.05)
        speed_err = abs(estimated_speed - desired_speed) 

        # Jerk: std of net acceleration (throttle - brake) across the horizon.
        # This directly penalises abrupt changes in longitudinal force.
        net_accel = actions[:, 0] - actions[:, 2]
        jerk = float(np.std(net_accel))

        # Traffic-rule penalty: when a red/yellow light requires stopping, any
        # candidate that still carries throttle is penalised proportionally.
        # This makes the cost-minimising candidate always prefer brake over throttle
        # at a red light, reinforcing the hard speed-ceiling applied upstream.
        traffic_penalty = 0.0
        if tl_must_stop:
            mean_throttle_penalty = float(np.mean(actions[:, 0]))
            traffic_penalty = cfg.W_TRAFFIC_RULE * mean_throttle_penalty

        return (
            cfg.W_LANE_CENTER  * lat_err +
            cfg.W_SPEED_TRACK  * speed_err +
            cfg.W_COMFORT_JERK * jerk +
            traffic_penalty
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
        Apply exponential penalty for distances below DANGER_RADIUS, scaled by
        a directional weight: forward actors (cos²θ near 1) cost up to 2×;
        pure-side actors (cosθ near 0) cost 0.25× — reduces lateral overreaction.
        """
        cfg    = self.cfg
        total  = 0.0

        ego_x0, ego_y0, ego_yaw0 = ego_traj[0]
        ego_xy = ego_traj[1:, :2]   # (H, 2) — skip step 0 (current)

        for traj in predictions.values():
            H = min(ego_xy.shape[0], traj.shape[0])
            dists = np.linalg.norm(ego_xy[:H] - traj[:H, :2], axis=1)  # (H,)
            min_d = dists.min()

            if min_d < cfg.DANGER_RADIUS:
                # Directional weight: project actor bearing onto ego forward axis.
                # cos²θ gives 1.0 dead-ahead, 0.0 pure-side, smooth and always ≥ 0.
                # We remap to [0.25, 2.0] so side actors still register but matter less.
                ax, ay = traj[0, 0], traj[0, 1]
                dx, dy = ax - ego_x0, ay - ego_y0
                dist0  = math.hypot(dx, dy)
                if dist0 > 0.1:
                    cos_sq = ((dx * math.cos(ego_yaw0) + dy * math.sin(ego_yaw0)) / dist0) ** 2
                else:
                    cos_sq = 1.0
                dir_weight = 0.1 + 1.9 * cos_sq  # range [0.1, 2.0] — side actors nearly ignored

                risk = math.exp(-min_d / max(cfg.COLLISION_RADIUS, 0.1))
                total += cfg.W_COLLISION * risk * dir_weight

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
        desired_speed: float,
        predictions: dict,
        tl_must_stop: bool = False,
    ) -> float:
        return (
            self.intrinsic.compute_trajectory(
                ego_traj, actions, waypoint, current_speed, desired_speed, tl_must_stop) +
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

    def _nominal_steer(self, ego_frame: EgoFrame, waypoint, lookahead_wp=None) -> float:
        """
        Three-term decomposed steering — only steers when the road actually demands it.

        Term 1 — Road curvature (feedforward):
          Δyaw = (lookahead_wp.yaw − current_wp.yaw) normalised to [−π, π].
          This is the arc the road will turn through over the lookahead distance.
          On a perfectly straight road Δyaw≈0, so this term outputs ≈0 with no
          ego-yaw noise feeding through.

        Term 2 — Heading alignment (small correction):
          Compares ego yaw to the *current* waypoint yaw.  Uses the on-graph waypoint,
          not the ego itself, so map-yaw noise is ≪ ego IMU noise.  Small gain keeps
          corrections gentle.

        Term 3 — Cross-track centering (tiny restoring force):
          Signed lateral offset of ego from the current waypoint centre-line.
          Very small gain + speed softening → near-zero on a straight with small CTE.
          The rate limiter in ControlSmoother prevents this from oscillating.
        """
        if waypoint is None:
            return 0.0
        cfg = self.cfg

        # ── Term 1: feedforward curvature ──────────────────────────────────
        ref_wp = lookahead_wp if lookahead_wp is not None else waypoint
        wp_yaw_cur  = math.radians(waypoint.transform.rotation.yaw)
        wp_yaw_look = math.radians(ref_wp.transform.rotation.yaw)
        road_curve  = (wp_yaw_look - wp_yaw_cur + math.pi) % (2 * math.pi) - math.pi
        curvature_steer = cfg.STEER_K_CURVE * road_curve

        # ── Term 2: heading alignment ───────────────────────────────────────
        ego_yaw = math.radians(ego_frame.transform.rotation.yaw)
        heading_err = (wp_yaw_cur - ego_yaw + math.pi) % (2 * math.pi) - math.pi
        heading_steer = cfg.STEER_K_YAW * heading_err

        # ── Term 3: cross-track centering ───────────────────────────────────
        wp_loc = waypoint.transform.location
        # CARLA left-handed (Y south). Right-of-heading at yaw θ: (−sin θ, +cos θ).
        # Verify: θ=0 (East) → (0, +1) = South = right ✓; θ=90 (South) → (−1, 0) = West = right ✓
        # Left-of-heading = sign-flip = (+sin θ, −cos θ).
        # cte > 0 → ego is LEFT of centre → need positive steer (right) to return ✓
        lat_nx =  math.sin(wp_yaw_cur)
        lat_ny = -math.cos(wp_yaw_cur)
        dx = ego_frame.location.x - wp_loc.x
        dy = ego_frame.location.y - wp_loc.y
        cte = dx * lat_nx + dy * lat_ny
        v   = max(0.0, math.hypot(ego_frame.velocity.x, ego_frame.velocity.y))
        cte_steer = math.atan2(cfg.STEER_K_CTE * cte, v + cfg.STEER_SPEED_EPS)

        raw = curvature_steer + heading_steer + cte_steer
        return float(np.clip(raw, -1.0, 1.0))

    def sample(self, ego_frame: EgoFrame, waypoint=None, speed_ceiling: float = None,
               lookahead_wp=None) -> np.ndarray:
        """
        Build a deterministic structured candidate set.

        The nominal steer is always used for every candidate — steering is handled
        by the deterministic controller, not by the optimiser.  Only throttle and
        brake are varied, over a fixed grid so the same state always produces
        identical candidates (no tick-to-tick randomness).

        Grid: N_THROTTLE throttle levels × N_BRAKE brake levels, replicated over
        the prediction horizon H.  Candidates where both throttle and brake are
        high are pruned (mutual exclusion) to avoid physically nonsensical actions.
        """
        cfg = self.cfg
        H   = cfg.PREDICTION_HORIZON

        if speed_ceiling is None:
            speed_ceiling = cfg.FREE_CRUISE_SPEED

        nominal_steer = self._nominal_steer(ego_frame, waypoint, lookahead_wp)

        throttle_vals = np.linspace(0.0, cfg.CRUISE_THROTTLE_MAX, cfg.MPPI_N_THROTTLE)
        brake_vals    = np.linspace(0.0, 1.0,                      cfg.MPPI_N_BRAKE)

        rows = []
        for t in throttle_vals:
            for b in brake_vals:
                if t > 0.05 and b > 0.1:
                    continue   # mutually exclusive — skip
                # Each candidate holds this action constant across the horizon
                action = np.array([t, nominal_steer, b], dtype=np.float32)
                rows.append(np.tile(action, (H, 1)))

        candidates = np.stack(rows, axis=0)   # (N, H, 3)
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
        speed_ceiling: float,
        tl_must_stop: bool = False,
    ) -> carla.VehicleControl:
        cfg   = self.cfg
        lam   = cfg.MPPI_LAMBDA
        dt    = cfg.FIXED_DELTA_SECONDS
        speed = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)

        costs = np.empty(len(candidates))
        for i, actions in enumerate(candidates):
            ego_traj = _simulate_ego_trajectory(ego_frame, actions, dt)
            costs[i] = self.cost.trajectory_cost(
                ego_traj, actions, waypoint, speed, speed_ceiling, predictions, tl_must_stop
            )

        beta    = costs.min()
        weights = np.exp(-(costs - beta) / lam)
        weights /= weights.sum()

        first_actions = candidates[:, 0, :]
        best_action   = (weights[:, None] * first_actions).sum(axis=0)

        throttle = float(np.clip(best_action[0], 0.0, cfg.CRUISE_THROTTLE_MAX))
        steer    = float(np.clip(best_action[1], -1.0, 1.0))
        brake    = float(np.clip(best_action[2], 0.0,  1.0))

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
    Hard reactive layer for imminent collisions inside EMERGENCY_DIST.

    Directional response policy:
      - Forward obstacle (|rel_angle| < 15°): brake hard + light steer-away blend.
      - Lateral obstacle (15° ≤ |rel_angle| < EMERGENCY_CONE): brake only — steering
        into a side-actor tends to make things worse; slowing down creates separation.
      - The steer-away magnitude scales with the forward fraction (cos²) so the
        transition is smooth across the cone rather than a binary switch.
    """
    ego_yaw   = math.radians(ego_frame.transform.rotation.yaw)
    ego_x     = ego_frame.location.x
    ego_y     = ego_frame.location.y
    half_cone = math.radians(cfg.EMERGENCY_CONE)

    closest_dist  = float("inf")
    steer_away    = 0.0
    steer_weight  = 0.0   # how much steer-away to mix in (0 for pure-lateral threats)

    for actor in actor_frames:
        dx = actor.location.x - ego_x
        dy = actor.location.y - ego_y
        dist = math.hypot(dx, dy)
        if dist > cfg.EMERGENCY_DIST or dist < 0.5:
            continue

        angle_to  = math.atan2(dy, dx)
        rel_angle = (angle_to - ego_yaw + math.pi) % (2 * math.pi) - math.pi

        if abs(rel_angle) >= half_cone:
            continue

        if dist < closest_dist:
            closest_dist = dist

            # cos² gives 1.0 dead-ahead, 0.0 pure-side; remap to [0, 1] steer weight
            cos_sq = math.cos(rel_angle) ** 2
            steer_weight = cos_sq   # pure lateral → 0, pure forward → 1

            # Steer-away direction (only applied proportionally to steer_weight)
            candidate_steer = -math.copysign(1.0, rel_angle) * min(
                1.0, (cfg.EMERGENCY_DIST - dist) / cfg.EMERGENCY_DIST * 2.0
            )
            steer_away = candidate_steer

    if closest_dist < cfg.EMERGENCY_DIST:
        brake_strength = float(np.clip(1.0 - closest_dist / cfg.EMERGENCY_DIST, 0.3, 1.0))
        # Blend steer-away only for forward-ish threats; lateral threats get 0 steer change
        blended_steer = float(np.clip(
            steer_weight * (steer_away * 0.5 + control.steer * 0.5) +
            (1.0 - steer_weight) * control.steer,
            -1.0, 1.0,
        ))
        return carla.VehicleControl(
            throttle=0.0,
            steer=blended_steer,
            brake=brake_strength,
            hand_brake=False,
        )

    return control


# ---------------------------------------------------------------------------
# 9. Control Smoother
# ---------------------------------------------------------------------------

class ControlSmoother:
    """
    Two-stage smoothing pipeline:
      1. Exponential moving average (EMA) — removes high-frequency noise.
      2. Hard rate limiter — caps the per-tick change after EMA so the signal
         never jumps faster than a physical jerk budget allows.
         Braking uses a separate (faster) rate limit so late-detected stops
         can react quickly without making normal acceleration jerky.

    Emergency signals bypass the EMA and snap state, preventing windup.
    """
    def __init__(self, alpha_accel: float, alpha_steer: float,
                 max_accel_rate: float, max_steer_rate: float,
                 max_brake_rate: float = None):
        self.alpha_accel    = alpha_accel
        self.alpha_steer    = alpha_steer
        self.max_accel_rate = max_accel_rate
        self.max_brake_rate = max_brake_rate if max_brake_rate is not None else max_accel_rate
        self.max_steer_rate = max_steer_rate
        self._net_accel     = 0.0
        self._steer         = 0.0

    def smooth(
        self,
        control: carla.VehicleControl,
        emergency: bool = False,
    ) -> carla.VehicleControl:
        raw_net = control.throttle - control.brake

        if emergency:
            self._net_accel = raw_net
            self._steer     = control.steer
            return control

        # --- Stage 1: EMA ---
        ema_net   = self.alpha_accel * raw_net       + (1.0 - self.alpha_accel) * self._net_accel
        ema_steer = self.alpha_steer * control.steer + (1.0 - self.alpha_steer) * self._steer

        # --- Stage 2: asymmetric rate limiter ---
        # Braking (net going more negative) uses max_brake_rate for fast response.
        # Accelerating (net going more positive) uses max_accel_rate for smooth onset.
        if ema_net < self._net_accel:
            rate = self.max_brake_rate
        else:
            rate = self.max_accel_rate
        net   = float(np.clip(ema_net,
                              self._net_accel - rate,
                              self._net_accel + self.max_accel_rate))
        steer = float(np.clip(ema_steer,
                              self._steer - self.max_steer_rate,
                              self._steer + self.max_steer_rate))

        # Commit the rate-limited values as new EMA state so the next step's
        # rate window is anchored to what we actually sent, not what EMA wanted.
        self._net_accel = net
        self._steer     = steer

        net   = float(np.clip(net,   -1.0, 1.0))
        steer = float(np.clip(steer, -1.0, 1.0))

        throttle = float(np.clip( net, 0.0, 1.0))
        brake    = float(np.clip(-net, 0.0, 1.0))
        return carla.VehicleControl(throttle=throttle, steer=steer, brake=brake, hand_brake=False)




# ---------------------------------------------------------------------------
# 10. Lap Pace Controller
# ---------------------------------------------------------------------------

class LapPaceController:
    """
    Time-to-target cruise speed governor.

    Each tick it computes:

        pace_speed = remaining_distance / remaining_time

    clamped to [PACE_SPEED_MIN, PACE_SPEED_MAX].  Safety layers (obstacles,
    TL, curves, emergency) always take priority — this only sets the *desired*
    open-road ceiling.

    Odometry: we integrate velocity magnitude every tick (dt = FIXED_DELTA_SECONDS).
    Lap closure: when the ego comes within LAP_CLOSE_RADIUS of the spawn location
    and has already travelled > LAP_DISTANCE / 2 (so the start-line crossing at
    the very beginning doesn't count), the lap counter increments, elapsed/odometer
    reset, and a new lap begins.

    The pace_speed smoothly rises when the car falls behind schedule (remaining
    dist / remaining time > current speed) and falls when it is ahead of schedule,
    so it acts like a gentle I-term that absorbs stops at traffic lights without
    any discontinuous jumps.
    """

    def __init__(self, spawn_location, cfg: type = Configurator,
                 dt: float = Configurator.FIXED_DELTA_SECONDS):
        self.cfg            = cfg
        self.dt             = dt
        self.spawn_loc      = spawn_location   # carla.Location at start of first lap
        self.lap_count      = 0
        self._odometer      = 0.0   # metres travelled this lap
        self._elapsed       = 0.0   # seconds elapsed this lap
        self._lap_started   = False  # True once we have moved away from spawn
        self._away          = False  # True once we are > LAP_CLOSE_RADIUS from spawn

    # ------------------------------------------------------------------

    def tick(self, ego_frame: EgoFrame) -> float:
        """
        Update state and return the desired cruise speed (m/s) for this tick.
        """
        cfg  = self.cfg
        speed = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)

        # Accumulate odometry and time.
        self._odometer += speed * self.dt
        self._elapsed  += self.dt

        # Lap-closure detection.
        dist_to_spawn = math.hypot(
            ego_frame.location.x - self.spawn_loc.x,
            ego_frame.location.y - self.spawn_loc.y,
        )
        if not self._away and dist_to_spawn > cfg.LAP_CLOSE_RADIUS:
            self._away = True   # we have left the start zone
        if self._away and dist_to_spawn <= cfg.LAP_CLOSE_RADIUS:
            # Only close the lap once we have covered at least half the expected
            # circuit — guards against a false trigger at the very first tick.
            if self._odometer > cfg.LAP_DISTANCE * 0.5:
                self.lap_count  += 1
                lap_time         = self._elapsed
                print(f"[Pace] Lap {self.lap_count} complete — "
                      f"{lap_time:.1f} s  "
                      f"(target {cfg.LAP_TARGET_TIME:.0f} s)  "
                      f"dist={self._odometer:.0f} m")
                self._odometer = 0.0
                self._elapsed  = 0.0
                self._away     = False

        # Compute pace speed.
        remaining_dist = max(0.0, cfg.LAP_DISTANCE - self._odometer)
        remaining_time = max(1.0, cfg.LAP_TARGET_TIME - self._elapsed)
        pace = remaining_dist / remaining_time

        return float(np.clip(pace, cfg.PACE_SPEED_MIN, cfg.PACE_SPEED_MAX))

    @property
    def odometer(self) -> float:
        return self._odometer

    @property
    def elapsed(self) -> float:
        return self._elapsed


# ---------------------------------------------------------------------------
# 11. Lane Changer
# ---------------------------------------------------------------------------

class LaneChanger:
    """
    Opportunistic lane-change logic.

    Every tick the changer asks:
      1. Is the current lane blocked by a slow *vehicle* ahead?
      2. Is there an adjacent *same-direction* Driving lane whose waypoint
         permits a lane change (lane_change flag) and that is not a junction?
      3. Is that lane clear of *vehicles* (LC_BEHIND_CLEAR behind,
         LC_AHEAD_CLEAR ahead)?

    Walkers and cyclists are deliberately ignored — the emergency-brake layer
    handles them; a lane change to dodge a pedestrian is unsafe and forbidden.

    Left lane is tried first (faster overtake); right is the fallback.

    Cooldowns prevent thrashing:
      • After a successful change: LC_COMMIT_TICKS before re-evaluating.
      • After a blocked attempt:   LC_BLOCKED_TICKS before retrying.

    The changer never acts:
      • Inside a junction.
      • When a traffic light requires stopping.
      • When the road-curve exceeds CURVE_YAW_THRESH.
    """

    def __init__(self, carla_map, cfg: type = Configurator):
        self.map       = carla_map
        self.cfg       = cfg
        self._cooldown = 0

    def _lane_clear(self, target_wp, actor_frames: list,
                    ego_x: float, ego_y: float, ego_yaw: float) -> bool:
        """Return True if target lane has enough vehicle gap ahead and behind."""
        cfg     = self.cfg
        fwd_x   = math.cos(ego_yaw)
        fwd_y   = math.sin(ego_yaw)
        tgt_yaw = math.radians(target_wp.transform.rotation.yaw)
        lat_nx  = -math.sin(tgt_yaw)
        lat_ny  =  math.cos(tgt_yaw)
        tgt_x   = target_wp.transform.location.x
        tgt_y   = target_wp.transform.location.y

        for actor in actor_frames:
            if not actor.is_vehicle:
                continue   # walkers never block a lane-change clearance check
            adx = actor.location.x - tgt_x
            ady = actor.location.y - tgt_y
            if abs(adx * lat_nx + ady * lat_ny) > cfg.LANE_WIDTH_HALF:
                continue   # not in target lane
            rdx   = actor.location.x - ego_x
            rdy   = actor.location.y - ego_y
            along = rdx * fwd_x + rdy * fwd_y
            if 0 < along < cfg.LC_AHEAD_CLEAR:
                return False
            if along <= 0 and abs(along) < cfg.LC_BEHIND_CLEAR:
                return False
        return True

    @staticmethod
    def _same_direction(current_wp, neighbor_wp) -> bool:
        """True when neighbor lane runs in the same traffic direction as current."""
        # CARLA sign convention: positive lane_id = right of road centre,
        # negative = left.  Lanes on the same side of the road (same sign)
        # are same-direction; opposite signs mean counter-traffic.
        return (current_wp.lane_id > 0) == (neighbor_wp.lane_id > 0)

    def tick(
        self,
        perception,
        ego_frame: EgoFrame,
        actor_frames: list,
        pace_speed: float,
        tl_must_stop: bool,
        road_curve: float,
    ) -> bool:
        cfg = self.cfg

        if self._cooldown > 0:
            self._cooldown -= 1
            return False

        current_wp = perception.current_waypoint
        if current_wp is None or current_wp.is_junction:
            return False
        if tl_must_stop:
            return False
        if road_curve > cfg.CURVE_YAW_THRESH:
            return False

        ego_x   = ego_frame.location.x
        ego_y   = ego_frame.location.y
        ego_yaw = math.radians(ego_frame.transform.rotation.yaw)
        fwd_x   = math.cos(ego_yaw)
        fwd_y   = math.sin(ego_yaw)

        # ── Is current lane blocked by a slow vehicle? ───────────────────
        wp_yaw = math.radians(current_wp.transform.rotation.yaw)
        lat_nx = -math.sin(wp_yaw)
        lat_ny =  math.cos(wp_yaw)
        wp_x   = current_wp.transform.location.x
        wp_y   = current_wp.transform.location.y

        lane_blocked = False
        for actor in actor_frames:
            if not actor.is_vehicle:
                continue   # never trigger lane change for pedestrians/walkers
            adx = actor.location.x - wp_x
            ady = actor.location.y - wp_y
            if abs(adx * lat_nx + ady * lat_ny) > cfg.LANE_WIDTH_HALF:
                continue
            rdx   = actor.location.x - ego_x
            rdy   = actor.location.y - ego_y
            along = rdx * fwd_x + rdy * fwd_y
            if 0 < along < cfg.SAFE_STOP_GAP:
                actor_spd = math.hypot(actor.velocity.x, actor.velocity.y)
                if actor_spd < pace_speed * cfg.LC_SLOW_THRESH:
                    lane_blocked = True
                    break

        if not lane_blocked:
            return False

        # ── Try left lane first, then right ──────────────────────────────
        for get_neighbor, required_permission in (
            (lambda wp: wp.get_left_lane(),  carla.LaneChange.Left),
            (lambda wp: wp.get_right_lane(), carla.LaneChange.Right),
        ):
            nb = get_neighbor(current_wp)
            if nb is None:
                continue
            if nb.lane_type != carla.LaneType.Driving:
                continue   # no sidewalks, bike lanes, parking, etc.
            if not self._same_direction(current_wp, nb):
                continue   # never cross into oncoming traffic
            if nb.is_junction:
                continue
            # Respect the map's lane-change permission markings.
            allowed = current_wp.lane_change
            if not (allowed == carla.LaneChange.Both or allowed == required_permission):
                continue
            if not self._lane_clear(nb, actor_frames, ego_x, ego_y, ego_yaw):
                self._cooldown = cfg.LC_BLOCKED_TICKS
                continue

            # Execute: re-anchor tracker to adjacent lane centre-line.
            perception._tracked_wp  = nb
            perception._approach_wp = nb
            self._cooldown = cfg.LC_COMMIT_TICKS
            return True

        return False


# ---------------------------------------------------------------------------
# 12. Actor
# ---------------------------------------------------------------------------

class Actor:
    def __init__(self, cost_module: CostModule, cfg: type = Configurator):
        self.policy    = Policy(cfg)
        self.optimizer = Optimizer(cost_module, cfg)
        self.smoother  = ControlSmoother(
            cfg.SMOOTH_ACCEL_ALPHA, cfg.SMOOTH_STEER_ALPHA,
            cfg.MAX_ACCEL_RATE,     cfg.MAX_STEER_RATE,
            max_brake_rate=cfg.MAX_BRAKE_RATE,
        )
        self.cfg = cfg

    # ------------------------------------------------------------------

    def _road_is_clear(self, ego_frame: EgoFrame, actor_frames: list) -> bool:
        """
        True when no actor is within CLEAR_ROAD_DIST in the forward cone.
        On a clear road we skip MPPI and apply the deterministic nominal action,
        eliminating all stochastic shiver.
        """
        cfg       = self.cfg
        ego_yaw   = math.radians(ego_frame.transform.rotation.yaw)
        ego_x     = ego_frame.location.x
        ego_y     = ego_frame.location.y
        half_cone = math.radians(cfg.CLEAR_CONE)

        for actor in actor_frames:
            dx   = actor.location.x - ego_x
            dy   = actor.location.y - ego_y
            dist = math.hypot(dx, dy)
            if dist >= cfg.CLEAR_ROAD_DIST:
                continue
            angle_to  = math.atan2(dy, dx)
            rel_angle = (angle_to - ego_yaw + math.pi) % (2 * math.pi) - math.pi
            if abs(rel_angle) < half_cone:
                return False
        return True

    def _nominal_control(
        self,
        ego_frame: EgoFrame,
        waypoint,
        lookahead_wp,
        speed_ceiling: float,
    ) -> carla.VehicleControl:
        """
        Deterministic control used on a clear road (no MPPI noise).
        Throttle is proportional to the gap between current speed and the ceiling,
        capped at CRUISE_THROTTLE_MAX so the car never lunges to full throttle.
        Brake is proportional to overspeed.
        """
        nominal_steer = self.policy._nominal_steer(ego_frame, waypoint, lookahead_wp)
        speed     = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)
        err_accel = max(0.0, speed_ceiling - speed)
        err_brake = max(0.0, speed - speed_ceiling)
        throttle  = float(np.clip(self.cfg.SPEED_KP_ACCEL * err_accel, 0.0, self.cfg.CRUISE_THROTTLE_MAX))
        brake     = float(np.clip(self.cfg.SPEED_KP_BRAKE  * err_brake, 0.0, 1.0))
        if throttle >= brake:
            brake = 0.0
        else:
            throttle = 0.0
        return carla.VehicleControl(throttle=throttle, steer=nominal_steer,
                                    brake=brake, hand_brake=False)

    def _speed_ceiling(self, ego_frame: EgoFrame, actor_frames: list, waypoint,
                       open_road_speed: float = None) -> float:
        """
        Returns a hard speed ceiling set by road clearance ahead.

        On a completely clear road this returns open_road_speed (the pace-controller
        target, or FREE_CRUISE_SPEED when no pace controller is active).
        As the closest same-lane actor closes in, the ceiling drops continuously
        toward zero using a distance-proportional taper.  The controller then
        simply drives up to (but never beyond) this ceiling — no separate target,
        no EMA on the target value.  The rate limiter on the throttle output
        already prevents abrupt accelerations.
        """
        cfg       = self.cfg
        if open_road_speed is None:
            open_road_speed = cfg.FREE_CRUISE_SPEED
        ego_yaw   = math.radians(ego_frame.transform.rotation.yaw)
        ego_x     = ego_frame.location.x
        ego_y     = ego_frame.location.y
        ego_spd   = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)
        half_cone = math.radians(cfg.CLEAR_CONE)

        # Lane lateral normal from waypoint (used for lane-offset computation)
        if waypoint is not None:
            wp_yaw   = math.radians(waypoint.transform.rotation.yaw)
            lane_nx  = -math.sin(wp_yaw)   # lateral normal (left of forward)
            lane_ny  =  math.cos(wp_yaw)
            wp_x     = waypoint.transform.location.x
            wp_y     = waypoint.transform.location.y
        else:
            lane_nx, lane_ny = None, None

        min_eff_dist = cfg.CLEAR_HORIZON
        speed_cap    = open_road_speed   # lowered by stopped or in-gap actors

        for actor in actor_frames:
            dx   = actor.location.x - ego_x
            dy   = actor.location.y - ego_y
            dist = math.hypot(dx, dy)
            if dist >= cfg.CLEAR_HORIZON:
                continue

            angle_to  = math.atan2(dy, dx)
            rel_angle = (angle_to - ego_yaw + math.pi) % (2 * math.pi) - math.pi
            abs_rel   = abs(rel_angle)

            if abs_rel >= 2.0 * half_cone:
                continue
            cone_weight = (1.0 if abs_rel < half_cone else
                           math.cos((abs_rel - half_cone) / half_cone * math.pi / 2) ** 2)

            # Lane weight: lateral offset of actor from ego lane center.
            if lane_nx is not None:
                adx = actor.location.x - wp_x
                ady = actor.location.y - wp_y
                lat_offset = abs(adx * lane_nx + ady * lane_ny)
                # 1.0 inside lane, linear fade to 0.0 at 2× LANE_WIDTH_HALF
                lane_weight = max(0.0, 1.0 - lat_offset / (2.0 * cfg.LANE_WIDTH_HALF))
            else:
                lane_weight = 1.0

            combined = cone_weight * lane_weight
            if combined < 0.05:   # actor is clearly in another lane — skip
                continue

            if dist > 0.1:
                ux, uy = dx / dist, dy / dist
            else:
                ux, uy = 1.0, 0.0

            actor_spd_along = actor.velocity.x * ux + actor.velocity.y * uy
            ego_spd_along   = ego_spd * math.cos(rel_angle)
            closing_speed   = max(0.0, ego_spd_along - actor_spd_along)

            eff_dist = max(0.0, dist - cfg.CLOSING_SPEED_GAIN * closing_speed * combined)
            # Blend eff_dist toward CLEAR_HORIZON for off-lane actors so they barely
            # affect the cruise speed.
            blended_eff = combined * eff_dist + (1.0 - combined) * cfg.CLEAR_HORIZON
            min_eff_dist = min(min_eff_dist, blended_eff)

            # Speed caps apply only to actors solidly in our lane (combined > 0.5)
            if combined > 0.5 and dist < cfg.SAFE_STOP_GAP:
                actor_speed = math.hypot(actor.velocity.x, actor.velocity.y)

                if actor_speed < cfg.STOP_ACTOR_SPEED:
                    # Stopped actor: taper to 0 at STOP_DISTANCE.
                    # Use FREE_CRUISE_SPEED as the taper reference so the pace target
                    # doesn't amplify the braking when we're running slow.
                    gap_t = max(0.0, (dist - cfg.STOP_DISTANCE) /
                                max(cfg.SAFE_STOP_GAP - cfg.STOP_DISTANCE, 0.1))
                    speed_cap = min(speed_cap, cfg.FREE_CRUISE_SPEED * gap_t)
                else:
                    # Moving actor: cap to its speed + small headroom.
                    gap_fraction = max(0.0, (dist - cfg.STOP_DISTANCE) /
                                       max(cfg.SAFE_STOP_GAP - cfg.STOP_DISTANCE, 0.1))
                    headroom  = cfg.FREE_CRUISE_SPEED * gap_fraction * 0.3
                    speed_cap = min(speed_cap, actor_speed + headroom)

        t = min_eff_dist / cfg.CLEAR_HORIZON
        # Taper is computed against FREE_CRUISE_SPEED so that reducing
        # open_road_speed (pace controller) never makes lateral/distant
        # actors cause braking — it merely caps the top end.
        cruise_speed = cfg.MIN_FOLLOW_SPEED + t * (cfg.FREE_CRUISE_SPEED - cfg.MIN_FOLLOW_SPEED)
        return min(cruise_speed, speed_cap, open_road_speed)

    def act(
        self,
        ego_frame: EgoFrame,
        waypoint,
        predictions: dict,
        actor_frames: list,
        lookahead_wp=None,
        cruise_target: float = None,
    ) -> carla.VehicleControl:
        # ── Traffic-light compliance ────────────────────────────────────────
        tl_state = getattr(ego_frame, 'traffic_light_state',
                           carla.TrafficLightState.Unknown)
        tl_red    = (tl_state == carla.TrafficLightState.Red)
        tl_yellow = (tl_state == carla.TrafficLightState.Yellow)
        tl_dist  = getattr(ego_frame, 'traffic_light_dist', float("inf"))

        # Right-turn exemption: in CARLA's left-handed coordinate system a
        # clockwise (rightward) turn produces a positive signed yaw change.
        # When the road ahead turns right by at least TL_RIGHT_TURN_YAW radians,
        # the car may treat the intersection as a yield-and-proceed — TL compliance
        # is disabled and only obstacle avoidance governs behaviour.
        turning_right = False
        if waypoint is not None and lookahead_wp is not None:
            wp_yaw_cur  = math.radians(waypoint.transform.rotation.yaw)
            wp_yaw_look = math.radians(lookahead_wp.transform.rotation.yaw)
            signed_curve = (wp_yaw_look - wp_yaw_cur + math.pi) % (2 * math.pi) - math.pi
            if signed_curve >= self.cfg.TL_RIGHT_TURN_YAW:
                turning_right = True

        tl_must_stop = (tl_red or tl_yellow) and not turning_right

        # Use pace-controller target as the open-road ceiling; fall back to config.
        open_road_speed = cruise_target if cruise_target is not None else self.cfg.FREE_CRUISE_SPEED
        speed_ceil = self._speed_ceiling(ego_frame, actor_frames, waypoint, open_road_speed)

        # ── Curve speed limit ──────────────────────────────────────────────
        # Reduce speed on bends so lateral inertia doesn't push the car off-centre.
        if waypoint is not None and lookahead_wp is not None:
            wp_yaw_cur  = math.radians(waypoint.transform.rotation.yaw)
            wp_yaw_look = math.radians(lookahead_wp.transform.rotation.yaw)
            road_curve  = abs((wp_yaw_look - wp_yaw_cur + math.pi) % (2 * math.pi) - math.pi)
            if road_curve > self.cfg.CURVE_YAW_THRESH:
                speed_ceil = min(speed_ceil, self.cfg.CURVE_SPEED_MAX)

        # ── Traffic-light gradual taper (skipped on right turns) ──────────
        if (tl_red or tl_yellow) and not turning_right:
            buf        = self.cfg.TL_STOP_BUFFER
            target_spd = 0.0 if tl_red else self.cfg.TL_YELLOW_SPEED_CAP
            d_eff      = max(0.0, tl_dist - buf)

            if d_eff <= 0.0:
                tl_ceil = target_spd
            elif d_eff < self.cfg.TL_TAPER_START:
                frac    = d_eff / self.cfg.TL_TAPER_START
                tl_ceil = target_spd + frac * (self.cfg.FREE_CRUISE_SPEED - target_spd)
            else:
                tl_ceil = self.cfg.FREE_CRUISE_SPEED
            speed_ceil = min(speed_ceil, tl_ceil)

        if self._road_is_clear(ego_frame, actor_frames) and not tl_must_stop:
            raw_control = self._nominal_control(ego_frame, waypoint, lookahead_wp, speed_ceil)
        else:
            candidates  = self.policy.sample(ego_frame, waypoint, speed_ceil, lookahead_wp)
            raw_control = self.optimizer.optimize(
                candidates, ego_frame, waypoint, predictions, speed_ceil, tl_must_stop)

        after_emergency = emergency_override(raw_control, ego_frame, actor_frames, self.cfg)
        is_emergency    = (after_emergency.brake != raw_control.brake or
                           after_emergency.throttle != raw_control.throttle)
        control = self.smoother.smooth(after_emergency, emergency=is_emergency)
        ego_frame.turning_right = turning_right
        return control, speed_ceil


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

class CameraFilter:
    """
    Smooths only the spectator yaw in complex-number space (handles ±180° wrap).
    Position is tracked exactly — any EMA lag on position creates a constant
    camera-to-vehicle offset that teleports by tiny amounts every tick, which
    CARLA renders as shimmer even when the car is stopped.
    """
    def __init__(self, alpha_yaw: float = 0.04):
        self.alpha_yaw = alpha_yaw
        self._cos = None
        self._sin = None

    def update(self, ego_tf: carla.Transform) -> tuple:
        """Returns (x, y, z, smoothed_yaw_deg) with exact position, smooth yaw."""
        loc     = ego_tf.location
        yaw_rad = math.radians(ego_tf.rotation.yaw)
        c, s    = math.cos(yaw_rad), math.sin(yaw_rad)

        if self._cos is None:
            self._cos, self._sin = c, s
        else:
            b         = self.alpha_yaw
            rc        = b * c + (1 - b) * self._cos
            rs        = b * s + (1 - b) * self._sin
            n         = math.hypot(rc, rs)
            self._cos = rc / n
            self._sin = rs / n

        return loc.x, loc.y, loc.z, math.degrees(math.atan2(self._sin, self._cos))


def update_spectator(spectator: carla.Actor, cam: CameraFilter,
                     ego_tf: carla.Transform, mode: str):
    x, y, z, yaw = cam.update(ego_tf)

    if mode == "bird":
        cam_tf = carla.Transform(
            carla.Location(x=x, y=y, z=z + 30.0),
            carla.Rotation(pitch=-90.0, yaw=yaw, roll=0.0),
        )
    else:
        rad      = math.radians(yaw)
        offset_x = -8.0 * math.cos(rad)
        offset_y = -8.0 * math.sin(rad)
        cam_tf   = carla.Transform(
            carla.Location(x=x + offset_x, y=y + offset_y, z=z + 4.0),
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
    cam_filter  = CameraFilter()
    stm         = ShortTermMemory()
    perception  = Perception(world, ego, stm)
    world_model = WorldModel(stm)
    cost_module  = CostModule(cfg)
    actor_ctrl   = Actor(cost_module, cfg)
    pace_ctrl    = LapPaceController(ego.get_location(), cfg)
    lane_changer = LaneChanger(world.get_map(), cfg)

    try:
        while True:
            world.tick()

            perception.tick()
            ego_frame = stm.latest_ego()
            if ego_frame is None:
                continue

            update_spectator(spectator, cam_filter, ego_frame.transform, cfg.CAMERA_VIEW)

            predictions  = world_model.predict()
            waypoint     = perception.current_waypoint
            lookahead_wp = perception.lookahead_waypoint
            actor_frames = stm.latest_actors()
            cruise_target = pace_ctrl.tick(ego_frame)

            # Pre-compute values the lane changer needs (same logic as in act()).
            tl_state_lc = getattr(ego_frame, 'traffic_light_state',
                                  carla.TrafficLightState.Unknown)
            tl_must_stop_lc = (tl_state_lc in (carla.TrafficLightState.Red,
                                                carla.TrafficLightState.Yellow))
            rc_lc = 0.0
            if waypoint is not None and lookahead_wp is not None:
                _y0 = math.radians(waypoint.transform.rotation.yaw)
                _y1 = math.radians(lookahead_wp.transform.rotation.yaw)
                rc_lc = abs((_y1 - _y0 + math.pi) % (2 * math.pi) - math.pi)
            lc_fired = lane_changer.tick(
                perception, ego_frame, actor_frames,
                cruise_target, tl_must_stop_lc, rc_lc)

            control, speed_ceil = actor_ctrl.act(
                ego_frame, waypoint, predictions, actor_frames, lookahead_wp,
                cruise_target=cruise_target)
            ego.apply_control(control)

            speed = math.hypot(ego_frame.velocity.x, ego_frame.velocity.y)
            tl_state = getattr(ego_frame, 'traffic_light_state', carla.TrafficLightState.Unknown)
            rt_flag  = getattr(ego_frame, 'turning_right', False)
            print(
                f"[AD] speed={speed:.1f} m/s ({speed*3.6:.1f} km/h)  "
                f"ceil={speed_ceil:.1f} pace={cruise_target:.1f} m/s  "
                f"lap={pace_ctrl.lap_count} dist={pace_ctrl.odometer:.0f}m t={pace_ctrl.elapsed:.0f}s  "
                f"tl={tl_state}{'(RT)' if rt_flag else ''}  "
                f"tl_dist={getattr(ego_frame, 'traffic_light_dist', float('inf')):.1f}m  "
                f"{'LC! ' if lc_fired else ''}"
                f"throttle={control.throttle:.2f}  brake={control.brake:.2f}  "
                f"steer={control.steer:.3f}"
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
