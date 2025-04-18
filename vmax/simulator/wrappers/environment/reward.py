# Copyright 2025 Valeo.

"""Reward functions for the simulator."""

import jax
import jax.numpy as jnp
from waymax import datatypes
from waymax import metrics as waymax_metrics
from waymax.env.planning_agent_environment import PlanningAgentEnvironment

from vmax.simulator import metrics, operations
from vmax.simulator.wrappers import environment


class RewardLinearWrapper(environment.Wrapper):
    """Wraps the environment to compute rewards using a linear combination of functions."""

    def __init__(self, env: PlanningAgentEnvironment, reward_config: dict) -> None:
        """Initialize the reward linear wrapper.

        Args:
            env: The environment to wrap.
            reward_config: Configuration for reward computation.

        """
        super().__init__(env)
        self._reward_config = reward_config

        if isinstance(reward_config, list | tuple):
            self.reward = self._reward_from_list
        elif isinstance(reward_config, dict):
            self.reward = self._reward_from_dict
        else:
            raise ValueError("Reward config must be a list or a dictionary.")

    def _reward_from_list(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Combine rewards provided as a list.

        Args:
            state: Current simulator state.
            action: Action taken.

        """
        reward = 0.0

        for reward_config in self._reward_config:
            reward_fn = _get_reward_fn(reward_config)
            reward += reward_fn(state)

        return reward

    def _reward_from_dict(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Combine rewards provided as a dictionary.

        Args:
            state: Current simulator state.
            action: Action taken.

        """
        reward = 0.0

        for reward_name, reward_config in self._reward_config.items():
            reward_fn = _get_reward_fn(reward_name)
            reward += reward_fn(state, **reward_config)

        return reward


class RewardCustomWrapper(environment.Wrapper):
    """Wraps the environment to compute rewards using a custom function."""

    def reward(self, state: datatypes.SimulatorState, action: datatypes.Action) -> jax.Array:
        """Compute a custom reward.

        This is a placeholder function, use it as you wish o7

        Args:
            state: Current simulator state.
            action: Action taken.

        Returns:
            Reward value.

        """


## TERMINATION REWARD FUNCTIONS


def _compute_overlap_reward(
    state: datatypes.SimulatorState,
    bonus: float = 0.0,
    penalty: float = -1.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward penalizing overlaps.

    Args:
        state: Current simulator state.
        bonus: Reward if no overlap.
        penalty: Penalty if an overlap occurs.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    overlap = waymax_metrics.OverlapMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_overlap = jax.tree_util.tree_map(lambda x: x[sdc_idx], overlap)

    reward_overlap = jnp.where(sdc_overlap == 1.0, penalty, bonus)

    return reward_overlap * weight


def _compute_offroad_reward(
    state: datatypes.SimulatorState,
    bonus: float = 0.0,
    penalty: float = -1.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward penalizing offroad behavior.

    Args:
        state: Current simulator state.
        bonus: Reward when on road.
        penalty: Penalty when offroad.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    offroad = waymax_metrics.OffroadMetric().compute(state).value

    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    sdc_offroad = jax.tree_util.tree_map(lambda x: x[sdc_idx], offroad)

    reward_offroad = jnp.where(sdc_offroad == 1.0, penalty, bonus)

    return reward_offroad * weight


def _compute_red_light_reward(state: datatypes.SimulatorState, penalty: float = -1.0, weight: float = 1.0) -> float:
    """Compute a reward penalizing red light violations.

    Args:
        state: Current simulator state.
        penalty: Penalty value.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    has_runned_red_light = metrics.RunRedLightMetric().compute(state).value

    red_light_reward = has_runned_red_light * penalty

    return red_light_reward * weight


## DENSE REWARD FUNCTIONS


def _compute_log_divergence_reward(state: datatypes.SimulatorState, weight: float = 1.0) -> jax.Array:
    """Compute reward based on log divergence.

    Args:
        state: Current simulator state.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    return log_divergence * weight


def _compute_log_divergence_clip_reward(
    state: datatypes.SimulatorState,
    threshold: float = 0.3,
    bonus: float = 0.0,
    penalty: float = 0.0,
    weight: float = 1.0,
) -> jax.Array:
    """Compute reward using a clipped log divergence.

    Args:
        state: Current simulator state.
        threshold: Distance threshold.
        bonus: Reward when within the threshold.
        penalty: Penalty if beyond the threshold.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    log_divergence = waymax_metrics.LogDivergenceMetric().compute(state).value
    log_divergence = jnp.sum(log_divergence, axis=-1)

    log_divergence_reward = jnp.where(log_divergence > threshold, penalty, bonus)

    return log_divergence_reward * weight


def _compute_ttc_reward(
    state: datatypes.SimulatorState,
    threshold: float = 1.5,
    bonus: float = 0.0,
    penalty: float = -1.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward based on time-to-collision.

    Args:
        state: Current simulator state.
        threshold: Time threshold.
        bonus: Reward if ttc is above threshold.
        penalty: Penalty if ttc is below threshold.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    ttc = metrics.TimeToCollisionMetric().compute(state).value

    ttc_reward = jnp.where(ttc < threshold, penalty, bonus)

    return ttc_reward * weight


def _compute_comfort_reward(state: datatypes.SimulatorState, weight: float = 1.0, penalty: float = -1.0) -> float:
    """Compute a comfort-related reward.

    Args:
        state: Current simulator state.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    comfort_metric_reward_2 = metrics.ComfortMetric().compute_reward(state).value

    return comfort_metric_reward_2 * weight


def _compute_speed_limit_reward(
    state: datatypes.SimulatorState,
    penalty: float = -1.0,
    bonus: float = 0.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward based on speed-limit adherence.

    Args:
        state: Current simulator state.
        penalty: Penalty for exceeding the speed limit.
        bonus: Reward for meeting the speed limit.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    speed_limit = metrics.infer_speed_limit_from_simulator_state(state)
    sdc_idx = operations.get_index(state.object_metadata.is_sdc)
    ego_speed = state.current_sim_trajectory.speed[sdc_idx].squeeze()

    reward = bonus * (jnp.abs(ego_speed) / speed_limit)
    speed_limit_reward = jnp.where(ego_speed > speed_limit + 2.23, penalty, reward)

    return speed_limit_reward * weight


def _compute_driving_direction_reward(
    state: datatypes.SimulatorState,
    penalty: float = -1.0,
    bonus: float = 0.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward for driving direction compliance.

    Args:
        state: Current simulator state.
        penalty: Penalty for deviating.
        bonus: Reward for maintaining course.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    driving_direction_metric = metrics.DrivingDirectionComplianceMetric().compute(state).value
    driving_direction_reward = jnp.where(driving_direction_metric > 0, penalty, bonus)

    return driving_direction_reward * weight


def _compute_deviate_lane_reward(
    state: datatypes.SimulatorState,
    penalty: float = -1.0,
    bonus: float = 0.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward penalizing lane deviations.

    Args:
        state: Current simulator state.
        penalty: Penalty for deviating from lane.
        bonus: Reward for staying in lane.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    on_multiple_lanes_metric = metrics.OnMultipleLanesMetric().compute(state).value
    on_multiple_lanes_reward = jnp.where(
        on_multiple_lanes_metric > 0,
        penalty,
        bonus,
    )

    return on_multiple_lanes_reward * weight


def _compute_making_progress_reward(
    state: datatypes.SimulatorState,
    penalty: float = 0.0,
    bonus: float = 1.0,
    weight: float = 1.0,
) -> float:
    """Compute a reward promoting progress.

    Args:
        state: Current simulator state.
        penalty: Penalty if progress is lacking.
        bonus: Reward for making progress.
        weight: Multiplier for the reward.

    Returns:
        Computed reward.

    """
    current_progression = waymax_metrics.ProgressionMetric().compute(state).value
    n_state = state.replace(timestep=state.timestep - 1)
    previous_progression = waymax_metrics.ProgressionMetric().compute(n_state).value

    making_progress_reward = jnp.where(current_progression > previous_progression, bonus, penalty)

    return making_progress_reward * weight


def _compute_off_route_reward(
    state: datatypes.SimulatorState,
    penalty: float = -1.0,
    weight: float = 1.0,
):
    # Waymax offroute's score: 0 if you're on-route, distance to route if you're offroute
    off_route_score = waymax_metrics.OffRouteMetric().compute(state).value

    return (off_route_score > 0) * weight * penalty


def _get_reward_fn(reward_name: str) -> callable:
    """Retrieve a reward function by its name.

    Args:
        reward_name: Name identifier of the reward function.

    Returns:
        Callable reward function.

    """
    reward_dict = {
        "log_div_clip": _compute_log_divergence_clip_reward,
        "log_div": _compute_log_divergence_reward,
        "overlap": _compute_overlap_reward,
        "offroad": _compute_offroad_reward,
        "off_route": _compute_off_route_reward,
        "ttc": _compute_ttc_reward,
        "red_light": _compute_red_light_reward,
        "comfort": _compute_comfort_reward,
        "speed": _compute_speed_limit_reward,
        "driving_direction": _compute_driving_direction_reward,
        "lane_deviation": _compute_deviate_lane_reward,
        "progression": _compute_making_progress_reward,
    }

    if reward_name not in reward_dict:
        raise ValueError(f"Reward function {reward_name} not implemented.")

    return reward_dict[reward_name]
