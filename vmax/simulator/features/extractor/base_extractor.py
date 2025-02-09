# Copyright 2025 Valeo.


"""Base class for feature extractors."""

import jax
import jax.numpy as jnp
import matplotlib as mpl
from waymax import datatypes

from vmax.simulator import features, operations, waymax_overrides
from vmax.simulator.features import extractor


FEATURE_MAP = {
    "waypoints": ("xy",),
    "velocity": ("vel_xy",),
    "speed": ("speed",),
    "yaw": ("yaw",),
    "size": ("length", "width"),
    "valid": ("valid",),
    "direction": ("dir_xy",),
    "types": ("types",),
    "state": ("state",),
    "object_types": ("object_types",),
}


class BaseFeaturesExtractor(extractor.AbstractFeaturesExtractor):
    """Base class for feature extractors."""

    def __init__(
        self,
        obs_past_num_steps: int | None = None,
        objects_config: dict | None = None,
        roadgraphs_config: dict | None = None,
        traffic_lights_config: dict | None = None,
        path_target_config: dict | None = None,
    ) -> None:
        """Initialize the base features extractor.

        Args:
            obs_past_num_steps: Number of past steps to consider.
            objects_config: Configuration for object features.
            roadgraphs_config: Configuration for roadgraph features.
            traffic_lights_config: Configuration for traffic light features.
            path_target_config: Configuration for path target features.

        """
        self._obs_past_num_steps = obs_past_num_steps or 1

        # Set default configs if None
        self._objects_config = objects_config or {"features": []}
        self._roadgraphs_config = roadgraphs_config or {"features": []}
        self._traffic_light_config = traffic_lights_config or {"features": []}
        self._path_target_config = path_target_config or {"features": []}

        # Extract parameters with defaults
        # Objects
        self._num_closest_objects = self._objects_config.get("num_closest_objects", 8)

        # Roadgraph points
        self._meters_box = self._roadgraphs_config.get("meters_box")
        self._roadgraph_top_k = self._roadgraphs_config.get("roadgraph_top_k", 1000)
        self._roadgraph_interval = self._roadgraphs_config.get("interval", 1)
        self._max_meters = self._roadgraphs_config.get("max_meters", 50)

        if self._meters_box is None:
            self._roadgraph_top_k_prefilter = max(self._roadgraph_top_k, 2000)
        else:
            self._roadgraph_top_k_prefilter = (self._meters_box["front"] + self._meters_box["back"]) * (
                self._meters_box["left"] + self._meters_box["right"]
            )

        # Traffic lights
        self._num_closest_traffic_lights = self._traffic_light_config.get("num_closest_traffic_lights", 16)

        # Path target
        self._num_target_path_points = self._path_target_config.get("num_points", 10)
        self._points_gap = self._path_target_config.get("points_gap", 5)

        self._dict_mapping = {
            "types": extractor.utils.RG_MAPPING,  # roadgraph
            "state": extractor.utils.TL_MAPPING,  # trafficlight
            "object_types": extractor.utils.OBJECT_MAPPING,  # objects
        }

        # Build feature keys
        self._object_features_key = [
            feature for key in self._objects_config["features"] for feature in FEATURE_MAP[key]
        ]
        self._roadgraph_features_key = [
            feature for key in self._roadgraphs_config["features"] for feature in FEATURE_MAP[key]
        ]
        self._traffic_lights_features_key = [
            feature for key in self._traffic_light_config["features"] for feature in FEATURE_MAP[key]
        ]
        self._path_target_features_key = [
            feature for key in self._path_target_config["features"] for feature in FEATURE_MAP[key]
        ]

    @property
    def obs_past_num_steps(self) -> int:
        """Return the number of past steps considered for observation."""
        return self._obs_past_num_steps

    def _get_sdc_observation(self, state):
        """Retrieve the SDC observation from the simulator state.

        Args:
            state: The simulator state.

        Returns:
            The observation corresponding to the SDC.

        """
        sdc_observation = waymax_overrides.sdc_observation_from_state(
            state,
            self._obs_past_num_steps,
            self._roadgraph_top_k_prefilter,
            self._meters_box,
        )

        return jax.tree.map(lambda x: x[0], sdc_observation)

    def _get_features_size(self, feature_keys: str) -> int:
        """Calculate the total feature size for given feature keys.

        Args:
            feature_keys: The key or keys representing the features.

        Returns:
            The sum of feature sizes.

        """
        return sum([extractor.get_feature_size(key, self._dict_mapping) for key in feature_keys])

    def extract_features(
        self,
        state: datatypes.SimulatorState,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        """Extract features from the simulator state.

        Args:
            state: The simulator state.

        Returns:
            A tuple containing:
              - SDC object features.
              - Other objects features.
              - Roadgraph features.
              - Traffic lights features.
              - Path target features.

        """
        sdc_observation = self._get_sdc_observation(state)

        objects_features = self._build_objects_features(sdc_observation)
        roadgraphs_features = self._build_roadgraph_features(sdc_observation)
        traffic_lights_features = self._build_traffic_lights_features(sdc_observation)
        path_target_features = self._build_target_features(sdc_observation)

        # (num_agents + 1, obs_past_num_steps, num_trajectories_features)
        stack_object_features = objects_features.stack_fields()
        # (obs_past_num_steps, num_trajectories_features), assuming the SDC is always the first object
        sdc_object_features = stack_object_features[0, :, :]
        # (num_closest_agents, obs_past_num_steps, num_trajectories_features)
        other_objects_features = stack_object_features[1:, :, :]

        return (
            sdc_object_features,
            other_objects_features,
            roadgraphs_features.stack_fields(),
            traffic_lights_features.stack_fields(),
            path_target_features.data,
        )

    def unflatten_features(self, vectorized_obs: jax.Array) -> tuple[tuple[jax.Array, ...], tuple[jax.Array, ...]]:
        """Unflatten a vectorized observation into features and masks.

        Args:
            vectorized_obs: The vectorized observation.

        Returns:
            A tuple that contains:
              - A tuple of features.
              - A tuple of masks.

        """
        batch_dims = vectorized_obs.shape[-3:-1]
        flatten_size = vectorized_obs.shape[-1]
        unflatten_size = 0

        object_features_size = self._get_features_size(self._object_features_key)
        roadgraph_features_size = self._get_features_size(self._roadgraph_features_key)
        traffic_lights_features_size = self._get_features_size(self._traffic_lights_features_key)
        path_target_feature_size = self._get_features_size(self._path_target_features_key)

        sdc_object_size = 1 * self._obs_past_num_steps * object_features_size
        sdc_object_features = vectorized_obs[..., unflatten_size : unflatten_size + sdc_object_size]
        sdc_object_features = sdc_object_features.reshape(
            *batch_dims,
            1,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += sdc_object_size

        other_objects_size = self._num_closest_objects * self._obs_past_num_steps * object_features_size
        other_objects_features = vectorized_obs[..., unflatten_size : unflatten_size + other_objects_size]
        other_objects_features = other_objects_features.reshape(
            *batch_dims,
            self._num_closest_objects,
            self._obs_past_num_steps,
            object_features_size,
        )
        unflatten_size += other_objects_size

        roadgraph_size = self._roadgraph_top_k * roadgraph_features_size
        roadgraphs_features = vectorized_obs[..., unflatten_size : unflatten_size + roadgraph_size]
        roadgraphs_features = roadgraphs_features.reshape(
            *batch_dims,
            self._roadgraph_top_k,
            roadgraph_features_size,
        )
        unflatten_size += roadgraph_size

        traffic_lights_size = self._num_closest_traffic_lights * self._obs_past_num_steps * traffic_lights_features_size
        traffic_lights_features = vectorized_obs[..., unflatten_size : unflatten_size + traffic_lights_size]
        traffic_lights_features = traffic_lights_features.reshape(
            *batch_dims,
            self._num_closest_traffic_lights,
            self._obs_past_num_steps,
            traffic_lights_features_size,
        )
        unflatten_size += traffic_lights_size

        path_target_size = self._num_target_path_points * path_target_feature_size
        path_target_features = vectorized_obs[..., unflatten_size : unflatten_size + path_target_size]
        path_target_features = path_target_features.reshape(
            *batch_dims,
            self._num_target_path_points,
            path_target_feature_size,
        )
        unflatten_size += path_target_size

        assert flatten_size == unflatten_size, f"Unflatten size {unflatten_size} does not match {flatten_size}"

        features = (
            sdc_object_features[..., :-1],
            other_objects_features[..., :-1],
            roadgraphs_features[..., :-1],
            traffic_lights_features[..., :-1],
            path_target_features,
        )
        masks = (
            sdc_object_features[..., -1].astype(bool),
            other_objects_features[..., -1].astype(bool),
            roadgraphs_features[..., -1].astype(bool),
            traffic_lights_features[..., -1].astype(bool),
        )

        return features, masks

    def plot_features(self, state: datatypes.SimulatorState, ax: mpl.axes.Axes) -> None:
        """Plot all extracted features on the provided axes.

        Args:
            state: The simulator state.
            ax: The matplotlib axes to plot on.

        """
        sdc_observation = self._get_sdc_observation(state)

        objects_features = self._build_objects_features(sdc_observation)
        roadgraphs_features = self._build_roadgraph_features(sdc_observation)
        traffic_lights_features = self._build_traffic_lights_features(sdc_observation)
        path_target_features = self._build_target_features(sdc_observation)

        # 1. Plot objects trajectories and bbox
        objects_features.plot(ax)

        # 2. Plot roadgraph points
        roadgraphs_features.plot(ax)

        # 3. Plot traffic lights
        traffic_lights_features.plot(ax)

        # 4. Plot path target
        path_target_features.plot(ax)

    def _build_objects_features(self, sdc_obs: datatypes.Observation) -> features.ObjectFeatures:
        """Create features for dynamic objects.

        The objects are selected based on the closest distance of the position of the SDC.



        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of ObjectFeatures.

        """
        object_features = features.ObjectFeatures(field_names=self._object_features_key)

        if len(self._object_features_key) == 0:
            return object_features

        # (num_agents,)
        distances_ego_objects = jnp.linalg.norm(sdc_obs.trajectory.xy[:, -1, :], axis=-1)
        # (num_agents,)
        distances_ego_valid_objects = jnp.where(sdc_obs.trajectory.valid[:, -1], distances_ego_objects, jnp.inf)
        # (num_closest_agents + 1,)
        closest_object_idxs = operations.get_index(
            -distances_ego_valid_objects,
            k=self._num_closest_objects + 1,
            squeeze=False,
        )

        object_features = features.ObjectFeatures(field_names=self._object_features_key)
        for key in self._object_features_key:
            feature = getattr(sdc_obs.metadata, key) if key == "object_types" else getattr(sdc_obs.trajectory, key)
            feature = feature[closest_object_idxs]
            feature = extractor.normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 2:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(object_features, key, feature)

        return object_features

    def _build_roadgraph_features(self, sdc_obs: datatypes.Observation) -> features.RoadgraphFeatures:
        """Create roadgraph features from the SDC observation.

        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of RoadgraphFeatures.

        """
        roadgraph_features = features.RoadgraphFeatures(field_names=self._roadgraph_features_key)

        if len(self._roadgraph_features_key) == 0:
            return roadgraph_features

        roadgraph_points = self._reduce_and_filter_roadgraph_points(sdc_obs.roadgraph_static_points)

        for key in self._roadgraph_features_key:
            feature = getattr(roadgraph_points, key)
            feature = extractor.normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 1:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(roadgraph_features, key, feature)

        return roadgraph_features

    def _build_traffic_lights_features(self, sdc_obs: datatypes.Observation) -> features.TrafficLightFeatures:
        """Create traffic light features from the SDC observation.

        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of TrafficLightFeatures.

        """
        traffic_light_features = features.TrafficLightFeatures(field_names=self._traffic_lights_features_key)

        if len(self._traffic_lights_features_key) == 0:
            return traffic_light_features

        # (num_agents,)
        distances_traffic_lights = jnp.linalg.norm(sdc_obs.traffic_lights.xy[:, -1], axis=-1)
        # (num_agents,)
        distances_traffic_lights_valid = jnp.where(
            sdc_obs.traffic_lights.valid[:, -1],
            distances_traffic_lights,
            jnp.inf,
        )
        # (num_closest_agents + 1,)
        closest_tl_idxs = operations.get_index(
            -distances_traffic_lights_valid,
            k=self._num_closest_traffic_lights,
            squeeze=False,
        )

        for key in self._traffic_lights_features_key:
            feature = getattr(sdc_obs.traffic_lights, key)[closest_tl_idxs]
            feature = extractor.normalize_by_feature(feature, key, self._max_meters, self._dict_mapping)

            if feature.ndim == 2:
                feature = jnp.expand_dims(feature, axis=-1)

            setattr(traffic_light_features, key, feature)

        return traffic_light_features

    def _build_target_features(self, sdc_obs: datatypes.Observation) -> features.PathTargetFeatures:
        """Build path target features from the SDC observation.

        Args:
            sdc_obs: The SDC observation.

        Returns:
            An instance of PathTargetFeatures.

        """
        self._path_target_features_key = ["xy"]
        if len(self._path_target_features_key) == 0:
            return jnp.array(())

        # (1, num_paths, num_points_per_path)
        sdc_paths = sdc_obs.sdc_paths
        # (1, num_paths, 1)
        on_route = sdc_paths.on_route
        # (1, num_paths, num_points_per_path)
        on_route = jnp.repeat(on_route, sdc_paths.num_points_per_path, axis=-1)
        # (1, num_paths, num_points_per_path)
        mask = jnp.logical_and(on_route, sdc_paths.valid)
        longest_path_idx = jnp.argmax(jnp.sum(mask, axis=-1))

        sdc_paths_xy = datatypes.MaskedArray.create_and_validate(
            value=sdc_paths.xy,
            valid=jnp.stack([mask, mask], axis=-1),
        )
        sdc_paths_xy = sdc_paths_xy.masked_value()

        sdc_path = sdc_paths_xy[longest_path_idx]

        indices = jnp.arange(self._points_gap, sdc_path.shape[0], self._points_gap)
        indices = indices[: self._num_target_path_points]
        path_target = jnp.take(sdc_path, indices, axis=0)

        path_target = extractor.normalize_path(path_target, self._max_meters)

        return features.PathTargetFeatures(xy=path_target)

    def _reduce_and_filter_roadgraph_points(self, roadgraph: datatypes.RoadgraphPoints) -> datatypes.RoadgraphPoints:
        """Reduce and filter roadgraph points.

        Args:
            roadgraph: The roadgraph points.

        Returns:
            A reduced and filtered roadgraph points.

        """
        roadgraph.valid = roadgraph.valid & self._filter(roadgraph)

        xy = roadgraph.xy
        dist = jnp.linalg.norm(xy, axis=-1)
        dist = jnp.where(roadgraph.valid, dist, jnp.inf)

        idx_to_keep = jnp.arange(0, len(dist), self._roadgraph_interval)
        mask = jnp.zeros_like(dist, dtype=bool)
        mask = mask.at[idx_to_keep].set(True)
        dist = jnp.where(mask, dist, jnp.inf)

        _, idx = jax.lax.top_k(-dist, self._roadgraph_top_k)
        roadgraph = jax.tree.map(lambda x: x[idx], roadgraph)

        return roadgraph

    def _filter(self, roadgraph: datatypes.RoadgraphPoints) -> jax.Array:
        """Filter roadgraph points.

        Placeholder function to filter roadgraph points.

        Args:
            roadgraph: The roadgraph points.

        Returns:
            A boolean array indicating the valid points.

        """
        return jnp.ones_like(roadgraph.valid, dtype=bool)
