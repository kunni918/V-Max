# Copyright 2025 Valeo.


"""Utility functions for training scripts."""

import logging
import os
import pickle
import textwrap
from argparse import ArgumentParser
from datetime import datetime
from typing import Any

import jax
import psutil
from etils import epath
from tensorboardX import SummaryWriter

from vmax.simulator import datasets


logger = logging.getLogger(__name__)


def resolve_output_dir(
    algorithm_name: str,
    observation_type: str,
    reward_type: str,
    encoder_config: dict,
    name_run: str,
    name_exp: str,
) -> str:
    """Determine the output directory based on training parameters.

    Args:
        algorithm_name: The name of the algorithm.
        observation_type: The observation type.
        reward_type: The reward type.
        encoder_config: The encoder configuration.
        name_run: The run name.
        name_exp: The experiment name.

    Returns:
        The output directory path.

    """
    _name_exp = "runs" if name_exp is None else name_exp

    if name_run is None:
        _name_run = f"{algorithm_name.upper()}_{observation_type.upper()}_{reward_type.upper()}"

        encoder_type = encoder_config["type"]
        if encoder_type != "none":
            _name_run += f"_{encoder_type.upper()}"

        _name_run += f"_{datetime.now().strftime('%d-%m_%H:%M:%S')}/"
    else:
        _name_run = name_run

    output_dir = f"{_name_exp}/{_name_run}"

    logger.info(f"Output directory: {output_dir}")

    return output_dir


def apply_xla_flags(config: dict) -> None:
    """Apply XLA flags for performance, debugging, and caching.

    Args:
        config: Training configuration.

    """
    xla_flags = ""

    if config["debug_flag"]:
        xla_flags += "--xla_gpu_autotune_level=0 "  # SEEDING
    if config["perf_flag"]:
        xla_flags += "--xla_gpu_enable_pipelined_reduce_scatter=true "
        xla_flags += "--xla_gpu_enable_triton_softmax_fusion=true "
        xla_flags += "--xla_gpu_triton_gemm_any=true "

    os.environ["XLA_FLAGS"] = xla_flags

    if config["cache_flag"]:
        jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)


def log_metrics(
    num_steps: int | None = None,
    metrics: dict | None = None,
    current_step: int | None = None,
    total_timesteps: int | None = None,
    writer: SummaryWriter = None,
) -> None:
    """Log and print training metrics and optionally send them to TensorBoard.

    Args:
        num_steps: Number of steps.
        metrics: Dictionary of metric names and values.
        current_step: Current step count.
        total_timesteps: Total timesteps.
        writer: TensorBoard summary writer.

    """
    if current_step is not None or total_timesteps is not None:
        logger.info(f"-> Step {current_step}/{total_timesteps} - {(current_step / total_timesteps) * 100:.2f}%")
        logger.info(f"-> Data time     : {metrics['runtime/data_time']:.2f}s")
        logger.info(f"-> Training time : {metrics['runtime/training_time']:.2f}s")
        logger.info(f"-> Log time      : {metrics['runtime/log_time']:.2f}s")
        logger.info(f"-> Eval time     : {metrics['runtime/eval_time']:.2f}s")

    for key, value in metrics.items():
        if writer:
            prefix = "metrics/" if "/" not in key else ""
            if "steps" in key or "rewards" in key:
                prefix = "training/"
            writer.add_scalar(f"{prefix}{key}", value, num_steps)
        logger.info(f"{key}: {value}")


def build_config_dicts(config: dict) -> tuple[dict, dict]:
    """Build separate configuration dictionaries for the environment and runtime.

    Args:
        config: The complete training configuration.

    Returns:
        A tuple containing the environment configuration and run configuration.

    """
    path_dataset = datasets.get_dataset(config["path_dataset"])
    path_dataset_eval = datasets.get_dataset(config["path_dataset_eval"])

    sdc_paths_from_data = not config["waymo_dataset"]

    env_config = {
        "path_dataset": path_dataset,
        "path_dataset_eval": path_dataset_eval,
        "sdc_paths_from_data": sdc_paths_from_data,
        "termination_keys": config["termination_keys"],
        "max_num_objects": config["max_num_objects"],
        "reward_type": config["reward_type"],
        "reward_config": config["reward_config"],
        "observation_type": config["observation_type"],
        "observation_config": config["observation_config"],
        "num_envs": config["num_envs"],
        "num_episode_per_epoch": config["num_episode_per_epoch"],
        "num_scenario_per_eval": config["num_scenario_per_eval"],
        "seed": config["seed"],
    }

    if config["network"]["encoder"]["type"] != "none":
        config["network"]["unflatten_config"] = config["observation_config"]

    network_config = config["network"]
    del config["algorithm"]["network"]

    if network_config["value"]["layer_sizes"] is None:
        del network_config["value"]

    run_config = {
        "total_timesteps": config["total_timesteps"],
        "scenario_length": config["scenario_length"],
        "log_freq": config["log_freq"],
        "save_freq": config["save_freq"],
        "num_envs": config["num_envs"],
        "num_episode_per_epoch": config["num_episode_per_epoch"],
        "num_scenario_per_eval": config["num_scenario_per_eval"],
        "seed": config["seed"],
        "eval_freq": config["eval_freq"],
        **config["algorithm"],
        "network_config": network_config,
    }
    del run_config["name"]

    return env_config, run_config


def get_memory_usage() -> float:
    """Retrieve the current process memory usage in GiB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / 1024**3


def print_hyperparameters(params: dict, name: str = "", count: int = -1) -> None:
    """Print hyperparameters in a structured and readable format.

    Args:
        params: Dictionary of hyperparameters.
        name: Optional name of the hyperparameters.
        count: Indentation level.

    """

    def _indent_str(count: int) -> str:
        return " " * 4 * count

    title = f"{name}".center(50, "=") if count <= 0 else f"{_indent_str(count - 1)}- {name}"
    print(title)

    for key, value in params.items():
        if isinstance(value, dict):
            print_hyperparameters(value, name=key, count=count + 1)
        else:
            value_str = str(value)
            wrapped_value = textwrap.fill(value_str, width=70, subsequent_indent=_indent_str(count + 1))
            print(f"{_indent_str(count)}{key}: {wrapped_value}")


def get_and_print_device_info() -> int:
    """Display and return the count of local JAX devices."""
    print("device".center(50, "="))
    print(f"jax.local_devices_to_use: {jax.local_device_count()}")
    print(f"jax.default_backend(): {jax.default_backend()}")
    print(f"jax.local_devices(): {jax.local_devices()}")

    return jax.local_device_count()


def save_params(path: str, params: Any) -> None:
    """Serialize and save model parameters to a specified file."""
    with epath.Path(path).open("wb") as fout:
        fout.write(pickle.dumps(params))


def setup_tensorboard(run_path: str) -> SummaryWriter:
    """Initialize and return a TensorBoard summary writer."""
    return SummaryWriter(log_dir=run_path)


def str2bool(v) -> bool:
    """Convert a string literal to a boolean."""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise ArgumentParser.ArgumentTypeError("Boolean value expected.")
