import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from cspace import RectangularCSpace
from robot import UnicycleRobot


@dataclass
class LocalizationProblem:
    # A localization problem instance is a robot, starting state, and a sequence of controls to apply.
    name: str
    robot: UnicycleRobot
    start: np.ndarray
    controls: np.ndarray
    # It also includes some parameters for the observation model: 
    # the lidar beam offsets, the lidar interpolation delta, and the measurement noise standard deviation.
    beam_offsets: np.ndarray
    observation_delta: float
    measurement_noise_std: float
    # It also includes some parameters for the dynamics model: 
    # the control noise and the duration of each control.
    control_noise: np.ndarray
    control_duration: float
    # We can optionally include the true states/observations for evaluation and debugging purposes
    # The "visible" scenarios include these, in part for deterministic replay
    true_states: np.ndarray | None = None
    observations: np.ndarray | None = None

# Load a localization problem from a JSON file
def load_problem(params_file):
    params_file = Path(params_file)
    with open(params_file, "r", encoding="utf-8") as file:
        data = json.load(file)
    # CSPACE
    workspace_boundary = np.array([
        [data["boundary"]["x_min"], data["boundary"]["x_max"]], 
        [data["boundary"]["y_min"], data["boundary"]["y_max"]]])
    cspace_boundary = np.concatenate([workspace_boundary, np.array([[0.0, 2.0 * np.pi]])], axis=0)
    obstacles = [np.array(obstacle) for obstacle in data["obstacles"]]
    interpolation_delta = float(data["dt"])
    cspace = RectangularCSpace(
        workspace_boundary=workspace_boundary,
        cspace_boundary=cspace_boundary,
        is_angular=np.array([False, False, True]),
        obstacles=obstacles,
        rectangle_height=float(data["robot_height"]),
        rectangle_width=float(data["robot_width"]),
        interpolation_delta=interpolation_delta)
    # ROBOT
    robot = UnicycleRobot(cspace, interpolation_delta=interpolation_delta)
    # LOCALIZATION
    problem = LocalizationProblem(
        name=data.get("name", params_file.stem),
        robot=robot,
        start=np.array(data["start"]),
        controls=np.array(data["controls"]),
        beam_offsets=np.array(data["beam_offsets"]),
        control_duration=float(data["control_duration"]),
        observation_delta=float(data["observation_delta"]),
        measurement_noise_std=float(data["measurement_noise_std"]),
        control_noise=np.array(data["control_noise"]),
        true_states=np.array(data["true_states"]) if "true_states" in data else None,
        observations=np.array(data["observations"]) if "observations" in data else None,
    )
    if problem.true_states is not None and len(problem.true_states) != len(problem.controls) + 1:
        raise ValueError(
            f"{params_file} stores {len(problem.true_states)} true states but {len(problem.controls)} controls."
        )
    if problem.observations is not None:
        expected_observation_count = len(problem.true_states) if problem.true_states is not None else len(problem.controls) + 1
        if len(problem.observations) != expected_observation_count:
            raise ValueError(
                f"{params_file} stores {len(problem.observations)} observations but expected {expected_observation_count}."
            )
    return problem

# a wrapper for apply_constant_control that first adds some noise to the control then also validates the trajectory
def apply_noisy_control(robot, cspace, config, control, duration, control_noise, rng=None):
    # specifying the rng is helpful for generating consistent data...
    # in particular for separating the action rng from observation rng 
    if rng is None:
        noise = np.random.uniform(low=-control_noise, high=control_noise, size=control.shape)
    else:
        noise = rng.uniform(low=-control_noise, high=control_noise, size=control.shape)
    noisy_control = control + noise
    # keep the control within the control boundary
    noisy_control = np.clip(
        noisy_control,
        robot.control_boundary[:, 0],
        robot.control_boundary[:, 1],
    )
    # apply the noisy control
    trajectory = robot.apply_constant_control(config, noisy_control, duration)
    # if the control is too small...
    if trajectory.shape[0] == 0:
        return config.copy()
    # validate, if invalid return None
    for pose in trajectory:
        if not cspace.is_valid(pose):
            return None
    return trajectory[-1]


# Our interface for localization mainly depends on two methods - act and observe
# We support both an offline "replay" version in which we just load step/observation data
# and an online version that actually samples noisy actions and controls.
# The main reason for the offline version is to allow for faster and more consistent debugging using precomputed data...
class LocalizationSimulator:
    def __init__(self, problem):
        self.problem = problem
        self.robot = problem.robot
        self.cspace = self.robot.cspace
        self.reset()

    def reset(self):
        if self.problem.true_states is not None:
            self._true_state = self.problem.true_states[0].copy()
        else:
            self._true_state = self.problem.start.copy()
        self._true_history = [self._true_state.copy()]
        self.step_index = 0
        return None

    # entry point for simulating observations from particles
    def observe_config(self, config, noisy=False):
        return np.array([self.cspace.observe_distance(
                            config,
                            delta=self.problem.observation_delta,
                            noise_std=self.problem.measurement_noise_std if noisy else 0.0,
                            beam_offset=beam_offset) for beam_offset in self.problem.beam_offsets])
    # entry point for simulating observations from the true state
    def observe(self):
        if self.problem.observations is not None:
            index = min(self.step_index, len(self.problem.observations) - 1)
            return self.problem.observations[index].copy()
        return self.observe_config(self._true_state, noisy=True)
    # entry point for simulating actions from particles
    def propagate_config(self, config, control, control_noise=None):
        return apply_noisy_control(
            self.robot,
            self.cspace,
            config,
            control,
            self.problem.control_duration,
            self.problem.control_noise if control_noise is None else control_noise,
        )
    # entry point for transitioning the true state in the environment
    def step(self, control):
        if self.problem.true_states is not None:
            next_index = min(self.step_index + 1, len(self.problem.true_states) - 1)
            self._true_state = self.problem.true_states[next_index].copy()
        else:
            next_true_state = self.propagate_config(self._true_state, control)
            if next_true_state is not None:
                self._true_state = next_true_state
        self.step_index += 1
        self._true_history.append(self._true_state.copy())
        return None


def evaluate_localization_history(simulator, history):
    true_history = list(simulator._true_history)
    if len(history) == len(true_history):
        aligned_true_history = true_history
    elif len(history) == len(true_history) - 1:
        aligned_true_history = true_history[1:]
    else:
        raise ValueError(
            f"History length {len(history)} does not match simulator truth history length {len(true_history)}."
        )

    annotated_history = []
    for step, true_state in zip(history, aligned_true_history):
        position_error = float(np.linalg.norm(step["estimate"][:2] - true_state[:2]))
        annotated_step = dict(step)
        annotated_step["true_state"] = true_state
        annotated_step["position_error"] = position_error
        annotated_step["error"] = position_error
        annotated_history.append(annotated_step)
    return annotated_history


def plot_localization_history(problem, history, title=None):
    if not history:
        fig, ax = plt.subplots(figsize=(7.5, 7.5))
        problem.robot.cspace.draw_workspace(ax)
        ax.set_aspect("equal", adjustable="box")
        if title is not None:
            ax.set_title(title)
        return [fig], [np.array([[ax]])]

    estimates = np.array([step["estimate"] for step in history])
    num_steps = len(history)
    has_truth = "true_state" in history[0] and "position_error" in history[0]
    true_states = np.array([step["true_state"] for step in history]) if has_truth else None

    steps_per_figure = 9
    num_cols = 3
    num_rows = 3
    figures = []
    axes_pages = []
    num_pages = int(np.ceil(num_steps / steps_per_figure))

    for page_idx in range(num_pages):
        start_idx = page_idx * steps_per_figure
        end_idx = min(start_idx + steps_per_figure, num_steps)
        page_history = history[start_idx:end_idx]

        fig, axes = plt.subplots(
            num_rows,
            num_cols,
            figsize=(14.5, 14.0),
            squeeze=False,
        )
        fig.subplots_adjust(top=0.88, bottom=0.08, left=0.05, right=0.98, hspace=0.35, wspace=0.22)

        legend_handles = None
        legend_labels = None
        for local_step_idx, step in enumerate(page_history):
            global_step_idx = start_idx + local_step_idx
            ax = axes.flat[local_step_idx]
            problem.robot.cspace.draw_workspace(ax)

            particles = step["particles"]
            ax.scatter(
                particles[:, 0],
                particles[:, 1],
                s=18,
                alpha=0.32,
                color="tab:blue",
                label="particle",
            )

            if has_truth:
                ax.plot(
                    true_states[: global_step_idx + 1, 0],
                    true_states[: global_step_idx + 1, 1],
                    color="tab:orange",
                    linewidth=2.0,
                    label="true trajectory",
                )
                ax.scatter(
                    true_states[: global_step_idx + 1, 0],
                    true_states[: global_step_idx + 1, 1],
                    s=22,
                    color="tab:orange",
                    zorder=3,
                )
                problem.robot.cspace.draw_state(
                    ax,
                    true_states[global_step_idx],
                    color="tab:orange",
                    alpha=0.95,
                )
            problem.robot.cspace.draw_state(
                ax,
                estimates[global_step_idx],
                color="tab:green",
                alpha=0.95,
                label="estimate",
            )

            control = step["control"]
            has_control = control is not None
            control_text = "initial observation" if not has_control else f"control=({control[0]:.2f}, {control[1]:.2f})"
            title_text = f"Step {global_step_idx}\n{control_text}"
            ax.set_title(title_text, fontsize=10)
            ax.set_aspect("equal", adjustable="box")

            if legend_handles is None:
                legend_handles, legend_labels = ax.get_legend_handles_labels()

        for ax in axes.flat[len(page_history):]:
            ax.set_visible(False)

        if title is None:
            base_title = problem.name
        else:
            base_title = title

        page_range_text = f"steps {start_idx}-{end_idx - 1} (page {page_idx + 1}/{num_pages})"
        summary_title = f"{base_title}\n{page_range_text}"
        fig.suptitle(summary_title, fontsize=14)

        if legend_handles and legend_labels:
            deduped = {}
            for handle, label in zip(legend_handles, legend_labels):
                if label and label not in deduped:
                    deduped[label] = handle
            fig.legend(
                deduped.values(),
                deduped.keys(),
                loc="lower center",
                ncol=min(3, len(deduped)),
                bbox_to_anchor=(0.5, 0.02),
            )

        figures.append(fig)
        axes_pages.append(axes)

    return figures, axes_pages
