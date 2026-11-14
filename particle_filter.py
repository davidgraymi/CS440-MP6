import matplotlib.pyplot as plt
import numpy as np

from localization import LocalizationSimulator


# The UPDATE step of a Bayesian Filter computes the posterior for each particle
def update_belief(
    simulator, observed_distances, particles, priors, particle_observation_noise=0.1
):
    # (IV): compute the posterior particle weights based on noisy observations
    # - Use simulator.observe_config(particle, noisy=False) to compute the expected observation for each particle configuration.
    # - For the gaussian observation likelihood model use particle_observation_noise as the standard deviation
    #   - You should tune this parameter and set the DEFAULT value above to whatever gives you good performance
    #   - ... which might not be the same as the data-generation noise, i.e., simulator.problem.measurement_noise_std
    # - Return a NumPy array of normalized weights that sum to 1; zero prior weights must remain zero
    # - See the instructions for the gaussian likelihood and stable normalization formulas

    # print(f"observed_distances: {observed_distances.shape}")
    # print(f"particles: {particles.shape}")
    # print(f"priors: {priors.shape}")

    h_x = np.array(
        [simulator.observe_config(particle, noisy=False) for particle in particles]
    )
    # print(f"h_x: {h_x.shape}")

    l = np.log(priors) + np.sum(
        (
            -np.log(np.sqrt(2 * np.pi) * particle_observation_noise)
            - (((observed_distances - h_x) ** 2) / (2 * particle_observation_noise**2))
        ),
        axis=1,
    )
    l[priors == 0] = -np.inf
    # print(f"l: {l.shape}")

    m = np.max(l)
    u = np.exp(l - m)
    u_total = np.sum(u)
    w = u / u_total

    return w


def resample(particles, weights):
    # (IV): sample with replacement from the particles according to the distribution defined by the weights
    # and return the new particles and uniform weights as NumPy arrays
    size = len(particles)
    choices = np.random.choice(np.arange(size), size=size, p=weights)
    return particles[choices], np.ones(size) / size


# The PREDICT step of a Bayesian Filter computes the predicted belief after applying the control,
# except in a particle filter we sample from the transition model p(s' | s, a)
def act(simulator, control, particles, particle_control_noise=0.1):
    # (IV): propagate each particle once with the simulator and return a NumPy array
    # - Use simulator.propagate_config(particle, control, control_noise=particle_control_noise) to propagate each particle
    #   - You should tune the particle_control_noise parameter and set the DEFAULT value above to whatever gives you good performance
    #   - ... which might not be the same as the data-generation noise, i.e., simulator.problem.control_noise
    # - If after propagating a particle through the control it ends up in an invalid state, keep the old particle
    h_x = [
        val
        if (
            val := simulator.propagate_config(
                particle, control, control_noise=particle_control_noise
            )
        )
        is not None
        else particle
        for particle in particles
    ]
    return particles


# we provide a simple function to initialize num_particles uniformly over valid states in the configuration space
def initialize_valid_particles(cspace, num_particles):
    particles = []
    while len(particles) < num_particles:
        candidate = cspace.sample_n_configs_in_boundary(1)[0]
        if cspace.is_valid(candidate):
            particles.append(candidate)
    return np.array(particles)


# There are many options for estimating the state from the particle distribution
# One reasonable option is to take the weighted mean, which is what we do here
# though another is to take the particle with the highest weight as the estimate (the MAP estimate)
# NOTE: this function exists purely for debugging purposes
def estimate_state(particles, weights):
    # Use a circular mean for heading because angles wrap at 2pi.
    estimate = np.average(particles, axis=0, weights=weights)
    sine = np.average(np.sin(particles[:, 2]), weights=weights)
    cosine = np.average(np.cos(particles[:, 2]), weights=weights)
    estimate[2] = np.arctan2(sine, cosine) % (2 * np.pi)
    return estimate


# The bayesian filter loop is:
# - Initialize belief - uniform particles --> observe --> update belief
# - For each control:
#   1. Predict - propagate each particle through the control, also step the simulator
#   2. Update - get observation then update the belief using the observation model and Bayes' rule
#   3. Resample - a particle filter specific step avoid particle deprivation
#
# num_particles is the number of particles to maintain in the belief at each step.
# More particles will give a better approximation to the true belief but will take more computation time.
def run_particle_filter(simulator, num_particles):
    particles = initialize_valid_particles(simulator.cspace, num_particles)
    observed_distances = simulator.observe()
    weights = update_belief(
        simulator,
        observed_distances,
        particles,
        priors=np.full(num_particles, 1.0 / num_particles),
    )
    particles, weights = resample(particles, weights)
    estimate = estimate_state(particles, weights)
    history = [
        {
            "step": 0,
            "control": None,
            "observed_distances": observed_distances,
            "particles": particles.copy(),
            "weights": weights.copy(),
            "estimate": estimate,
        }
    ]

    for step_idx, control in enumerate(simulator.problem.controls):
        particles = act(simulator, control, particles)
        simulator.step(control)
        observed_distances = simulator.observe()
        weights = update_belief(simulator, observed_distances, particles, weights)
        particles, weights = resample(particles, weights)
        estimate = estimate_state(particles, weights)
        history.append(
            {
                "step": step_idx + 1,
                "control": control,
                "observed_distances": observed_distances,
                "particles": particles.copy(),
                "weights": weights.copy(),
                "estimate": estimate,
            }
        )

    return history


if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="CS440 MP6 Particle Filter Localization"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="data/localization_visible_1.json",
        help="Path to a localization JSON scenario",
    )
    parser.add_argument(
        "--num_particles", type=int, default=500, help="Number of particles to maintain"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Random seed for repeatable experiments"
    )
    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save a final particle plot",
    )
    parser.add_argument(
        "--show", action="store_true", help="Display the final particle cloud"
    )
    args = parser.parse_args()
    from localization import (
        evaluate_localization_history,
        load_problem,
        plot_localization_history,
    )

    np.random.seed(args.seed)
    problem = load_problem(args.env)
    simulator = LocalizationSimulator(problem)
    history = run_particle_filter(simulator, num_particles=args.num_particles)

    evaluated_history = evaluate_localization_history(simulator, history)
    final_step = evaluated_history[-1]

    print(f"Scenario: {problem.name}")
    print(f"Final true state: {np.round(final_step['true_state'], 3)}")
    print(f"Final estimate:   {np.round(final_step['estimate'], 3)}")
    print(f"Final position error: {final_step['position_error']:.3f}")

    figures = []
    if args.show or args.save is not None:
        figures, _ = plot_localization_history(
            problem,
            evaluated_history,
            title=problem.name,
        )
    if args.save is not None and figures:
        output_path = Path(args.save)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        for figure_idx, figure in enumerate(figures):
            page_path = output_path.with_name(
                f"{output_path.stem}_{figure_idx + 1}{output_path.suffix}"
            )
            figure.savefig(page_path, bbox_inches="tight")
    if args.show and figures:
        plt.show()
