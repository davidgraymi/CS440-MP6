from dataclasses import dataclass
import numpy as np
from cspace import CSpace2D

# A Robot is defined by how it transitions between configurations in a CSpace
# namely it is a set of controls and ways to apply these controls to get new configurations
class Robot:
    def __init__(self, cspace: CSpace2D, control_boundary, interpolation_delta=0.1):
        self.cspace = cspace
        self.control_boundary = control_boundary
        # a resolution parameter for how finely we simulate the trajectory when applying a control, smaller is more accurate but more expensive
        self.interpolation_delta = interpolation_delta

    def is_valid_control(self, control):
        return np.all(control >= self.control_boundary[:, 0]) and np.all(control <= self.control_boundary[:, 1])

    def sample_random_controls(self, num_controls):
        return np.random.uniform(low=self.control_boundary[:, 0], 
                                 high=self.control_boundary[:, 1], 
                                 size=(num_controls, self.control_boundary.shape[0]))

    # return a single next configuration given current config and control applied for dt time
    # by default if dt=None apply the control for self.interpolation_delta time
    def dynamics(self, config, control, dt=None):
        raise NotImplementedError("Must implement dynamics method in subclass")

    # Apply control starting from config for t time and return a list of intermediate configurations and final configuration
    def apply_constant_control(self, config, control, t):
        cur_time = 0
        poses = [np.copy(config)]
        while cur_time < t - self.interpolation_delta:
            poses.append(self.dynamics(poses[-1], control))
            cur_time += self.interpolation_delta
        if cur_time < t:  # final step to exactly reach t
            poses.append(self.dynamics(poses[-1], control, dt=t - cur_time))
        return np.array(poses[1:])  # exclude the initial config from the returned trajectory


class UnicycleRobot(Robot):
    def __init__(self, cspace : CSpace2D, interpolation_delta=0.1):
        # control consists of (velocity, change in angle)
        control_boundary = np.array([[0.0, 2.0], [-1.0, 1.0]]) # velocity between 0 and 2.0, turn rate between -1 and 1
        # make sure the given CSpace has 3 dimensions (x, y, angle) since that's what our dynamics assume
        # NOTE: alternatively we can define a specific CSpace class that guarantees these things...
        assert cspace.cspace_boundary.shape[0] == 3, "UnicycleRobot requires a 3D CSpace"
        assert np.array_equal(cspace.is_angular, np.array([False, False, True])), "UnicycleRobot requires a CSpace with 2 translational and 1 angular dimension"
        super().__init__(cspace, control_boundary, interpolation_delta)

    # for a unicycle robot, the dynamics are defined as follows for control = (velocity, turn_rate) and config = (x, y, theta):
    # config_dot = (x_dot, y_dot, theta_dot) where
    # x_dot = velocity*cos(theta)
    # y_dot = velocity*sin(theta)
    # theta_dot = turn_rate
    # The new configuration after applying this control for dt time should be computed via:
    # new_config = config + config_dot * dt
    def dynamics(self, config, control, dt=None):
        if dt is None:
            dt = self.interpolation_delta
        config = np.copy(config)  # avoid modifying in place
        v, theta_dot = control
        config[0] += v*np.cos(config[2]) * dt
        config[1] += v*np.sin(config[2]) * dt
        config[2] += theta_dot * dt
        config[2] %= 2*np.pi  # keep angle within [0, 2pi)
        return config


# define a struct for car params
@dataclass
class DubinsCarParams:
    robot: UnicycleRobot # a Dubins car is a unicycle robot with fixed velocity and turn rate
    velocity: float = 1.0 # a FIXED velocity for all controls
    turn_rate: float = 1.0 # a FIXED turn rate for left and right turns
    control_duration: float = 1.0 # a FIXED duration for each control
    goal_tolerance: float = 1.0 # our goal condition will be to get within this distance of the goal configuration