import numpy as np
from shapely.geometry import Polygon, Point
from matplotlib.axes import Axes

RAY_EPSILON = 1e-9

# check if a point (shape (n,)) is inside a boundary (shape (n, 2) - min and max for each dimension)
# touching the boundary counts as being inside, so we use >= and <=
def point_in_boundary(x, boundary : np.ndarray) -> bool:
    return np.all(x >= boundary[:,0]) and np.all(x <= boundary[:,1])

# Abstract base class for a Configuration Space with a 2D workspace
class CSpace2D:
    def __init__(self, workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta=0.1) -> None:
        # shape (2, 2), [[x_min, x_max], [y_min, y_max]]
        self.workspace_boundary = workspace_boundary
        # shape (n, 2), [[dim1_min, dim1_max], ..., [dimn_min, dimn_max]]
        self.cspace_boundary = cspace_boundary
        # boolean array of shape (n,), True if dimension i is angular, meaning it wraps around
        self.is_angular = is_angular
        # list of obstacles, each obstacle i is an array of shape (m_i, 2)
        # representing its vertices in workspace in counter-clockwise order
        self.obstacles = obstacles
        # a resolution parameter for how finely we check collision along motion between two configurations
        self.interpolation_delta = interpolation_delta
        self.max_observation_range = np.linalg.norm(self.workspace_boundary[:,1] - self.workspace_boundary[:,0])
        self.lidar_segments = []
        self.build_lidar_segments()

    # draw the workspace with boundary and obstacles on given Axes
    def draw_workspace(self, ax : Axes):
        ax.set_xlim(self.workspace_boundary[0])
        ax.set_ylim(self.workspace_boundary[1])
        # draw obstacles boundaries in black
        for obstacle in self.obstacles:
            # close the polygon by repeating the first point at the end
            ax.plot(np.append(obstacle[:,0], obstacle[0,0]), np.append(obstacle[:,1], obstacle[0,1]), color="black")

    # return a unit vector pointing from start to end, taking into account angular dimensions
    # works for both single points (shape (n,)) and batches of points (shape (m, n))
    def point_to_point_direction(self, start : np.ndarray, end : np.ndarray) -> np.ndarray:
        if len(start.shape) == 1:
            direction = end - start
            direction[self.is_angular] = (direction[self.is_angular] + np.pi) % (2 * np.pi) - np.pi
            return direction
        else:
            direction = end - start
            # can replace this ":" with "..." to allow broadcasting for any number of leading dimensions,
            # but we will just assume the first dimension is the batch dimension for simplicity
            direction[:, self.is_angular] = (direction[:, self.is_angular] + np.pi) % (2 * np.pi) - np.pi
            return direction

    def point_to_point_distance(self, start : np.ndarray, end : np.ndarray) -> np.ndarray:
        direction = self.point_to_point_direction(start, end)
        return np.linalg.norm(direction, axis=-1)

    # sampling-based motion planning depends on sampling configurations in the cspace boundary
    # this method returns n such uniformly random samples shaped (n, cspace_dim)
    def sample_n_configs_in_boundary(self, n : int) -> np.ndarray:
        return np.random.uniform(self.cspace_boundary[:,0],
                                 self.cspace_boundary[:,1],
                                 size=(n, self.cspace_boundary.shape[0]))

    # Return a sequence of configurations that define an *unvalidated* straight-line motion from start_config to end_config
    #   - Do not include the start and end configs in the returned sequence, only the intermediate configs
    #   - Account for angular dimension wrap-around
    # In general you might expect local_planner to depend on dynamic constraints (defined in the Robot class)
    #   but in practice in the presence of dynamic constraints we only ever do "extension" not "connection"
    #   and so the only real local_planner we need is a straight line local planner in CSpace
    def straight_line_local_planner(self, start_config, end_config):
        direction = self.point_to_point_direction(start_config, end_config)
        distance = np.linalg.norm(direction)
        if distance <= self.interpolation_delta:
            return []  # no intermediate points needed, can directly connect start and end (assuming both are valid themselves)
        direction /= distance  # normalize to get unit direction vector
        num_steps = int(distance / self.interpolation_delta) # number of interpolation steps (round down to not overshoot)
        intermediates = np.full((num_steps, start_config.shape[0]), start_config)  # shape (num_steps, num_dims)
        intermediates += np.outer(np.arange(1, num_steps + 1), direction * self.interpolation_delta)  # shape (num_steps, num_dims)
        intermediates[:, self.is_angular] = (intermediates[:, self.is_angular]) % (2 * np.pi)
        return intermediates

    # does not check endpoints
    def is_valid_edge(self, start_config, end_config):
        intermediates = self.straight_line_local_planner(start_config, end_config)
        for config in intermediates:
            if not self.is_valid(config):
                return False
        return True

    # draw the robot at given configuration on given Axes
    def draw_state(self, ax : Axes, config):
        pass  # to be implemented in subclasses

    # get workspace points corresponding to the given configuration
    def forward_kinematics(self, config):
        pass  # to be implemented in subclasses

    # check if the robot at given configuration is valid (inside boundary and not colliding with obstacles)
    def is_valid(self, config) -> bool:
        pass # to be implemented in subclasses

    def cross_2d(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(a[0] * b[1] - a[1] * b[0])

    def build_lidar_segments(self):
        x_min, x_max = self.workspace_boundary[0]
        y_min, y_max = self.workspace_boundary[1]
        corners = np.array(
            [
                [x_min, y_min],
                [x_max, y_min],
                [x_max, y_max],
                [x_min, y_max],
            ]
        )

        self.lidar_segments = []
        for index in range(4):
            self.lidar_segments.append((corners[index], corners[(index + 1) % 4]))

        for obstacle in self.obstacles:
            for index in range(obstacle.shape[0]):
                self.lidar_segments.append((obstacle[index], obstacle[(index + 1) % obstacle.shape[0]]))
        return self.lidar_segments

    def ray_segment_distance(self, origin, direction, seg_start, seg_end):
        segment_direction = seg_end - seg_start
        denominator = self.cross_2d(direction, segment_direction)
        if abs(denominator) <= RAY_EPSILON:
            return None

        diff = seg_start - origin
        ray_distance = self.cross_2d(diff, segment_direction) / denominator
        segment_fraction = self.cross_2d(diff, direction) / denominator

        if ray_distance < 0.0:
            return None
        if segment_fraction < -RAY_EPSILON or segment_fraction > 1.0 + RAY_EPSILON:
            return None
        return ray_distance

    def observe_distance(self, config, delta=0.05, noise_std=0.0, beam_offset=0.0, rng=None):
        _ = delta
        origin = config[:2]
        beam_angle = float(config[2] + beam_offset)
        direction = np.array([np.cos(beam_angle), np.sin(beam_angle)])

        distance = float(self.max_observation_range)
        for seg_start, seg_end in self.lidar_segments:
            hit_distance = self.ray_segment_distance(origin, direction, seg_start, seg_end)
            if hit_distance is not None and hit_distance < distance:
                distance = float(hit_distance)

        if noise_std > 0.0:
            if rng is None:
                distance += np.random.normal(loc=0.0, scale=noise_std)
            else:
                distance += rng.normal(loc=0.0, scale=noise_std)
        return max(distance, 0.0)

# Polygonal CSpace where the robot and obstacles are interpreted as (shapely) polygons
# for the purpose of collision checking (not necessarily efficient but simple and generic)
class PolygonalCSpace(CSpace2D):
    def __init__(self, workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta=0.1) -> None:
        super().__init__(workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta=interpolation_delta)
        # a list of shapely Polygons representing obstacles
        self.shapely_obstacles = [Polygon(o) for o in obstacles]

    # default implementation of forward kinematics for a point robot (i.e., the configuration itself is the workspace position)
    def forward_kinematics(self, config):
        return [config[:2]] # for a point robot the workspace position is just the (x,y) configuration

    def is_valid(self, config):
        # first check cspace boundary (we can skip collision checking if we are outside the cspace boundary)
        if not point_in_boundary(config, self.cspace_boundary):
            return False
        # collision checking always begins with forward kinematics
        workspace_corners = self.forward_kinematics(config)
        # check boundary
        for c in workspace_corners:
            if not point_in_boundary(c, self.workspace_boundary):
                return False
        if len(workspace_corners) == 1:
            # if we are a point robot, we can just check if we are inside any obstacle
            point = workspace_corners[0]
            for o in self.shapely_obstacles:
                if o.contains(Point(point)):
                    return False
            return True
        # convert to polygon
        cur_polygon = Polygon(workspace_corners)
        # check collision with obstacles
        for o in self.shapely_obstacles:
            if cur_polygon.intersects(o):
                return False
        # if we are inside the boundary and not colliding with any obstacle, we are valid
        return True

# A PolygonalCSpace where the robot is a rectangle that can translate and rotate in the plane (x,y,theta configuration space)
# NOTE: the dynamics of such a robot (e.g., whether it can rotate in place or only move forward and turn)
#       is not captured in the CSpace but rather in the State class (e.g., DubinsCarState)
class RectangularCSpace(PolygonalCSpace):
    def __init__(self, workspace_boundary, cspace_boundary,
                 is_angular=None, obstacles=[],
                 rectangle_height=1.0, rectangle_width=2.0, interpolation_delta=0.1) -> None:
        assert cspace_boundary.shape[0] == 3, "Expected 3D configuration space"  # we expect (x,y,theta) configuration space

        if is_angular is None:
            print("Assuming third dimension is angular since is_angular not provided, overriding cspace_boundary to be [0, 2pi] for third dimension")
            is_angular = np.array([False, False, True])  # only the angle dimension is angular
            cspace_boundary[2] = np.array([0, 2 * np.pi])  # override to ensure angle wraps around correctly
        super().__init__(workspace_boundary, cspace_boundary, is_angular, obstacles, interpolation_delta=interpolation_delta)

        self.rectangle_height = rectangle_height
        self.rectangle_width = rectangle_width
        self.obstacle_bounds = [o.bounds for o in self.shapely_obstacles]

    # check rectangle bounds before doing polygon collision checks
    def is_valid(self, config):
        b = self.cspace_boundary
        if not (b[0, 0] <= config[0] <= b[0, 1]
                and b[1, 0] <= config[1] <= b[1, 1]
                and b[2, 0] <= config[2] <= b[2, 1]):
            return False

        corners = self.forward_kinematics(config)
        low = corners.min(axis=0)
        high = corners.max(axis=0)
        b = self.workspace_boundary
        if not (b[0, 0] <= low[0] and high[0] <= b[0, 1]
                and b[1, 0] <= low[1] and high[1] <= b[1, 1]):
            return False

        cur_polygon = None
        for obstacle, (xmin, ymin, xmax, ymax) in zip(self.shapely_obstacles, self.obstacle_bounds):
            # disjoint bounding boxes cannot collide
            if high[0] < xmin or low[0] > xmax or high[1] < ymin or low[1] > ymax:
                continue
            if cur_polygon is None:
                cur_polygon = Polygon(corners)
            if cur_polygon.intersects(obstacle):
                return False
        return True

    # draw the car at given configuration on given Axes
    def draw_state(self, ax : Axes, config, **kwargs):
        # do forward kinematics to get the four corners of the rectangle in workspace
        corners = self.forward_kinematics(config)
        # close the rectangle by repeating the first corner at the end
        ax.plot(corners[[0,1,2,3,0],0], corners[[0,1,2,3,0],1], **kwargs)
        # draw an arrow to indicate orientation
        arrow_length = min(self.rectangle_width, self.rectangle_height) / 2
        ax.arrow(config[0], config[1], arrow_length * np.cos(config[2]), arrow_length * np.sin(config[2]),
                 head_width=arrow_length/2, head_length=arrow_length/2, fc='k', ec='k')

    # get the four corners of the car given its (x,y,theta) configuration
    # (x,y) is the center of the car, theta is the orientation angle
    def forward_kinematics(self, config):
        # for readability...
        cos_theta = np.cos(config[2])
        sin_theta = np.sin(config[2])
        W, H = self.rectangle_width, self.rectangle_height
        # the four corners of the rectangle before rotation and translation
        corners = np.array([[-W/2,-H/2],
                            [-W/2, H/2],
                            [ W/2, H/2],
                            [ W/2,-H/2]])
        # we will now rotate counter clockwise by theta then translate
        rotation_matrix = np.array([[cos_theta, -sin_theta],
                                    [sin_theta, cos_theta]])
        rotated_corners = np.dot(rotation_matrix, corners.T).T # shape (4,2)
        translated_corners = rotated_corners + config[:2] # shape (4,2)
        return translated_corners
