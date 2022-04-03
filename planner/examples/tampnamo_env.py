"""Tampnamo environment.
"""

import glob
import os
import itertools
import time
import numpy as np
import pybullet as p
from planner.utils.pybullet_utils import get_kinematic_chain, inverse_kinematics
import planner.utils.pybullet_controllers as controllers
import planner.utils.structs as structs
import planner.utils.constants as constants
from planner.utils.env_base import Environment, EnvironmentFailure
from planner.utils.assets import p_constants
from planner.utils.utils import WORLD, get_asset_path
from planner.solvers.rrt import MRBiRRT, BiRRT

ENV_WIDTH = 6
ENV_HEIGHT = 3
DOOR_SCALE = 0.25  # fraction between 0 and 1
OBJ_SIZE = 0.3
OBJ_HEIGHT = 0.7
ROBOT_Z = 0.3
COLLISION_TOLERANCE = 0.3 # 0.2
CLEAR_BP_DIST = 0.75  # how far to be from object when clearing it


class TampnamoEnvironment(Environment):
    """Tampnamo environment.
    """
    # Object types
    _box_type = structs.Type("box")
    _target_type = structs.Type("target")
    _xdim_type = structs.ContinuousType("xdim")
    _ydim_type = structs.ContinuousType("ydim")
    _xbase_type = structs.ContinuousType("xbase")
    _ybase_type = structs.ContinuousType("ybase")
    _xdim_type.set_domain([-10, 10])
    _ydim_type.set_domain([-10, 10])
    _xbase_type.set_domain([-10, 10])
    _ybase_type.set_domain([-10, 10])

    # Constant world object
    _world = WORLD

    # Predicates
    GoalClear = structs.Predicate("GoalClear", 0, [])
    GoalReach = structs.Predicate("GoalReach", 0, [])
    IsPose = structs.Predicate("IsPose", 3,
                           [_xdim_type, _ydim_type, _box_type])
    _all_predicates = {GoalClear, GoalReach, IsPose}
    _all_predicate_names_to_preds = {p.name: p for p in _all_predicates}
    _continuous_predicates = {pred for pred in _all_predicates
                              if any(t.is_continuous for t in pred.var_types)}

    # Actions
    ReachGoal = structs.Predicate("ReachGoal", 0, [])
    ClearObject = structs.Predicate("ClearObject", 3,
                                    [_box_type, _xdim_type, _ydim_type])
    action_predicates = {ReachGoal, ClearObject}

    def __init__(self, num_objs, seed):
        super().__init__(num_objs, seed)
        self._num_boxes = self._num_objs
        self._seed = self._seed
        self._boxes = []
        for i in range(self._num_boxes):
            self._boxes.append(self._box_type("box{}".format(i)))
        self._targets = [self._target_type("target{}".format(0))]
        self._initial_pybullet_setup()
        self._transmodel_cache = {}

    def parse_state(self, state):
        goal_reach = self._is_goal_reach(state)
        # Build literal set.
        lits = set()
        predicate_names = self._all_predicate_names_to_preds.keys()
        for pred_name in predicate_names:
            pred = self._all_predicate_names_to_preds[pred_name]
            if pred_name == "GoalClear" and not goal_reach:
                lits.add(pred())
            if pred_name == "GoalReach" and goal_reach:
                lits.add(pred())
        return lits

    def get_train_initial_states(self):
        return [self._sample_initial_state() for _ in range(1)] # Set the number of problems

    def simulate(self, state, action):
        next_state = {k: v.copy() for k, v in state.items()}
        do_interpolate = (p.getConnectionInfo()["connectionMethod"] == p.GUI and
                          constants.DO_RENDER)
        self._update_collision_motion_planner(state)

        if action.predicate == self.ReachGoal:
            succ, path = self._get_path(state[self._targets[0]]['pose'][0],
                                        state[self._targets[0]]['pose'][1])
            if not succ:
                raise EnvironmentFailure("No path exists")
            # Check path for object collisions before actually moving.
            for pt in path:
                for p_box in self._p_boxes:
                    if self._is_in_collision(pt, p_box):
                        raise EnvironmentFailure("objcollision")
            self._follow(path, p.getEulerFromQuaternion(state[self._world]['base_pose'][3:])[2],
                         do_interpolate)
        if action.predicate == self.ClearObject: # Currently it only works for a single box
            quat = p.getQuaternionFromEuler([0, 0, -np.pi / 2])
            clear_pose = [state[action.variables[0]]['pose'][0] - CLEAR_BP_DIST,
                        state[action.variables[0]]['pose'][1],
                        0., quat] # Grasping from left

            # Move to clear_base_pose.
            succ, path = self._get_path(clear_pose[0], clear_pose[1])
            if not succ:
                raise EnvironmentFailure("wallcollision")
            # Check path for object collisions before actually moving.
            for pt in path:
                if self._is_in_collision(pt, self._p_boxes[self._boxes.index(action.variables[0])]):
                    raise EnvironmentFailure("objcollision")
            self._follow(path, p.getEulerFromQuaternion(state[self._world]['base_pose'][3:])[2],
                         do_interpolate)

            # Move object out of the way.
            base_pos, base_orn = p.getBasePositionAndOrientation(self._p_robot)
            target_loc, target_orn = p.multiplyTransforms(
                base_pos, base_orn, [0, CLEAR_BP_DIST * 3, 0.65], [0, 1, 0, 0])
            self._move_end_effector(target_loc, target_orn, do_interpolate)
            self._grab_object(self._p_boxes[self._boxes.index(action.variables[0])])
            succ, path = self._get_path(base_pos[0], base_pos[1])
            assert succ, "Spinning in place should always succeed"
            self._follow(path, p.getEulerFromQuaternion(base_orn)[2] + 2 * np.pi,
                         do_interpolate)  # spin in place
            self._grab_object(None)
            self._reset_joint_state(num_steps=(2500 if do_interpolate else 1))

        # Update fields in next_state.
        for box_id in range(len(self._boxes)):
            pos, orn = p.getBasePositionAndOrientation(self._p_boxes[box_id])
            next_state[self._boxes[box_id]]["pose"] = list(pos)+list(orn)
        pos, orn = p.getBasePositionAndOrientation(self._p_targets[0])
        next_state[self._targets[0]]["pose"] = list(pos)+list(orn)
        pos, orn = p.getBasePositionAndOrientation(self._p_robot)
        next_state[self._world]["base_pose"] = list(pos)+list(orn)

        return next_state, None, None

    def _initial_pybullet_setup(self):
        """One-time pybullet setup stuff.
        """
        # Load things into environment.
        if p.getConnectionInfo()["isConnected"]:
            self._setup_motion_planner()
            return
        if constants.USE_VIEWER:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        p.setGravity(0, 0, -10)
        p.setAdditionalSearchPath("planner/utils/assets/tampnamo/")
        p.loadURDF("plane.urdf")
        camera_distance = 6
        yaw = 45
        pitch = -80
        camera_target = [0, 0, 0]
        p.resetDebugVisualizerCamera(camera_distance, yaw, pitch, camera_target,
                                     physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        self._p_robot, = p.loadSDF("kuka_with_gripper_recolored.sdf")
        p.resetBasePositionAndOrientation(self._p_robot,
                                          [-1.5, 0, ROBOT_Z],
                                          [0, 0, 0, 1])
        # Make URDF files.
        wall_height = 3
        with open("planner/utils/assets/tampnamo/side_wall_horiz.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                ENV_WIDTH, 0.01, wall_height, 0.5, 0.5, 0.5, 1))
        with open("planner/utils/assets/tampnamo/side_wall_vert.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                0.01, 3, wall_height, 0.5, 0.5, 0.5, 1))
        with open("planner/utils/assets/tampnamo/inner_wall_horiz.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                ENV_HEIGHT * (1 - DOOR_SCALE) / 2, 0.01, wall_height, 0.5, 0.5, 0.5, 1))
        with open("planner/utils/assets/tampnamo/inner_wall_vert.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                0.01, 3 * (1 - DOOR_SCALE) / 2, wall_height, 0.5, 0.5, 0.5, 1))
        with open("planner/utils/assets/tampnamo/blue_object.urdf", "w") as fil:
            fil.write(p_constants.CUBE_URDF.format(
                OBJ_SIZE, OBJ_SIZE, OBJ_HEIGHT, 0, 0, 1, 1))
        with open("planner/utils/assets/tampnamo/red_object.urdf", "w") as fil:
            fil.write(p_constants.TARGET_URDF.format(
                OBJ_SIZE, OBJ_SIZE, OBJ_HEIGHT, 1, 0, 0, 0.5))
        # Set up walls.
        self._p_walls = []
        self._p_walls.append(p.loadURDF("side_wall_vert.urdf", [-ENV_WIDTH/2, 0, 0]))
        self._p_walls.append(p.loadURDF("side_wall_vert.urdf", [ENV_WIDTH/2, 0, 0]))
        self._p_walls.append(p.loadURDF("side_wall_horiz.urdf", [0, -ENV_HEIGHT/2, 0]))
        self._p_walls.append(p.loadURDF("side_wall_horiz.urdf", [0, ENV_HEIGHT/2, 0]))

        self._p_walls.append(p.loadURDF("inner_wall_vert.urdf", [0, -0.9375, 0]))
        self._p_walls.append(p.loadURDF("inner_wall_vert.urdf", [0, 0.9375, 0]))

        # Set up objects.
        self._p_boxes, self._p_targets = [], []
        for _ in range(self._num_boxes):
            self._p_boxes.append(p.loadURDF("blue_object.urdf"))
        self._p_targets.append(p.loadURDF("red_object.urdf"))  # representing a goal
        self._p_held_obj_tf = None
        for fil in glob.glob("assets/tampnamo/*wall*.urdf"):
            os.remove(fil)
        for fil in glob.glob("assets/tampnamo/*object*.urdf"):
            os.remove(fil)
        p_constants.set_ps([self._p_robot, self._p_walls, self._p_boxes])

    def _is_goal_reach(self, state):
        if state[self._targets[0]]['pose'][0] - state[self._world]['base_pose'][0] < 0.1 and \
            state[self._targets[0]]['pose'][1] - state[self._world]['base_pose'][1] < 0.1:
            return True
        else:
            return False

    def _get_path(self, goalx, goaly):
        """Get a smoothed path to the given (goalx, goaly), obtained using the
        motion planner. Returns a tuple of (success, <path or collision set>).
        """
        cur_xy = p.getBasePositionAndOrientation(self._p_robot)[0][:2]
        precision = 4
        key = (round(cur_xy[0], precision), round(cur_xy[1], precision),
               round(goalx, precision), round(goaly, precision))
        # path = self._rrt.query(cur_xy, (goalx, goaly), step_size=2, timeout=30)
        path = self._rrt.query(cur_xy, (goalx, goaly), timeout=5)
        if path is None:
            return False, None
        else:
            path = self._rrt.smooth_path(path)
            return True, path  # successfully found a path

    def _follow(self, path, target_orn, do_interpolate):
        """Follow the given path, yielded by self._get_path.
        Uses self._move_base().
        """
        (_, _, cur_z), cur_orn = p.getBasePositionAndOrientation(self._p_robot)
        cur_orn = p.getEulerFromQuaternion(cur_orn)[2]
        for i, loc in enumerate(path):
            interp_orn = cur_orn*(1-i/len(path))+target_orn*i/len(path)
            self._move_base(np.r_[loc, cur_z], interp_orn, do_interpolate)

    def _move_base(self, target_loc, target_orn, do_interpolate):
        orig_loc, orig_orn = p.getBasePositionAndOrientation(self._p_robot)
        orig_orn = p.getEulerFromQuaternion(orig_orn)[2]
        orig = np.r_[orig_loc, orig_orn]
        target = np.r_[target_loc, target_orn]
        dist = np.linalg.norm(target-orig)
        if do_interpolate:
            speed = 0.05
            num_steps = int(np.ceil(abs(dist)/speed))
        else:
            num_steps = 1
        for i in range(1, num_steps+1):
            interpolated = orig*(1-i/num_steps)+target*i/num_steps
            interp_orien = p.getQuaternionFromEuler([0, 0, interpolated[3]])
            p.resetBasePositionAndOrientation(
                self._p_robot, interpolated[:3], interp_orien)
            base_link = np.r_[p.getLinkState(self._p_robot, 9)[:2]]
            if self._p_held_obj_tf is not None:
                obj_id, transf = self._p_held_obj_tf
                obj_loc, obj_orn = p.multiplyTransforms(
                    base_link[:3], base_link[3:], transf[0], transf[1])
                p.resetBasePositionAndOrientation(
                    obj_id, obj_loc, obj_orn)
            if do_interpolate:
                time.sleep(0.01)

    def _move_end_effector(self, target_loc, target_orn, do_interpolate,
                           open_amt=None, avoid_cols=False, succ_thresh=0.05):
        num_iter = 0
        while True:
            num_iter += 1
            if num_iter > 10000:
                return False
            njo = min(p.getNumJoints(self._p_robot), 7)
            old_joints = np.array([p.getJointState(self._p_robot, joint_idx)[0]
                                   for joint_idx in range(njo)])
            new_joints = np.array(p.calculateInverseKinematics(
                self._p_robot, endEffectorLinkIndex=6,
                targetPosition=target_loc,
                targetOrientation=target_orn, solver=0))[:njo]
            old_open_amt = p.getJointState(self._p_robot, 11)[0]
            dist = np.linalg.norm(new_joints-old_joints)
            if open_amt is not None:
                dist = max(dist, abs(old_open_amt-open_amt))
            dist_thresh = 1e-2
            if avoid_cols:
                assert open_amt is None
                dist_thresh = 1e-1
            if dist < dist_thresh:
                link_pos, link_orn = p.getLinkState(self._p_robot, 6)[-2:]
                error = np.linalg.norm(np.r_[link_pos, link_orn]-
                                       np.r_[target_loc, target_orn])
                if error < succ_thresh:
                    return True  # success: reached target pose
                return False  # failure: didn't reach target pose
            if do_interpolate:
                speed = 0.05
                num_steps = int(np.ceil(abs(dist)/speed))
            else:
                num_steps = 1
            for i in range(1, num_steps+1):
                interpolated = old_joints*(1-i/num_steps)+new_joints*i/num_steps
                if avoid_cols:
                    p.setJointMotorControlArray(
                        self._p_robot,
                        range(njo),
                        controlMode=p.POSITION_CONTROL,
                        targetPositions=new_joints,
                        targetVelocities=[0 for _ in range(njo)],
                        forces=[500 for _ in range(njo)],
                        positionGains=[0.03 for _ in range(njo)],
                        velocityGains=[1 for _ in range(njo)])
                    p.stepSimulation()
                else:
                    for joint_idx in range(njo):
                        p.resetJointState(self._p_robot, joint_idx,
                                          interpolated[joint_idx])
                if open_amt is not None:
                    # Open gripper to open_amt.
                    open_interp = (old_open_amt*(1-i/num_steps)+
                                   open_amt*i/num_steps)
                    p.resetJointState(self._p_robot, 8, -open_interp)
                    p.resetJointState(self._p_robot, 11, open_interp)
                else:
                    p.resetJointState(self._p_robot, 8, -old_open_amt)
                    p.resetJointState(self._p_robot, 11, old_open_amt)
                base_link = np.r_[p.getLinkState(self._p_robot, 9)[:2]]
                if self._p_held_obj_tf is not None:
                    obj_id, transf = self._p_held_obj_tf
                    obj_loc, obj_orn = p.multiplyTransforms(
                        base_link[:3], base_link[3:], transf[0], transf[1])
                    p.resetBasePositionAndOrientation(
                        obj_id, obj_loc, obj_orn)
                if do_interpolate:
                    time.sleep(0.01)

    def _grab_object(self, obj_id):
        if obj_id is None:
            self._p_held_obj_tf = None
        else:
            base_link_to_world = np.r_[p.invertTransform(
                *p.getLinkState(self._p_robot, 9)[:2])]
            world_to_obj = np.r_[p.getBasePositionAndOrientation(obj_id)]
            self._p_held_obj_tf = (obj_id, p.multiplyTransforms(
                base_link_to_world[:3], base_link_to_world[3:],
                world_to_obj[:3], world_to_obj[3:]))

    def _reset_joint_state(self, num_steps):
        njo = min(p.getNumJoints(self._p_robot), 7)
        old_joints = np.array([p.getJointState(self._p_robot, joint_idx)[0]
                               for joint_idx in range(njo)])
        new_joints = np.zeros(njo)
        for i in range(1, num_steps+1):
            interpolated = old_joints*(1-i/num_steps)+new_joints*i/num_steps
            for joint_idx in range(njo):
                p.resetJointState(self._p_robot, joint_idx,
                                  interpolated[joint_idx])

    def _sample_initial_state(self):
        state = {}
        base_position = [-1.5, 0., ROBOT_Z]
        base_orientation = [0., 0., 0., 1.]
        p.resetBasePositionAndOrientation(self._p_robot,
                                          base_position,
                                          base_orientation)

        # Set up box and target states.
        for box in self._boxes:
            box_state = {
                "pose": [0., 0., 0., 0., 0., 0., 1.]
            }
            state[box] = box_state

        target_state = {
            "pose": [1.5, 0., 0., 0., 0., 0., 1.]
        }
        state[self._targets[0]] = target_state
        for p_box in self._p_boxes:
            p.resetBasePositionAndOrientation(
                p_box, box_state["pose"][0:3], box_state["pose"][3:])
        p.resetBasePositionAndOrientation(
            self._p_targets[0], target_state["pose"][0:3], target_state["pose"][3:])

        # Set up world state.
        world_state = {}
        world_state["base_pose"] = base_position+base_orientation
        state[self._world] = world_state

        self._setup_motion_planner(state)
        return state

    def _setup_motion_planner(self, state):
        def _sample_fn(_):
            return [np.random.uniform(low=-ENV_WIDTH/2+COLLISION_TOLERANCE,
                                      high=ENV_WIDTH/2-COLLISION_TOLERANCE),
                    np.random.uniform(low=-ENV_HEIGHT/2+COLLISION_TOLERANCE,
                                      high=ENV_HEIGHT/2-COLLISION_TOLERANCE)]
        def _extend_fn(pt1, pt2):
            pt1 = np.array(pt1)
            pt2 = np.array(pt2)
            num = int(np.ceil(max(abs(pt1-pt2))))*10
            if num == 0:
                yield pt2, True
            for i in range(1, num+1):
                yield np.r_[pt1*(1-i/num)+pt2*i/num], True
        def _collision_fn(pt):
            x, y = pt
            for box in self._boxes:
                if x < state[box]['pose'][0] + OBJ_SIZE / 2 + COLLISION_TOLERANCE and \
                        x > state[box]['pose'][0] - OBJ_SIZE / 2 - COLLISION_TOLERANCE and \
                        y < state[box]['pose'][1] + OBJ_SIZE / 2 + COLLISION_TOLERANCE and \
                        y > state[box]['pose'][1] - OBJ_SIZE / 2 - COLLISION_TOLERANCE:
                    return "object"
            if x > 0 - COLLISION_TOLERANCE and x < 0 + COLLISION_TOLERANCE and \
                y > 0.9375 - (3 * (1 - DOOR_SCALE) / 2) / 2 and y < ENV_HEIGHT/2:
                return "wall"
            if x > 0 - COLLISION_TOLERANCE and x < 0 + COLLISION_TOLERANCE and \
                y > -ENV_HEIGHT/2 and y < -0.9375 + (3 * (1 - DOOR_SCALE) / 2) / 2:
                return "wall"
            return None
        def _distance_fn(pt1, pt2):
            return (pt1[0]-pt2[0])**2+(pt1[1]-pt2[1])**2
        self._rrt = BiRRT(_sample_fn, _extend_fn, _collision_fn, _distance_fn, self._seed)

    def _update_collision_motion_planner(self, state):
        def _collision_fn(pt):
            x, y = pt
            for box in self._boxes:
                if x < state[box]['pose'][0] + OBJ_SIZE / 2 + COLLISION_TOLERANCE and \
                        x > state[box]['pose'][0] - OBJ_SIZE / 2 - COLLISION_TOLERANCE and \
                        y < state[box]['pose'][1] + OBJ_SIZE / 2 + COLLISION_TOLERANCE and \
                        y > state[box]['pose'][1] - OBJ_SIZE / 2 - COLLISION_TOLERANCE:
                    return "object"
            if x > 0 - COLLISION_TOLERANCE and x < 0 + COLLISION_TOLERANCE and \
                y > 0.9375 - (3 * (1 - DOOR_SCALE) / 2) / 2 and y < ENV_HEIGHT/2:
                return "wall"
            if x > 0 - COLLISION_TOLERANCE and x < 0 + COLLISION_TOLERANCE and \
                y > -ENV_HEIGHT/2 and y < -0.9375 + (3 * (1 - DOOR_SCALE) / 2) / 2:
                return "wall"
            return None
        self._rrt.update_collision_fn(_collision_fn)

    @staticmethod
    def _is_in_collision(pt, p_obj):
        (obj_x, obj_y, obj_z), _ = p.getBasePositionAndOrientation(p_obj)
        if obj_z < 0:
            return False
        # NOTE: using full extents instead of half extents intentionally,
        # so robot can't get too close to an object.
        return abs(obj_x - pt[0]) < OBJ_SIZE and abs(obj_y - pt[1]) < OBJ_SIZE

    @property
    def literal_goal(self):
        return {self.GoalReach()}

    @property
    def discrete_predicates(self):
        return self._all_predicates - self._continuous_predicates

    def sample_IsPose(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidGrasp.
        Return dict from predicate argument index to value.
        """
        _ = rng  # unused
        posex, posey = state[obj]['pose'][0] - 1., state[obj]['pose'][1]
        return {0: posex, 1: posey}