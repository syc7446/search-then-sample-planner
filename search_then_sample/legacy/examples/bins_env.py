"""Bins environment.
"""

import numpy as np
import pybullet as p
from search_then_sample.legacy.utils.pybullet_utils import get_kinematic_chain, inverse_kinematics
import search_then_sample.legacy.utils.pybullet_controllers as controllers
import search_then_sample.legacy.utils.structs as structs
import search_then_sample.legacy.utils.constants as constants
from search_then_sample.legacy.utils.env_base import Environment, EnvironmentFailure
from search_then_sample.legacy.utils.utils import WORLD, get_asset_path


class BinsEnvironment(Environment):
    """Bins environment.
    """
    # Object types
    _obj_type = structs.Type("obj")
    _target_type = structs.Type("target")
    _xbase_type = structs.ContinuousType("xbase")
    _ybase_type = structs.ContinuousType("ybase")
    _zbase_type = structs.ContinuousType("zbase")
    _xgrip_type = structs.ContinuousType("xgrip")
    _ygrip_type = structs.ContinuousType("ygrip")
    _zgrip_type = structs.ContinuousType("zgrip")
    _xbase_type.set_domain([-10, 10])
    _ybase_type.set_domain([-10, 10])
    _zbase_type.set_domain([-10, 10])
    _xgrip_type.set_domain([-10, 10])
    _ygrip_type.set_domain([-10, 10])
    _zgrip_type.set_domain([-10, 10])

    # Constant world object
    _world = WORLD

    # Predicates
    OnTable = structs.Predicate("OnTable", 1, [_obj_type])
    Holding = structs.Predicate("Holding", 1, [_obj_type])
    HoldingSide = structs.Predicate("HoldingSide", 1, [_obj_type])
    HoldingTop = structs.Predicate("HoldingTop", 1, [_obj_type])
    InShelf = structs.Predicate("InShelf", 1, [_obj_type])
    InBox = structs.Predicate("InBox", 1, [_obj_type])
    HandEmpty = structs.Predicate("HandEmpty", 0, [])
    HandFull = structs.Predicate("HandFull", 0, [])
    IsTargetForObj = structs.Predicate("IsTargetForObj", 2,
                                       [_target_type, _obj_type])
    TargetInShelf = structs.Predicate("TargetInShelf", 1, [_target_type])
    TargetInBox = structs.Predicate("TargetInBox", 1, [_target_type])
    IsValidGrasp = structs.Predicate("IsValidGrasp", 7,
                                     [_xbase_type, _ybase_type, _zbase_type,
                                      _xgrip_type, _ygrip_type, _zgrip_type,
                                      _obj_type])
    IsValidPlace = structs.Predicate("IsValidPlace", 7,
                                     [_xbase_type, _ybase_type, _zbase_type,
                                      _xgrip_type, _ygrip_type, _zgrip_type,
                                      _target_type])
    _all_predicates = {OnTable, Holding, HoldingSide, HoldingTop,
                       InShelf, InBox, HandEmpty, HandFull,
                       IsTargetForObj, TargetInShelf, TargetInBox,
                       IsValidGrasp, IsValidPlace}
    _all_predicate_names_to_preds = {p.name: p for p in _all_predicates}
    _continuous_predicates = {pred for pred in _all_predicates
                              if any(t.is_continuous for t in pred.var_types)}

    # Actions
    Pick = structs.Predicate("Pick", 7,
                             [_obj_type,
                              _xbase_type, _ybase_type, _zbase_type,
                              _xgrip_type, _ygrip_type, _zgrip_type])
    Place = structs.Predicate("Place", 8,
                              [_target_type, _obj_type,
                               _xbase_type, _ybase_type, _zbase_type,
                               _xgrip_type, _ygrip_type, _zgrip_type])
    action_predicates = {Pick, Place}

    def __init__(self, num_objs, seed):
        super().__init__(num_objs, seed)
        self._num_targets = num_objs  # each object has a target
        self._objs = []
        self._targets = []
        for i in range(self._num_objs):
            self._objs.append(self._obj_type("obj{}".format(i)))
            self._targets.append(self._target_type("target{}".format(i)))
        self._initial_pybullet_setup()
        self._objs_to_obj_ids = {}
        self._targets_to_target_ids = {}
        self._transmodel_cache = {}

    def parse_state(self, state):
        # Build literal set.
        lits = set()
        predicate_names = self._all_predicate_names_to_preds.keys()
        obj_radius = constants.BINS_OBJ_RADIUS
        if state[self._world]["cur_holding_tf"] is not None:
            held_obj, _, _, top_or_side = state[self._world]["cur_holding_tf"]
        else:
            held_obj = None
        for pred_name in predicate_names:
            pred = self._all_predicate_names_to_preds[pred_name]
            for obj, target in zip(self._objs, self._targets):
                obj_pose = state[obj]["pose"]
                target_pose = state[target]["pose"]
                if pred_name == "IsTargetForObj":
                    lits.add(pred(target, obj))
                if pred_name == "TargetInBox" and target_pose[1] < 1:
                    lits.add(pred(target))
                if pred_name == "TargetInShelf" and target_pose[1] > 1:
                    lits.add(pred(target))
                if obj == held_obj:
                    continue
                if pred_name == "OnTable" and \
                   -0.9 < obj_pose[1] < -0.1 and \
                   0.2 < obj_pose[2] < 0.4:
                    lits.add(pred(obj))
                if pred_name == "InShelf" and obj_pose[1] > 1 and \
                   target_pose[1]-2*obj_radius < obj_pose[1] and \
                   obj_pose[1] < target_pose[1]+2*obj_radius and \
                   0.2 < obj_pose[2] < 0.4:
                    lits.add(pred(obj))
                if pred_name == "InBox" and obj_pose[1] < 1 and \
                   np.sum(np.subtract(obj_pose, target_pose)**2) < 0.1:
                    lits.add(pred(obj))
            if pred_name == "HoldingTop" and held_obj is not None and \
               top_or_side == "top":
                lits.add(pred(held_obj))
            if pred_name == "HoldingSide" and held_obj is not None and \
               top_or_side == "side":
                lits.add(pred(held_obj))
            if pred_name == "Holding" and held_obj is not None:
                lits.add(pred(held_obj))
            if pred_name == "HandEmpty" and held_obj is None:
                lits.add(pred())
            if pred_name == "HandFull" and held_obj is not None:
                lits.add(pred())
        return lits

    def get_train_initial_states(self):
        return [self._sample_initial_state() for _ in range(10)]

    def simulate(self, state, action):
        # First check whether the given continuous arguments are good.
        # If not, quickly give up, without even trying the controller.
        if not self._good_continuous_args(state, action):
            hl_state = self.parse_state(state)
            reward = int(self.literal_goal.issubset(hl_state))
            done = (reward == 1)
            return state, reward, done
        sa_hashable = SAHashable(state, action, self._world, self._objs)
        if sa_hashable in self._transmodel_cache:
            return self._transmodel_cache[sa_hashable]
        next_state = {k: v.copy() for k, v in state.items()}
        # Reset pybullet state.
        p.resetBasePositionAndOrientation(
            self._fetch_id, state[self._world]["base_position"],
            [0, 0, 0, 1], physicsClientId=self._physics_client_id)
        for joint_idx, joint_val in zip(
                self._arm_joints, state[self._world]["joints"]):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)
        for obj in self._objs:
            p.resetBasePositionAndOrientation(
                self._objs_to_obj_ids[obj],
                state[obj]["pose"][:3], state[obj]["pose"][3:],
                physicsClientId=self._physics_client_id)
        for target in self._targets:
            p.resetBasePositionAndOrientation(
                self._targets_to_target_ids[target],
                state[target]["pose"][:3], state[target]["pose"][3:],
                physicsClientId=self._physics_client_id)
        held_constraint_id = None
        if state[self._world]["cur_holding_tf"] is not None:
            _, obj_id, base_to_obj, _ = state[self._world]["cur_holding_tf"]
            held_constraint_id = self._make_constraint(obj_id, base_to_obj)
        # Follow controller and step environment.
        controller_name = "Bins"+action.predicate.name+"Controller"
        action_vars = [var.value if var.is_continuous else var
                       for var in action.variables]
        if action.predicate == self.Pick:
            controller_args = [action_vars[0]]+action_vars[4:]
            base_position = action_vars[1:4]
            if controller_args[3] > 0.2+constants.BINS_OBJ_HEIGHT:  # top grasp
                top_or_side = "top"
                target_ee_orien = self._ee_orn_down
            else:  # we're side grasping
                top_or_side = "side"
                target_ee_orien = self._ee_orn_side
        elif action.predicate == self.Place:
            controller_args = action_vars[:2]+action_vars[5:]
            base_position = action_vars[2:5]
            if controller_args[3] < 1:  # top-place into the bin
                top_or_side = "top"
                target_ee_orien = self._ee_orn_down
            else:  # side-place into the shelf
                top_or_side = "side"
                target_ee_orien = self._ee_orn_side
        p.resetBasePositionAndOrientation(  # update base pose based on action
            self._fetch_id, base_position, [0, 0, 0, 1],
            physicsClientId=self._physics_client_id)
        if held_constraint_id is not None:  # also update the held object's pose
            _, obj_id, _, _ = state[self._world]["cur_holding_tf"]
            base_position_delta = (np.array(base_position)-
                                   state[self._world]["base_position"])
            cur_pose = p.getBasePositionAndOrientation(
                obj_id, physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                obj_id, np.add(cur_pose[0], base_position_delta),
                cur_pose[1], physicsClientId=self._physics_client_id)
        controller = getattr(controllers, controller_name)(
            controller_args, self._all_predicate_names_to_preds, self._world)
        orien_finished = False
        control_loop_itr = 0
        lits = self.parse_state(next_state)
        while True:
            control_loop_itr += 1
            current_position, current_orien = p.getLinkState(
                self._fetch_id, self._ee_id,
                physicsClientId=self._physics_client_id)[4:6]
            orien_finished = (orien_finished or
                              np.sum(np.subtract(
                                  current_orien,
                                  target_ee_orien)**2) < controllers.ATOL*10)
            if orien_finished:
                ee_orien_to_use = target_ee_orien
                action, done = controller.step(lits, next_state)
                action *= 0.05
                ee_delta, finger_action = action[:3], action[3]
                target_position = np.add(current_position, ee_delta)
            else:
                # Fix orientation.
                ee_orien_to_use = np.add(current_orien, np.subtract(
                    target_ee_orien, current_orien)/10)
                action, done = np.zeros(4), False
                target_position = current_position
                finger_action = 0
            joint_values = inverse_kinematics(
                self._fetch_id, self._ee_id, target_position,
                ee_orien_to_use, self._arm_joints,
                physics_client_id=self._physics_client_id)
            # Set arm joint motors.
            for joint_idx, joint_val in zip(self._arm_joints, joint_values):
                p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                        jointIndex=joint_idx,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=joint_val,
                                        physicsClientId=self._physics_client_id)
            # Set finger joint motors.
            for finger_id in [self._left_finger_id, self._right_finger_id]:
                current_val = p.getJointState(
                    self._fetch_id, finger_id,
                    physicsClientId=self._physics_client_id)[0]
                target_val = current_val+finger_action
                p.setJointMotorControl2(bodyIndex=self._fetch_id,
                                        jointIndex=finger_id,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPosition=target_val,
                                        physicsClientId=self._physics_client_id)
            # Handle grasping or letting go.
            left_finger_pos = next_state[self._world]["left_finger_pos"]
            if held_constraint_id is not None and left_finger_pos > 0.04:
                p.removeConstraint(held_constraint_id,
                                   physicsClientId=self._physics_client_id)
                # Step the environment once to propagate constraint removal
                p.stepSimulation(physicsClientId=self._physics_client_id)
                held_constraint_id = None
                next_state[self._world]["cur_holding_tf"] = None
            elif held_constraint_id is None and left_finger_pos < 0.04:
                held_object = self._get_held_obj(next_state)
                if held_object is not None:
                    base_to_world = np.r_[p.invertTransform(*p.getLinkState(
                        self._fetch_id, self._ee_id,
                        physicsClientId=self._physics_client_id)[:2])]
                    obj_id = self._objs_to_obj_ids[held_object]
                    world_to_obj = np.r_[p.getBasePositionAndOrientation(
                        obj_id, physicsClientId=self._physics_client_id)]
                    base_to_obj = p.invertTransform(*p.multiplyTransforms(
                        base_to_world[:3], base_to_world[3:],
                        world_to_obj[:3], world_to_obj[3:]))
                    held_constraint_id = self._make_constraint(
                        obj_id, base_to_obj)
                    next_state[self._world]["cur_holding_tf"] = (
                        held_object, obj_id, base_to_obj, top_or_side)
            for _ in range(25):  # let action run in environment
                p.stepSimulation(physicsClientId=self._physics_client_id)
            # Update fields in next_state.
            base_position = p.getBasePositionAndOrientation(
                self._fetch_id, physicsClientId=self._physics_client_id)[0]
            for obj in self._objs:
                obj_id = self._objs_to_obj_ids[obj]
                pos, orn = p.getBasePositionAndOrientation(
                    obj_id, physicsClientId=self._physics_client_id)
                next_state[obj]["pose"] = np.r_[pos, orn]
            joint_pos = [x[0] for x in p.getJointStates(
                self._fetch_id, self._arm_joints,
                physicsClientId=self._physics_client_id)]
            gripper_position, gripper_orien = p.getLinkState(
                self._fetch_id, self._ee_id,
                physicsClientId=self._physics_client_id)[4:6]
            left_finger_pos = p.getJointState(
                self._fetch_id, self._left_finger_id,
                physicsClientId=self._physics_client_id)[0]
            next_state[self._world]["base_position"] = base_position
            next_state[self._world]["joints"] = joint_pos
            next_state[self._world]["gripper_position"] = gripper_position
            next_state[self._world]["gripper_orien"] = gripper_orien
            next_state[self._world]["left_finger_pos"] = left_finger_pos
            # cur_holding_tf was updated above, when we handled grasping
            if done or control_loop_itr > 500:
                for _ in range(250):  # let the world settle
                    p.stepSimulation(physicsClientId=self._physics_client_id)
                break
        # Controller is done, so assert that all velocities are close to 0.
        error_message = None
        joint_vel = [x[1] for x in p.getJointStates(
            self._fetch_id, self._arm_joints,
            physicsClientId=self._physics_client_id)]
        if np.linalg.norm(joint_vel) > constants.VEL_TOL:
            error_message = "Robot moving with velocity {}".format(
                np.linalg.norm(joint_vel))
        for obj_id in self._objs_to_obj_ids.values():
            if error_message is not None:
                break
            linear_vels = p.getBaseVelocity(
                obj_id, physicsClientId=self._physics_client_id)[0]
            if np.linalg.norm(linear_vels) > constants.VEL_TOL:
                error_message = "Obj moving with velocity {}".format(
                    np.linalg.norm(linear_vels))
        # Before raising an error or returning, remove constraints
        if next_state[self._world]["cur_holding_tf"] is not None:
            # Remove this constraint, since this function should be stateless.
            p.removeConstraint(held_constraint_id,
                               physicsClientId=self._physics_client_id)
            # Step the environment once to propagate constraint removal
            p.stepSimulation(physicsClientId=self._physics_client_id)
        # Now it's safe to raise an exception if necessary
        if error_message is not None:
            raise EnvironmentFailure(error_message)
        hl_next_state = self.parse_state(next_state)
        reward = int(self.literal_goal.issubset(hl_next_state))
        done = (reward == 1)
        self._transmodel_cache[sa_hashable] = (next_state, reward, done)
        return next_state, reward, done

    @property
    def literal_goal(self):
        shelf_goals = {self.InShelf(obj) for obj in self._objs[:-1]}
        box_goals = {self.InBox(self._objs[-1])}
        return shelf_goals | box_goals

    @property
    def discrete_predicates(self):
        return self._all_predicates-self._continuous_predicates

    def sample_IsValidGrasp(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidGrasp.
        Return dict from predicate argument index to value.
        """
        if rng.randint(2) == 0:
            ideal = self._get_ideal_topgrasp(state, obj)
        else:
            ideal = self._get_ideal_sidegrasp(state, obj)
        basex, basey, basez, gripx, gripy, gripz = ideal
        return {0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def sample_IsValidPlace(self, state, target, rng=None):
        """Sample values for continuous arguments of IsValidPlace.
        Return dict from predicate argument index to value.
        """
        _ = rng  # unused
        ideal = self._get_ideal_place(state, target)
        basex, basey, basez, gripx, gripy, gripz = ideal
        return {0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    @staticmethod
    def _get_ideal_topgrasp(state, obj):
        obj_position = state[obj]["pose"][:3]
        base = [constants.BINS_ROBOT_X+0.2, obj_position[1],
                constants.BINS_ROBOT_Z]
        grip = np.add(obj_position, [0, 0, 0.15])
        return np.r_[base, grip]

    @staticmethod
    def _get_ideal_sidegrasp(state, obj):
        obj_position = state[obj]["pose"][:3]
        base = [constants.BINS_ROBOT_X, obj_position[1],
                constants.BINS_ROBOT_Z]
        grip = np.add(obj_position, [-0.15, 0, 0.05])
        return np.r_[base, grip]

    @staticmethod
    def _get_ideal_place(state, target):
        target_position = state[target]["pose"][:3]
        if target_position[1] < 1:  # target is in the box
            base = [constants.BINS_ROBOT_X+0.2, target_position[1],
                    constants.BINS_ROBOT_Z]
            grip = np.add(target_position, [0, 0, 0.32])
        else:  # target is in the shelf
            base = [constants.BINS_ROBOT_X-0.2, target_position[1],
                    constants.BINS_ROBOT_Z+0.02]
            grip = np.add(target_position, [-0.3, 0, 0.03])
        return np.r_[base, grip]

    def _good_continuous_args(self, state, action):
        if action.predicate == self.Pick:
            # Must either be a valid top grasp or a valid side grasp.
            obj = action.variables[0]
            basex, basey, basez, gripx, gripy, gripz = action.variables[1:]
            return (np.all([basex.value, basey.value, basez.value,
                            gripx.value, gripy.value, gripz.value] ==
                           self._get_ideal_topgrasp(state, obj)) or
                    np.all([basex.value, basey.value, basez.value,
                            gripx.value, gripy.value, gripz.value] ==
                           self._get_ideal_sidegrasp(state, obj)))
        if action.predicate == self.Place:
            # Must be a valid place.
            target = action.variables[0]
            basex, basey, basez, gripx, gripy, gripz = action.variables[2:]
            return np.all([basex.value, basey.value, basez.value,
                           gripx.value, gripy.value, gripz.value] ==
                          self._get_ideal_place(state, target))
        raise Exception(f"Unexpected action: {action}")

    def _initial_pybullet_setup(self):
        """One-time pybullet setup stuff.
        """
        # Load things into environment.
        camera_distance = 1.7
        yaw = 270
        pitch = -15
        camera_target = [1.05, 0.5, 0.42]
        if constants.USE_VIEWER:
            self._physics_client_id = p.connect(p.GUI)
        else:
            self._physics_client_id = p.connect(p.DIRECT)
        p.resetDebugVisualizerCamera(camera_distance, yaw, pitch, camera_target,
                                     physicsClientId=self._physics_client_id)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0,
                                   physicsClientId=self._physics_client_id)
        p.resetSimulation(physicsClientId=self._physics_client_id)
        p.setGravity(0., 0., -10., physicsClientId=self._physics_client_id)
        p.setAdditionalSearchPath("envs/assets/")
        p.loadURDF(get_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True, physicsClientId=self._physics_client_id)
        self._fetch_id = p.loadURDF(
            get_asset_path("urdf/robots/fetch.urdf"), useFixedBase=True,
            physicsClientId=self._physics_client_id)
        # Get joints info.
        joint_names = [p.getJointInfo(
            self._fetch_id, i,
            physicsClientId=self._physics_client_id)[1].decode("utf-8")
                       for i in range(p.getNumJoints(
                           self._fetch_id,
                           physicsClientId=self._physics_client_id))]
        self._ee_id = joint_names.index("gripper_axis")
        self._ee_orn_down = p.getQuaternionFromEuler((0, np.pi/2, -np.pi))
        self._ee_orn_side = p.getQuaternionFromEuler((0, 0, 0))
        self._arm_joints = get_kinematic_chain(
            self._fetch_id, self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)
        self._init_joint_values = inverse_kinematics(
            self._fetch_id, self._ee_id, [1., 0, 0.75], self._ee_orn_down,
            self._arm_joints, physics_client_id=self._physics_client_id)
        # Add table.
        table_urdf = get_asset_path("urdf/table2.urdf")
        table_id = p.loadURDF(table_urdf, useFixedBase=True,
                              physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            table_id, (1.65, 0.5, 0.0), [0., 0., 0., 1.],
            physicsClientId=self._physics_client_id)
        # Add shelf.
        link_vis = []
        link_cols = []
        link_pos = []
        shelf_l = constants.BINS_SHELF_L
        shelf_d = constants.BINS_SHELF_D
        shelf_h = constants.BINS_SHELF_H
        shelf_box_t = constants.BINS_SHELF_BOX_T
        cur_height = 0.15+shelf_h/2
        for i in range(constants.BINS_SHELF_SLABS):
            # Left wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, 0.01/2, shelf_h/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, 0.01/2, shelf_h/2],
                rgbaColor=(0.6, 0.3, 0.0, shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, 1.5-shelf_l/2, cur_height])
            # Right wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, 0.01/2, shelf_h/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, 0.01/2, shelf_h/2],
                rgbaColor=(0.6, 0.3, 0.0, shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, 1.5+shelf_l/2, cur_height])
            # Back wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[0.01/2, shelf_l/2, shelf_h/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[0.01/2, shelf_l/2, shelf_h/2],
                rgbaColor=(0.7, 0.4, 0.1, shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65+shelf_d/2, 1.5, cur_height])
            # Bottom wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, shelf_l/2, 0.01/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, shelf_l/2, 0.01/2],
                rgbaColor=(0.7, 0.4, 0.1, shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, 1.5, cur_height])
            if i == constants.BINS_SHELF_SLABS-1:
                continue
            # Top wall.
            link_cols.append(p.createCollisionShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, shelf_l/2, 0.01/2],
                physicsClientId=self._physics_client_id))
            link_vis.append(p.createVisualShape(
                p.GEOM_BOX, halfExtents=[shelf_d/2, shelf_l/2, 0.01/2],
                rgbaColor=(0.7, 0.4, 0.1, shelf_box_t),
                physicsClientId=self._physics_client_id))
            link_pos.append([1.65, 1.5, cur_height+shelf_h])
            cur_height += shelf_h
        p.createMultiBody(
            linkMasses=[10 for _ in link_pos],
            linkCollisionShapeIndices=link_cols,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkInertialFramePositions=[[0, 0, 0] for _ in link_pos],
            linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkParentIndices=[0 for _ in link_pos],
            linkJointTypes=[p.JOINT_FIXED for _ in link_pos],
            linkJointAxis=[[0, 0, 0] for _ in link_pos],
            physicsClientId=self._physics_client_id)
        # Add box.
        link_vis = []
        link_cols = []
        link_pos = []
        box_y = constants.BINS_BOX_Y
        box_s = constants.BINS_BOX_S
        # Left wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[box_s/2, 0.01/2, box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[box_s/2, 0.01/2, box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65, box_y-box_s/2, 0.2])
        # Right wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[box_s/2, 0.01/2, box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[box_s/2, 0.01/2, box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65, box_y+box_s/2, 0.2])
        # Front wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.01/2, box_s/2, box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.01/2, box_s/2, box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65-box_s/2, box_y, 0.2])
        # Right wall.
        link_cols.append(p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.01/2, box_s/2, box_s/2],
            physicsClientId=self._physics_client_id))
        link_vis.append(p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.01/2, box_s/2, box_s/2],
            rgbaColor=(0.6, 0.3, 0.0, shelf_box_t),
            physicsClientId=self._physics_client_id))
        link_pos.append([1.65+box_s/2, box_y, 0.2])
        p.createMultiBody(
            linkMasses=[10 for _ in link_pos],
            linkCollisionShapeIndices=link_cols,
            linkVisualShapeIndices=link_vis,
            linkPositions=link_pos,
            linkOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkInertialFramePositions=[[0, 0, 0] for _ in link_pos],
            linkInertialFrameOrientations=[[0, 0, 0, 1] for _ in link_pos],
            linkParentIndices=[0 for _ in link_pos],
            linkJointTypes=[p.JOINT_FIXED for _ in link_pos],
            linkJointAxis=[[0, 0, 0] for _ in link_pos],
            physicsClientId=self._physics_client_id)

    def _rebuild_pybullet_objects(self):
        # Remove any existing objects.
        for obj_id in self._objs_to_obj_ids.values():
            p.removeBody(obj_id, physicsClientId=self._physics_client_id)
        self._objs_to_obj_ids = {}
        # Add new objects.
        colors = [
            (0.95, 0.05, 0.1, 1.),
            (0.05, 0.95, 0.1, 1.),
            (0.1, 0.05, 0.95, 1.),
            (0.4, 0.05, 0.6, 1.),
            (0.6, 0.4, 0.05, 1.),
            (0.05, 0.04, 0.6, 1.),
            (0.95, 0.95, 0.1, 1.),
            (0.95, 0.05, 0.95, 1.),
            (0.05, 0.95, 0.95, 1.),
        ]
        obj_height = constants.BINS_OBJ_HEIGHT
        obj_radius = constants.BINS_OBJ_RADIUS
        for i, obj in enumerate(self._objs):
            color = colors[i%len(colors)]
            mass, friction = 0.04, 1.2
            collision_id = p.createCollisionShape(
                p.GEOM_CYLINDER, radius=obj_radius, height=obj_height,
                physicsClientId=self._physics_client_id)
            visual_id = p.createVisualShape(
                p.GEOM_CYLINDER, radius=obj_radius, length=obj_height,
                rgbaColor=color, physicsClientId=self._physics_client_id)
            obj_id = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self._physics_client_id)
            p.changeDynamics(obj_id, -1, lateralFriction=friction,
                             physicsClientId=self._physics_client_id)
            self._objs_to_obj_ids[obj] = obj_id
        # Remove any existing targets.
        for target_id in self._targets_to_target_ids.values():
            p.removeBody(target_id, physicsClientId=self._physics_client_id)
        self._targets_to_target_ids = {}
        # Add new targets.
        box_s = constants.BINS_BOX_S
        shelf_h = constants.BINS_SHELF_H
        for i, target in enumerate(self._targets):
            color = colors[i%len(colors)]
            color = color[:-1]+(0.5,)  # make targets a bit transparent
            if i == len(self._targets)-1:
                half_extents = [box_s/2-0.005, box_s/2-0.005, 0.005]
            else:
                half_extents = [0.005, obj_radius, shelf_h/2-0.005]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents,
                physicsClientId=self._physics_client_id)
            visual_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                physicsClientId=self._physics_client_id)
            target_id = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0],
                baseOrientation=[0, 0, 0, 1],
                physicsClientId=self._physics_client_id)
            self._targets_to_target_ids[target] = target_id

    def _sample_initial_state(self):
        self._rebuild_pybullet_objects()
        state = {}
        # Set up object and target states.
        obj_ys = []
        obj_height = constants.BINS_OBJ_HEIGHT
        obj_radius = constants.BINS_OBJ_RADIUS
        box_y = constants.BINS_BOX_Y
        shelf_d = constants.BINS_SHELF_D
        shelf_h = constants.BINS_SHELF_H
        for obj, target in zip(self._objs, self._targets):
            while True:
                this_y = self._rng.uniform(-0.8, -0.2)
                if all(abs(this_y-other_y) > 3.5*obj_radius
                       for other_y in obj_ys):
                    break
            obj_ys.append(this_y)
            obj_state = {
                "pose": [1.75, this_y, 0.2+obj_height/2, 0, 0, 0, 1]
            }
            if obj == self._objs[-1]:
                target_state = {
                    "pose": [1.65, box_y, 0.2, 0, 0, 0, 1]
                }
            else:
                target_state = {
                    "pose": [1.65+shelf_d/2-0.02, this_y+2, 0.15+shelf_h,
                             0, 0, 0, 1]
                }
            state[obj] = obj_state
            state[target] = target_state
            p.resetBasePositionAndOrientation(
                self._objs_to_obj_ids[obj],
                obj_state["pose"][:3], obj_state["pose"][3:],
                physicsClientId=self._physics_client_id)
            p.resetBasePositionAndOrientation(
                self._targets_to_target_ids[target],
                target_state["pose"][:3], target_state["pose"][3:],
                physicsClientId=self._physics_client_id)
        # Reset robot joints and base.
        for joint_idx, joint_val in zip(
                self._arm_joints, self._init_joint_values):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)
        base_position = [constants.BINS_ROBOT_X, 0.5,
                         constants.BINS_ROBOT_Z]
        base_orientation = [0., 0., 0., 1.]
        p.resetBasePositionAndOrientation(
            self._fetch_id, base_position, base_orientation,
            physicsClientId=self._physics_client_id)
        # Assert that all velocities are close to 0.
        joint_pos = [x[0] for x in p.getJointStates(
            self._fetch_id, self._arm_joints,
            physicsClientId=self._physics_client_id)]
        joint_vel = [x[1] for x in p.getJointStates(
            self._fetch_id, self._arm_joints,
            physicsClientId=self._physics_client_id)]
        if np.linalg.norm(joint_vel) > constants.VEL_TOL:
            raise EnvironmentFailure("Robot moving with velocity {}".format(
                np.linalg.norm(joint_vel)))
        for obj_id in self._objs_to_obj_ids.values():
            linear_vels = p.getBaseVelocity(
                obj_id, physicsClientId=self._physics_client_id)[0]
            if np.linalg.norm(linear_vels) > constants.VEL_TOL:
                raise EnvironmentFailure("Obj moving with velocity {}".format(
                    np.linalg.norm(linear_vels)))
        # Set up world state.
        base_position = p.getBasePositionAndOrientation(
            self._fetch_id, physicsClientId=self._physics_client_id)[0]
        gripper_position, gripper_orien = p.getLinkState(
            self._fetch_id, self._ee_id,
            physicsClientId=self._physics_client_id)[4:6]
        left_finger_pos = p.getJointState(
            self._fetch_id, self._left_finger_id,
            physicsClientId=self._physics_client_id)[0]
        world_state = {}
        world_state["base_position"] = base_position
        world_state["joints"] = joint_pos
        world_state["gripper_position"] = gripper_position
        world_state["gripper_orien"] = gripper_orien
        world_state["left_finger_pos"] = left_finger_pos
        world_state["cur_holding_tf"] = None
        state[self._world] = world_state
        return state

    def _make_constraint(self, obj_id, base_to_obj):
        num_current_constraints = p.getNumConstraints(
            physicsClientId=self._physics_client_id)
        assert num_current_constraints == 0
        return p.createConstraint(
            parentBodyUniqueId=self._fetch_id,
            parentLinkIndex=self._ee_id,
            childBodyUniqueId=obj_id,
            childLinkIndex=-1,
            jointType=p.JOINT_FIXED,
            jointAxis=[0, 0, 0],
            parentFramePosition=[0, 0, 0],
            childFramePosition=base_to_obj[0],
            parentFrameOrientation=[0, 0, 0, 1],
            childFrameOrientation=base_to_obj[1],
            physicsClientId=self._physics_client_id)

    def _get_held_obj(self, state):
        gripper_position = state[self._world]["gripper_position"]
        for obj in self._objs:
            obj_position = state[obj]["pose"][:3]
            if np.sum(np.subtract(gripper_position,
                                  obj_position)**2) < constants.GRASPING_TOL:
                return obj
        return None


class SAHashable:
    """Defines a hashable object so that we can cache the transition model.
    Hashes a state and action.
    """
    def __init__(self, state, action, world, objs):
        self.state = state
        self.action = action
        self.world = world
        self.objs = objs

    def to_tuple(self):
        """Convert to tuple for hashing and checking equality.
        """
        lst = []
        # Object poses.
        for obj in self.objs:
            lst.extend(self.state[obj]["pose"])
        # Robot base position.
        lst.extend(self.state[self.world]["base_position"])
        # Robot gripper position.
        lst.extend(self.state[self.world]["gripper_position"])
        # Continuous action arguments.
        lst.extend([var.value for var in self.action.variables
                    if var.is_continuous])
        # Everything so far is a float, so round it.
        lst = list(np.round(lst, 5))
        # Action name and discrete action arguments.
        lst.append(self.action.predicate.name)
        lst.extend([var.name for var in self.action.variables
                    if not var.is_continuous])
        return tuple(lst)

    def __hash__(self):
        return hash(self.to_tuple())

    def __eq__(self, other):
        return self.to_tuple() == other.to_tuple()
