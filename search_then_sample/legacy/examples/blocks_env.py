"""Blocks environment.
"""

import itertools
import numpy as np
import pybullet as p
from search_then_sample.legacy.utils.pybullet_utils import get_kinematic_chain, inverse_kinematics
import search_then_sample.legacy.utils.pybullet_controllers as controllers
import search_then_sample.legacy.utils.structs as structs
import search_then_sample.legacy.utils.constants as constants
from search_then_sample.legacy.utils.env_base import Environment, EnvironmentFailure
from search_then_sample.legacy.utils.utils import WORLD, get_asset_path


class BlocksEnvironment(Environment):
    """Blocks environment.
    """
    # Object types
    _block_type = structs.Type("block")
    _xdim_type = structs.ContinuousType("xdim")
    _ydim_type = structs.ContinuousType("ydim")
    _zdim_type = structs.ContinuousType("zdim")
    _xdim_type.set_domain([-10, 10])
    _ydim_type.set_domain([-10, 10])
    _zdim_type.set_domain([-10, 10])

    # Constant world object
    _world = WORLD

    # Predicates
    On = structs.Predicate("On", 2, [_block_type, _block_type])
    OnTable = structs.Predicate("OnTable", 1, [_block_type])
    Holding = structs.Predicate("Holding", 1, [_block_type])
    Clear = structs.Predicate("Clear", 1, [_block_type])
    HandEmpty = structs.Predicate("HandEmpty", 0, [])
    HandFull = structs.Predicate("HandFull", 0, [])
    IsValidGrasp = structs.Predicate("IsValidGrasp", 4,
                                     [_xdim_type, _ydim_type,
                                      _zdim_type, _block_type])
    IsValidPlaceOnBlock = structs.Predicate("IsValidPlaceOnBlock", 4,
                                            [_xdim_type, _ydim_type,
                                             _zdim_type, _block_type])
    IsValidPlaceOnTable = structs.Predicate("IsValidPlaceOnTable", 3,
                                            [_xdim_type, _ydim_type,
                                             _zdim_type])
    _all_predicates = {On, OnTable, Holding, Clear, HandEmpty, HandFull,
                       IsValidGrasp, IsValidPlaceOnBlock, IsValidPlaceOnTable}
    _all_predicate_names_to_preds = {p.name: p for p in _all_predicates}
    _continuous_predicates = {pred for pred in _all_predicates
                              if any(t.is_continuous for t in pred.var_types)}

    # Actions
    PickupFromBlock = structs.Predicate("PickupFromBlock", 5,
                                        [_block_type, _block_type, _xdim_type,
                                         _ydim_type, _zdim_type])
    PickupFromTable = structs.Predicate("PickupFromTable", 4,
                                        [_block_type, _xdim_type,
                                         _ydim_type, _zdim_type])
    PutOnBlock = structs.Predicate("PutOnBlock", 5,
                                   [_block_type, _block_type, _xdim_type,
                                    _ydim_type, _zdim_type])
    PutOnTable = structs.Predicate("PutOnTable", 4,
                                   [_block_type, _xdim_type,
                                    _ydim_type, _zdim_type])
    action_predicates = {PickupFromBlock, PickupFromTable,
                         PutOnBlock, PutOnTable}

    def __init__(self, num_objs, seed):
        super().__init__(num_objs, seed)
        self._num_blocks = self._num_objs
        self._min_num_piles = 2
        self._max_num_piles = 4
        self._blocks = []
        for i in range(self._num_blocks):
            self._blocks.append(self._block_type("block{}".format(i)))
        self._initial_pybullet_setup()
        self._blocks_to_block_ids = {}
        self._transmodel_cache = {}

    def parse_state(self, state):
        piles, held_block = self._get_piles(state)
        # Build literal set.
        lits = set()
        predicate_names = self._all_predicate_names_to_preds.keys()
        for pred_name in predicate_names:
            pred = self._all_predicate_names_to_preds[pred_name]
            if pred_name == "On":
                for pile in piles:
                    if len(pile) == 1:
                        continue
                    for below, above in zip(pile[:-1], pile[1:]):
                        lits.add(pred(above, below))
            if pred_name == "OnTable":
                for pile in piles:
                    lits.add(pred(pile[0]))
            if pred_name == "Holding" and held_block is not None:
                lits.add(pred(held_block))
            if pred_name == "Clear":
                for pile in piles:
                    lits.add(pred(pile[-1]))
            if pred_name == "HandEmpty" and held_block is None:
                lits.add(pred())
            if pred_name == "HandFull" and held_block is not None:
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
        sa_hashable = SAHashable(state, action, self._world, self._blocks)
        if sa_hashable in self._transmodel_cache:
            return self._transmodel_cache[sa_hashable]
        next_state = {k: v.copy() for k, v in state.items()}
        # Reset pybullet state.
        for joint_idx, joint_val in zip(
                self._arm_joints, state[self._world]["joints"]):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
                              physicsClientId=self._physics_client_id)
        for block in self._blocks:
            p.resetBasePositionAndOrientation(
                self._blocks_to_block_ids[block],
                state[block]["pose"][:3], state[block]["pose"][3:],
                physicsClientId=self._physics_client_id)
        held_constraint_id = None
        if state[self._world]["cur_holding_tf"] is not None:
            block_id, base_to_obj, _ = state[self._world]["cur_holding_tf"]
            held_constraint_id = self._make_constraint(block_id, base_to_obj)
        # Follow controller and step environment.
        controller_name = "Blocks"+action.predicate.name+"Controller"
        action_vars = [var.value if var.is_continuous else var
                       for var in action.variables]
        controller = getattr(controllers, controller_name)(
            action_vars, self._all_predicate_names_to_preds, self._world)
        lits = self.parse_state(next_state)
        while True:
            action, done = controller.step(lits, next_state)
            action *= 0.05
            ee_delta, finger_action = action[:3], action[3]
            current_position, _ = p.getLinkState(
                self._fetch_id, self._ee_id,
                physicsClientId=self._physics_client_id)[4:6]
            target_position = np.add(current_position, ee_delta)
            joint_values = inverse_kinematics(
                self._fetch_id, self._ee_id, target_position,
                self._ee_orientation, self._arm_joints,
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
                    block_id = self._blocks_to_block_ids[held_object]
                    world_to_obj = np.r_[p.getBasePositionAndOrientation(
                        block_id, physicsClientId=self._physics_client_id)]
                    base_to_obj = p.invertTransform(*p.multiplyTransforms(
                        base_to_world[:3], base_to_world[3:],
                        world_to_obj[:3], world_to_obj[3:]))
                    held_constraint_id = self._make_constraint(
                        block_id, base_to_obj)
                    next_state[self._world]["cur_holding_tf"] = (
                        block_id, base_to_obj, held_object)
            for _ in range(25):  # let action run in environment
                p.stepSimulation(physicsClientId=self._physics_client_id)
            # Update fields in next_state.
            for block in self._blocks:
                block_id = self._blocks_to_block_ids[block]
                pos, orn = p.getBasePositionAndOrientation(block_id)
                next_state[block]["pose"] = np.r_[pos, orn]
            joint_pos = [x[0] for x in p.getJointStates(
                self._fetch_id, self._arm_joints,
                physicsClientId=self._physics_client_id)]
            gripper_position, gripper_orien = p.getLinkState(
                self._fetch_id, self._ee_id,
                physicsClientId=self._physics_client_id)[4:6]
            left_finger_pos = p.getJointState(
                self._fetch_id, self._left_finger_id,
                physicsClientId=self._physics_client_id)[0]
            next_state[self._world]["joints"] = joint_pos
            next_state[self._world]["gripper_position"] = gripper_position
            next_state[self._world]["gripper_orien"] = gripper_orien
            next_state[self._world]["left_finger_pos"] = left_finger_pos
            # cur_holding_tf was updated above, when we handled grasping
            if done:
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
        for block_id in self._blocks_to_block_ids.values():
            if error_message is not None:
                break
            linear_vels = p.getBaseVelocity(
                block_id, physicsClientId=self._physics_client_id)[0]
            if np.linalg.norm(linear_vels) > constants.VEL_TOL:
                error_message = "Block moving with velocity {}".format(
                    np.linalg.norm(linear_vels))
        if held_constraint_id is not None:
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
        return {self.On(self._blocks[0], self._blocks[1]),
                self.On(self._blocks[1], self._blocks[2])}

    @property
    def discrete_predicates(self):
        return self._all_predicates-self._continuous_predicates

    def sample_IsValidGrasp(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidGrasp.
        Return dict from predicate argument index to value.
        """
        _ = rng  # unused
        ideal = self._get_ideal_grasp(state, obj)
        posex, posey, posez = ideal
        return {0: posex, 1: posey, 2: posez}

    def sample_IsValidPlaceOnBlock(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidPlaceOnBlock.
        Return dict from predicate argument index to value.
        """
        _ = rng  # unused
        ideal = self._get_ideal_placeonblock(state, obj)
        posex, posey, posez = ideal
        return {0: posex, 1: posey, 2: posez}

    def sample_IsValidPlaceOnTable(self, state, rng=None):
        """Sample values for continuous arguments of IsValidPlaceOnTable.
        Return dict from predicate argument index to value.
        """
        _ = rng  # unused
        ideal = self._get_ideal_placeontable(state)
        posex, posey, posez = ideal
        return {0: posex, 1: posey, 2: posez}

    @staticmethod
    def _get_ideal_grasp(state, obj):
        block_position = state[obj]["pose"][:3]
        return np.add(block_position, [0, 0, 0])

    @staticmethod
    def _get_ideal_placeonblock(state, obj):
        block_position = state[obj]["pose"][:3]
        return np.add(block_position, [0, 0, 0.1])

    def _get_ideal_placeontable(self, state):
        x = 1.25
        z = 0.15
        min_y, max_y = 0.4, 1.0
        block_ys = []
        for block in self._blocks:
            # Only look at blocks on the table
            if 0.1 < state[block]["pose"][2] < 0.3:
                block_ys.append(state[block]["pose"][1])
        best_y = min_y
        best_y_dist = 0
        for y in np.linspace(min_y, max_y, num=50):
            y_dist = np.inf
            for block_y in block_ys:
                y_dist = min(y_dist, abs(block_y-y))
            if y_dist > best_y_dist:
                best_y_dist = y_dist
                best_y = y
        return np.array([x, best_y, z])

    def _good_continuous_args(self, state, action):
        if action.predicate in (self.PickupFromBlock, self.PickupFromTable):
            obj = action.variables[0]
            posex, posey, posez = action.variables[-3:]
            return np.all([posex.value, posey.value, posez.value] ==
                          self._get_ideal_grasp(state, obj))
        if action.predicate == self.PutOnBlock:
            _, obj, posex, posey, posez = action.variables
            return np.all([posex.value, posey.value, posez.value] ==
                          self._get_ideal_placeonblock(state, obj))
        if action.predicate == self.PutOnTable:
            posex, posey, posez = action.variables[-3:]
            return np.all([posex.value, posey.value, posez.value] ==
                          self._get_ideal_placeontable(state))
        raise Exception(f"Unexpected action: {action}")

    def _initial_pybullet_setup(self):
        """One-time pybullet setup stuff.
        """
        # Load things into environment.
        base_position = [0.75, 0.7441, 0]
        base_orientation = [0., 0., 0., 1.]
        camera_distance = 1.5
        yaw = 90
        pitch = -24
        camera_target = [1.65, 0.75, 0.42]
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
        p.loadURDF(get_asset_path("urdf/plane.urdf"), [0, 0, -1],
                   useFixedBase=True, physicsClientId=self._physics_client_id)
        self._fetch_id = p.loadURDF(get_asset_path("urdf/robots/fetch.urdf"),
                                    useFixedBase=True,
                                    physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            self._fetch_id, base_position, base_orientation,
            physicsClientId=self._physics_client_id)
        # Get joints info.
        joint_names = [p.getJointInfo(
            self._fetch_id, i,
            physicsClientId=self._physics_client_id)[1].decode("utf-8")
                       for i in range(p.getNumJoints(
                           self._fetch_id,
                           physicsClientId=self._physics_client_id))]
        self._ee_id = joint_names.index("gripper_axis")
        self._ee_orientation = [1., 0., -1., 0.]
        self._arm_joints = get_kinematic_chain(
            self._fetch_id, self._ee_id,
            physics_client_id=self._physics_client_id)
        self._left_finger_id = joint_names.index("l_gripper_finger_joint")
        self._right_finger_id = joint_names.index("r_gripper_finger_joint")
        self._arm_joints.append(self._left_finger_id)
        self._arm_joints.append(self._right_finger_id)
        # Add table.
        table_urdf = get_asset_path("urdf/table.urdf")
        table_id = p.loadURDF(table_urdf, useFixedBase=True,
                              physicsClientId=self._physics_client_id)
        p.resetBasePositionAndOrientation(
            table_id, (1.65, 0.75, 0.0), [0., 0., 0., 1.],
            physicsClientId=self._physics_client_id)

    def _rebuild_pybullet_blocks(self):
        # Remove any existing blocks.
        for block_id in self._blocks_to_block_ids.values():
            p.removeBody(block_id, physicsClientId=self._physics_client_id)
        self._blocks_to_block_ids = {}
        # Add new blocks.
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
        for i, block in enumerate(self._blocks):
            color = colors[i%len(colors)]
            width, length, height = 0.075, 0.075, constants.BLOCKS_BLOCK_HEIGHT
            mass, friction = 0.04, 1.2
            orn_x, orn_y, orn_z, orn_w = 0, 0, 0, 1
            half_extents = [width/2, length/2, height/2]
            collision_id = p.createCollisionShape(
                p.GEOM_BOX, halfExtents=half_extents,
                physicsClientId=self._physics_client_id)
            visual_id = p.createVisualShape(
                p.GEOM_BOX, halfExtents=half_extents, rgbaColor=color,
                physicsClientId=self._physics_client_id)
            block_id = p.createMultiBody(
                baseMass=mass, baseCollisionShapeIndex=collision_id,
                baseVisualShapeIndex=visual_id, basePosition=[0, 0, 0],
                baseOrientation=[orn_x, orn_y, orn_z, orn_w],
                physicsClientId=self._physics_client_id)
            p.changeDynamics(block_id, -1, lateralFriction=friction,
                             physicsClientId=self._physics_client_id)
            self._blocks_to_block_ids[block] = block_id

    def _sample_initial_state(self):
        self._rebuild_pybullet_blocks()
        state = {}
        block_height = constants.BLOCKS_BLOCK_HEIGHT
        # Set up block states.
        block_counter = itertools.count()
        num_blocks = 0
        num_piles = self._rng.randint(
            self._min_num_piles, min(self._max_num_piles, self._num_blocks)+1)
        for pile in range(num_piles):
            if num_blocks == self._num_blocks:
                break
            if pile == num_piles-1:
                num_blocks_in_pile = self._num_blocks-num_blocks
            else:
                num_blocks_in_pile = self._rng.randint(
                    1, self._num_blocks-num_blocks+1)
            num_blocks += num_blocks_in_pile
            previous_block_top = 0.2
            for _ in range(num_blocks_in_pile):
                block = self._blocks[next(block_counter)]
                block_state = {
                    "pose": [1.25, 0.5+pile*0.2,
                             previous_block_top+block_height/2, 0, 0, 0, 1],
                }
                previous_block_top += block_height  # update previous top
                state[block] = block_state
                p.resetBasePositionAndOrientation(
                    self._blocks_to_block_ids[block],
                    block_state["pose"][:3], block_state["pose"][3:],
                    physicsClientId=self._physics_client_id)
        assert next(block_counter) == len(self._blocks)
        # Reset robot joints.
        initial_joint_values = inverse_kinematics(
            self._fetch_id, self._ee_id, [1., 0.7, 0.75], self._ee_orientation,
            self._arm_joints, physics_client_id=self._physics_client_id)
        for joint_idx, joint_val in zip(self._arm_joints, initial_joint_values):
            p.resetJointState(self._fetch_id, joint_idx, joint_val,
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
        for block_id in self._blocks_to_block_ids.values():
            linear_vels = p.getBaseVelocity(
                block_id, physicsClientId=self._physics_client_id)[0]
            if np.linalg.norm(linear_vels) > constants.VEL_TOL:
                raise EnvironmentFailure("Block moving with velocity {}".format(
                    np.linalg.norm(linear_vels)))
        # Set up world state.
        gripper_position, gripper_orien = p.getLinkState(
            self._fetch_id, self._ee_id,
            physicsClientId=self._physics_client_id)[4:6]
        left_finger_pos = p.getJointState(
            self._fetch_id, self._left_finger_id,
            physicsClientId=self._physics_client_id)[0]
        world_state = {}
        world_state["joints"] = joint_pos
        world_state["gripper_position"] = gripper_position
        world_state["gripper_orien"] = gripper_orien
        world_state["left_finger_pos"] = left_finger_pos
        world_state["cur_holding_tf"] = None
        state[self._world] = world_state
        return state

    def _get_piles(self, state):
        # First check whether we're holding a block.
        if state[self._world]["cur_holding_tf"] is None:
            held_block = None
        else:
            _, _, held_block = state[self._world]["cur_holding_tf"]
        # Find which blocks are vertically aligned, indicating a pile.
        piles = []
        for block in self._blocks:
            if block == held_block:
                continue
            x, y = state[block]["pose"][:2]
            contender_pile = None
            best_pile_dist = np.inf
            for pile in piles:
                for block_in_pile in pile:
                    base_x, base_y = state[block_in_pile]["pose"][:2]
                    contender_pile_dist = abs(x-base_x)+abs(y-base_y)
                    if contender_pile_dist < 1e-1 and \
                       contender_pile_dist < best_pile_dist:
                        contender_pile = pile
                        best_pile_dist = contender_pile_dist
            if contender_pile is None:
                piles.append([block])
            else:
                contender_pile.append(block)
                # Sort in increasing height order.
                contender_pile.sort(key=lambda s: state[s]["pose"][2])
        return piles, held_block

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
        for block in self._blocks:
            block_position = state[block]["pose"][:3]
            if np.sum(np.subtract(gripper_position,
                                  block_position)**2) < constants.GRASPING_TOL:
                return block
        return None


class SAHashable:
    """Defines a hashable object so that we can cache the transition model.
    Hashes a state and action.
    """
    def __init__(self, state, action, world, blocks):
        self.state = state
        self.action = action
        self.world = world
        self.blocks = blocks

    def to_tuple(self):
        """Convert to tuple for hashing and checking equality.
        """
        lst = []
        # Block poses.
        for block in self.blocks:
            lst.extend(self.state[block]["pose"])
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
