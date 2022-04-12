import numpy as np
import pybullet as p
import random

import search_then_sample.utils.structs as structs
from search_then_sample.utils.env_base import Environment, EnvironmentFailure
from search_then_sample.utils.utils import WORLD
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import get_ik_ir_gen, base_motion, get_base_custom_limits, MergedPath, \
    apply_margin, SAHashable, ROOM_FLOOR, ROOMS
from pybullet_planning.pybullet_tools.utils import get_pose, get_joint_positions, joints_from_names, is_placement, \
    set_base_values, load_pybullet, create_box, set_point, sample_placement, set_pose, joint_from_name, \
    set_joint_positions, wait_for_duration, TABLE_URDF, WorldSaver
from pybullet_planning.pybullet_tools.pr2_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, PR2_GROUPS, \
    arm_conf, close_arm, REST_LEFT_ARM
from pybullet_planning.pybullet_tools.pr2_primitives import get_stable_gen, get_grasp_gen, Pose
from pybullet_planning.pybullet_tools.pr2_problems import create_pr2, create_floor, Problem, TABLE_MAX_Z


class PickPlaceEnvironment(Environment):
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
    OnTargetTable = structs.Predicate("OnTargetTable", 1, [_obj_type])
    Holding = structs.Predicate("Holding", 1, [_obj_type])
    HoldingSide = structs.Predicate("HoldingSide", 1, [_obj_type])
    HandEmpty = structs.Predicate("HandEmpty", 0, [])
    HandFull = structs.Predicate("HandFull", 0, [])
    InRoom0 = structs.Predicate("InRoom0", 0, [])
    InRoom1 = structs.Predicate("InRoom1", 0, [])
    IsValidPick = structs.Predicate("IsValidPick", 7,
                                     [_xbase_type, _ybase_type, _zbase_type,
                                      _xgrip_type, _ygrip_type, _zgrip_type,
                                      _obj_type])
    IsValidPlace = structs.Predicate("IsValidPlace", 7,
                                    [_xbase_type, _ybase_type, _zbase_type,
                                     _xgrip_type, _ygrip_type, _zgrip_type,
                                     _obj_type])
    IsValidMove = structs.Predicate("IsValidMove", 3,
                                    [_xbase_type, _ybase_type, _zbase_type])
    _all_predicates = {OnTable, OnTargetTable, Holding, HoldingSide, HandEmpty,
                       HandFull, InRoom0, InRoom1, IsValidPick, IsValidPlace, IsValidMove}
    _all_predicate_names_to_preds = {p.name: p for p in _all_predicates}
    _continuous_predicates = {pred for pred in _all_predicates
                              if any(t.is_continuous for t in pred.var_types)}

    # Actions
    Pick = structs.Predicate("Pick", 7,
                             [_obj_type,
                              _xbase_type, _ybase_type, _zbase_type,
                              _xgrip_type, _ygrip_type, _zgrip_type])
    Place = structs.Predicate("Place", 7,
                             [_obj_type,
                              _xbase_type, _ybase_type, _zbase_type,
                              _xgrip_type, _ygrip_type, _zgrip_type])
    MoveToRoom1 = structs.Predicate("MoveToRoom1", 4,
                                    [_obj_type, _xbase_type, _ybase_type, _zbase_type])
    action_predicates = {Pick, Place, MoveToRoom1}

    def __init__(self, num_objs, num_rooms, num_sample_trials, margin_to_walls, seed):
        super().__init__(num_objs, seed)
        self._num_targets = num_objs  # each object has a target
        self._num_rooms = num_rooms
        self._num_sample_trials = num_sample_trials
        self._margin_to_walls = margin_to_walls
        self._objs = []
        for i in range(self._num_objs):
            self._objs.append(self._obj_type("obj{}".format(i)))
        self._objs_to_obj_ids = {}
        self._transmodel_cache = {}

    def parse_state(self, state):
        # Build literal set.
        lits = set()
        predicate_names = self._all_predicate_names_to_preds.keys()
        obj_radius = constants.BINS_OBJ_RADIUS
        if state[self._world]["cur_holding_tf"] is not None:
            held_obj, top_or_side = state[self._world]["cur_holding_tf"]
        else:
            held_obj = None
        for pred_name in predicate_names:
            pred = self._all_predicate_names_to_preds[pred_name]
            for obj in self._objs:
                obj_pose = state[obj]["pose"]
                if obj == held_obj:
                    continue
                if pred_name == "OnTable" and is_placement(self._objs_to_obj_ids[obj], self.table):
                    lits.add(pred(obj))
                if pred_name == "OnTargetTable" and is_placement(self._objs_to_obj_ids[obj], self.target_table):
                    lits.add(pred(obj))
            if pred_name == "HoldingSide" and held_obj is not None and \
               top_or_side == "side":
                lits.add(pred(held_obj))
            if pred_name == "Holding" and held_obj is not None:
                lits.add(pred(held_obj))
            if pred_name == "HandEmpty" and held_obj is None:
                lits.add(pred())
            if pred_name == "HandFull" and held_obj is not None:
                lits.add(pred())
            if pred_name == "InRoom0" and is_placement(self.robot, self.room_floors[0]):
                lits.add(pred())
            if pred_name == "InRoom1" and is_placement(self.robot, self.room_floors[1]):
                lits.add(pred())
        return lits

    def simulate(self, state, action):
        sa_hashable = SAHashable(state, action, self._world, self._objs)
        if sa_hashable in self._transmodel_cache:
            return self._transmodel_cache[sa_hashable]
        next_state = {k: v.copy() for k, v in state.items()}

        # TODO: currently only a single object is considered
        if self.attachment:
            for i, obj in enumerate(self._objs):
                next_state[obj]["pose"] = get_pose(self.problem.movable[i])[0]+get_pose(self.problem.movable[i])[1]
                if obj == action.variables[0]:
                    next_state[self._world]["cur_holding_tf"] = (obj, self.problem.grasp_types[0])
        else: next_state[self._world]["cur_holding_tf"] = None
        next_state[self._world]["base_position"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        next_state[self._world]["joints"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS[self.arm+'_arm']))

        hl_next_state = self.parse_state(next_state)
        reward = int(self.literal_goal.issubset(hl_next_state))
        done = (reward == 1)
        self._transmodel_cache[sa_hashable] = (next_state, reward, done)
        return next_state, reward, done

    def initial_pybullet_setup(self, arm, grasp_type):
        self.arm = arm
        self.problem = self._create_problem(grasp_type)
        self.merged_path = MergedPath(self.robot, self.arm)
        self.attachment = None

        self.grasp_gen_fn = get_grasp_gen(self.problem, collisions=True)
        self.placement_gen_fn = get_stable_gen(self.problem)

        for i, obj in enumerate(self._objs):
            self._objs_to_obj_ids[obj] = self.problem.movable[i]

        state = {}
        for i, obj in enumerate(self._objs):
            state[obj] = {"pose": get_pose(self.problem.movable[i])[0]+get_pose(self.problem.movable[i])[1]}
        world_state = {}
        world_state["base_position"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        world_state["joints"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS[self.arm+'_arm']))
        world_state["cur_holding_tf"] = None
        state[self._world] = world_state
        return state, self.robot

    def get_merged_path(self):
        return self.merged_path

    def sample_IsValidPick(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidPick.
        Return dict from predicate argument index to value.
        """
        print('===Sampling in IsValidPick===')
        # TODO: currently hardcoded for Room0
        self.ik_ir_fn = get_ik_ir_gen(self.problem,
                                      custom_limits=get_base_custom_limits(self.robot, self.room_floors[0]))

        saved_world = WorldSaver()
        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        p = Pose(self._objs_to_obj_ids[obj])
        grasps = list(self.grasp_gen_fn(self._objs_to_obj_ids[obj]))

        for i in range(self._num_sample_trials):
            saved_world.restore()
            (g,) = random.choice(grasps)
            self.attachment = g.get_attachment(self.robot, self.arm)
            output = next(self.ik_ir_fn(self.arm, self._objs_to_obj_ids[obj], p, g), None)
            if not output:
                self._error_msg(i, 'IsValidPick')
                continue
            result_saved_world = WorldSaver()
            base_path = base_motion(self.robot, self.room_floors, base_start, output[0].values,
                                    obstacles=self.problem.fixed)
            if base_path: break
            self._error_msg(i, 'IsValidPick')
        arm_path = [output[1].commands[0].path[i].values for i in range(len(output[1].commands[0].path))]
        self.merged_path.add(actions=['base', 'arm'], paths=[base_path, arm_path],
                             attachments=[None, None])

        # Visualization for debugging
        # saved_world.restore()
        # for bq in base_path:
        #     set_joint_positions(self.robot, [joint_from_name(self.robot, name) for name in PR2_GROUPS['base']], bq)
        #     wait_for_duration(0.01)
        # for aq in arm_path:
        #     set_joint_positions(self.robot, [joint_from_name(self.robot, name) for name in PR2_GROUPS['left_arm']], aq)
        #     wait_for_duration(0.01)
        # base_path_2 = base_motion(self.robot, self.room_floors, output[0].values, (0, 2, 0), obstacles=self.problem.fixed)
        # for bq in base_path_2:
        #     set_joint_positions(self.robot, [joint_from_name(self.robot, name) for name in PR2_GROUPS['base']], bq)
        #     output[2][self._objs_to_obj_ids[obj]].assign()
        #     wait_for_duration(0.01)

        result_saved_world.restore()
        basex, basey, basez = output[0].values
        gripx, gripy, gripz = output[3]
        return {0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def sample_IsValidPlace(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidPick.
        Return dict from predicate argument index to value.
        """
        print('===Sampling in IsValidPlace===')
        # TODO: currently hardcoded for Room1
        self.ik_ir_fn = get_ik_ir_gen(self.problem,
                                      custom_limits=get_base_custom_limits(self.robot, self.room_floors[1]))

        saved_world = WorldSaver()
        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        placement_gen = self.placement_gen_fn(self._objs_to_obj_ids[obj], self.target_table)
        grasps = list(self.grasp_gen_fn(self._objs_to_obj_ids[obj]))

        for i in range(self._num_sample_trials):
            saved_world.restore()
            (p,) = next(placement_gen)
            (g,) = random.choice(grasps)
            output = next(self.ik_ir_fn(self.arm, self._objs_to_obj_ids[obj], p, g), None)
            if not output:
                self._error_msg(i, 'IsValidPlace')
                continue
            result_saved_world = WorldSaver()
            base_path = base_motion(self.robot, self.room_floors, base_start, output[0].values,
                                    obstacles=self.problem.fixed, attachments=[self.attachment])
            if base_path: break
            self._error_msg(i, 'IsValidPlace')
        arm_path = [output[1].commands[0].path[i].values for i in range(len(output[1].commands[0].path))]
        self.merged_path.add(actions=['base', 'arm'], paths=[base_path, arm_path],
                             attachments=[self.attachment, self.attachment])
        self.attachment = None

        result_saved_world.restore()
        basex, basey, basez = output[0].values
        gripx, gripy, gripz = output[3]
        return {0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def sample_IsValidMove(self, state, obj, rng=None):
        """Sample values for continuous arguments of IsValidMove.
        Return dict from predicate argument index to value.
        """
        print('===Sampling in IsValidMove===')
        target_room = self.room_floors[1] # TODO: currently hardcoded for Room1 but change it to any Rooms later
        custom_limits = get_base_custom_limits(self.robot, self.room_floors[1]) # TODO: currently hardcoded
        saved_world = WorldSaver()
        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        goal_gen = self.placement_gen_fn(self.robot, target_room)

        for i in range(self._num_sample_trials):
            (base_goal,) = next(goal_gen)
            base_goal.value = apply_margin(base_goal, custom_limits, self._margin_to_walls) # Find a safer base goal
            saved_world.restore()
            base_path = base_motion(self.robot, self.room_floors, base_start, base_goal.value[0],
                                    obstacles=self.problem.fixed, attachments=[self.attachment],
                                    target=target_room)
            if base_path: break
            self._error_msg(i, 'IsValidMove')
        self.merged_path.add(actions=['base'], paths=[base_path],
                             attachments=[self.attachment])

        basex, basey, basez = base_goal.value[0]
        return {0: basex, 1: basey, 2: basez}

    def _error_msg(self, i, msg):
        if i == self._num_sample_trials - 1:
            # TODO: appropriately handle this later
            raise SystemExit('ERROR at {}: Number of samples is not enough to find a path.'.format(msg))

    def _create_problem(self, grasp_type):
        other_arm = get_other_arm(self.arm)
        initial_conf = get_carry_conf(self.arm, grasp_type)

        plane = create_floor()
        self.room_floors = []
        for i in range(self._num_rooms):
            self.room_floors.append(load_pybullet(ROOM_FLOOR))
        set_point(self.room_floors[1], (0, 6, 0))
        set_point(self.room_floors[2], (0, 12, 0))
        rooms = load_pybullet(ROOMS)

        self.table = load_pybullet(TABLE_URDF)
        set_point(self.table, (0, -2, 0))
        self.target_table = load_pybullet(TABLE_URDF)
        set_point(self.target_table, (0, 8, 0))
        box = create_box(.07, .05, .15)
        set_point(box, (0, -2, TABLE_MAX_Z + .15 / 2))

        self.robot = create_pr2()
        set_base_values(self.robot, (0, 0, 0))
        set_arm_conf(self.robot, self.arm, initial_conf)
        open_arm(self.robot, self.arm)
        set_arm_conf(self.robot, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(self.robot, other_arm)
        return Problem(robot=self.robot, movable=[box], arms=[self.arm], grasp_types=[grasp_type],
                       surfaces=[self.table, self.target_table],
                       goal_conf=get_pose(self.robot), goal_holding=[(self.arm, box)])

    @property
    def literal_goal(self):
        return {self.OnTargetTable(self._objs[-1]), self.HandEmpty(), self.InRoom1()}

    @property
    def discrete_predicates(self):
        return self._all_predicates - self._continuous_predicates