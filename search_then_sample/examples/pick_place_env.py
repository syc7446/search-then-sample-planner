import numpy as np
import pybullet as p
import random
import copy

import search_then_sample.utils.structs as structs
from search_then_sample.utils.env_base import Environment, EnvironmentFailure
from search_then_sample.utils.utils import WORLD
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import get_ik_ir_gen, base_motion, SavePath, \
    SINGLE_ROOM
from pybullet_planning.pybullet_tools.utils import get_pose, get_joint_positions, joints_from_names, is_placement, \
    set_base_values, load_pybullet, create_box, set_point, sample_placement, set_pose, joint_from_name, \
    set_joint_positions, wait_for_duration, TABLE_URDF, WorldSaver, STOVE_URDF
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
    OnStove = structs.Predicate("OnStove", 1, [_obj_type])
    Holding = structs.Predicate("Holding", 1, [_obj_type])
    HoldingSide = structs.Predicate("HoldingSide", 1, [_obj_type])
    HandEmpty = structs.Predicate("HandEmpty", 0, [])
    HandFull = structs.Predicate("HandFull", 0, [])
    IsValidPick = structs.Predicate("IsValidPick", 7,
                                     [_xbase_type, _ybase_type, _zbase_type,
                                      _xgrip_type, _ygrip_type, _zgrip_type,
                                      _obj_type])
    IsValidPlace = structs.Predicate("IsValidPlace", 7,
                                    [_xbase_type, _ybase_type, _zbase_type,
                                     _xgrip_type, _ygrip_type, _zgrip_type,
                                     _obj_type])
    _all_predicates = {OnTable, OnStove, Holding, HoldingSide, HandEmpty,
                       HandFull, IsValidPick, IsValidPlace}
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
    action_predicates = {Pick, Place}

    def __init__(self, num_objs, seed):
        super().__init__(num_objs, seed)
        self._num_targets = num_objs  # each object has a target
        self._objs = []
        for i in range(self._num_objs):
            self._objs.append(self._obj_type("obj{}".format(i)))
        self._objs_to_obj_ids = {}

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
                if pred_name == "OnTable" and is_placement(self._objs_to_obj_ids[obj], self.table[0]):
                    lits.add(pred(obj))
                if pred_name == "OnStove" and is_placement(self._objs_to_obj_ids[obj], self.stove):
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
        return lits

    def simulate(self, state, action, save_data):
        next_state = {k: v.copy() for k, v in state.items()}

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

        save_data.add_rest(base_states=next_state[self._world]["base_position"],
                           arm_states=next_state[self._world]["joints"],
                           obj_states=[next_state[obj]["pose"] for obj in self._objs],
                           hand_hold=next_state[self._world]["cur_holding_tf"], feasibilities=True)
        return next_state, reward, done, save_data

    def initial_pybullet_setup(self, arm, grasp_type, save_data):
        self.arm = arm
        self.problem = self._create_problem(grasp_type, save_data)
        self.save_path = SavePath(self.robot, self.arm)
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

        save_data.init(sym_actions='init', base_states=world_state["base_position"],
                       arm_states=world_state["joints"], obj_states=[state[obj]['pose'] for obj in self._objs],
                       configs=None, hand_hold=world_state["cur_holding_tf"], feasibilities=None, steps=-1)
        return state, self.robot, save_data

    def get_save_path(self):
        return self.save_path

    def sample_IsValidPick(self, state, obj, rng=None, pre_saved_world=None):
        """Sample values for continuous arguments of IsValidPick.
        Return dict from predicate argument index to value.
        """
        print('===Sampling in IsValidPick===')
        movable_obstacles = copy.deepcopy(self.problem.movable)
        movable_obstacles.remove(self._objs_to_obj_ids[obj])
        collision_objs = self.problem.fixed + movable_obstacles
        self.ik_ir_fn = get_ik_ir_gen(self.problem, custom_limits=self.custom_limits, collision_objs=collision_objs)

        if not pre_saved_world: saved_world = WorldSaver()
        else:
            saved_world = pre_saved_world
            saved_world.restore()

        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        p = Pose(self._objs_to_obj_ids[obj])
        grasps = list(self.grasp_gen_fn(self._objs_to_obj_ids[obj]))

        saved_world.restore()
        (g,) = random.choice(grasps)
        self.attachment = g.get_attachment(self.robot, self.arm)
        output = next(self.ik_ir_fn(self.arm, self._objs_to_obj_ids[obj], p, g), None)
        if not output:
            print('Plan fails: pick in IsValidPick')
            saved_world.restore()
            return g.value
        result_saved_world = WorldSaver()
        base_path = base_motion(self.robot, base_start, output[0].values,
                                obstacles=self.problem.fixed, custom_limits=self.custom_limits)
        if not base_path:
            print('Plan fails: base motion in IsValidPick')
            saved_world.restore()
            return g.value

        arm_path = [output[1].commands[0].path[i].values for i in range(len(output[1].commands[0].path))]
        self.save_path.add(actions=['base', 'arm'], paths=[base_path, arm_path],
                           attachments=[None, None])

        result_saved_world.restore()
        basex, basey, basez = output[0].values
        gripx, gripy, gripz = output[3]
        return {'saved_world': saved_world, 'config': g.value,
                0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def sample_IsValidPlace(self, state, obj, rng=None, pre_saved_world=None):
        """Sample values for continuous arguments of IsValidPick.
        Return dict from predicate argument index to value.
        """
        print('===Sampling in IsValidPlace===')
        movable_obstacles = copy.deepcopy(self.problem.movable)
        movable_obstacles.remove(self._objs_to_obj_ids[obj])
        collision_objs = self.problem.fixed + movable_obstacles
        self.ik_ir_fn = get_ik_ir_gen(self.problem, custom_limits=self.custom_limits, collision_objs=collision_objs)

        if not pre_saved_world: saved_world = WorldSaver()
        else:
            saved_world = pre_saved_world
            saved_world.restore()

        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        placement_gen = self.placement_gen_fn(self._objs_to_obj_ids[obj], self.stove)
        grasps = list(self.grasp_gen_fn(self._objs_to_obj_ids[obj]))

        saved_world.restore()
        (p,) = next(placement_gen)
        (g,) = random.choice(grasps)
        self.attachment = g.get_attachment(self.robot, self.arm)
        output = next(self.ik_ir_fn(self.arm, self._objs_to_obj_ids[obj], p, g), None)
        if not output:
            print('Plan fails: place in IsValidPlace')
            saved_world.restore()
            return p.value
        result_saved_world = WorldSaver()
        base_path = base_motion(self.robot, base_start, output[0].values, obstacles=self.problem.fixed,
                                attachments=[self.attachment], custom_limits=self.custom_limits)
        if not base_path:
            print('Plan fails: base motion in IsValidPlace')
            saved_world.restore()
            return p.value

        arm_path = [output[1].commands[0].path[i].values for i in range(len(output[1].commands[0].path))]
        self.save_path.add(actions=['base', 'arm'], paths=[base_path, arm_path],
                           attachments=[self.attachment, self.attachment])
        self.attachment = None

        result_saved_world.restore()
        basex, basey, basez = output[0].values
        gripx, gripy, gripz = output[3]
        return {'saved_world': saved_world, 'config': p.value,
                0: basex, 1: basey, 2: basez, 3: gripx, 4: gripy, 5: gripz}

    def _create_problem(self, grasp_type, save_data):
        other_arm = get_other_arm(self.arm)
        initial_conf = get_carry_conf(self.arm, grasp_type)

        plane = create_floor()
        room = load_pybullet(SINGLE_ROOM)
        self.custom_limits = {0: (-3., 3.), 1: (-3., 3.)}

        self.table = []
        self.table.append(load_pybullet(TABLE_URDF))
        set_point(self.table[0], (0, -2, 0))
        self.table.append(load_pybullet(TABLE_URDF))
        set_point(self.table[1], (0, 2, 0))
        self.stove = load_pybullet(STOVE_URDF)
        set_point(self.stove, (0.2, 1.8, TABLE_MAX_Z + .15 / 2))
        boxes = []
        displacement = 0
        for i in range(self._num_objs):
            boxes.append(create_box(.07, .05, .15))
            set_point(boxes[i], (-.4+displacement, -1.8, TABLE_MAX_Z + .15 / 2))
            displacement += 0.2

        self.robot = create_pr2()
        set_base_values(self.robot, (0, 0, 0))
        set_arm_conf(self.robot, self.arm, initial_conf)
        open_arm(self.robot, self.arm)
        set_arm_conf(self.robot, other_arm, arm_conf(other_arm, REST_LEFT_ARM))
        close_arm(self.robot, other_arm)

        # bottom_aabb = get_aabb(bottom_body, link=bottom_link)

        return Problem(robot=self.robot, movable=boxes, arms=[self.arm], grasp_types=[grasp_type],
                       surfaces=[self.table[0], self.stove])

    @property
    def literal_goal(self):
        goal = {self.OnStove(self._objs[i]) for i in range(self._num_objs)}
        goal.add(self.HandEmpty())
        return goal

    @property
    def discrete_predicates(self):
        return self._all_predicates - self._continuous_predicates