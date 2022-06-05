import numpy as np
import pybullet as p
import random
import copy

import search_then_sample.utils.structs as structs
from search_then_sample.utils.env_base import Environment, EnvironmentFailure
from search_then_sample.utils.utils import WORLD
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import get_ik_ir_gen, base_motion, SavePath, \
    SINGLE_ROOM, SINGLE_BIG_ROOM, SINGLE_SMALL_ROOM, create_shelf, create_shelf_placement, get_namo_rp_gen, \
    get_goal_position, is_box_on_placement, is_numerical_equal_two_tuples
from search_then_sample.train.constants import PRE_BOX_HOLDING_LEFT_ARM, BOX_HOLDING_LEFT_ARM, POST_BOX_HOLDING_LEFT_ARM, \
    ROOM_WIDTH, ROOM_HEIGHT, BOX_PLACEMENT_SIZE, BOX_REACHABLE_MARGIN, BOX_SIZE, BOX_HOLDING_PARENT_LINK_POSE

from pybullet_utils.transformations import euler_from_quaternion
from pybullet_planning.pybullet_tools.utils import get_pose, get_joint_positions, joints_from_names, is_placement, \
    set_base_values, load_pybullet, create_box, set_point, sample_placement, set_pose, set_euler, joint_from_name, \
    set_joint_positions, wait_for_duration, multiply, invert, get_euler, TABLE_URDF, WorldSaver, Attachment, pairwise_collision, \
    STOVE_URDF, RED, BLUE, BROWN, GREY, WHITE
from pybullet_planning.pybullet_tools.pr2_utils import get_other_arm, get_carry_conf, set_arm_conf, open_arm, PR2_GROUPS, \
    arm_conf, close_arm, rightarm_from_leftarm, get_gripper_link, REST_LEFT_ARM
from pybullet_planning.pybullet_tools.pr2_primitives import get_stable_gen, get_grasp_gen, create_attachment, Pose
from pybullet_planning.pybullet_tools.pr2_problems import create_pr2, create_floor, Problem, TABLE_MAX_Z


'''
Simplified NAMO in the following ways:
1. Objects that must be cleared are predetermined.
2. Bi-manipulation planning is not implemented.
3. Grasp values and reachable values (should be replaced by object placement values) are fixed (no sampling occurs).
'''

class NAMOEnvironment(Environment):
    # Object types
    _obj_type = structs.Type("obj")
    _target_type = structs.Type("target")
    _xbase_type = structs.ContinuousType("xbase")
    _ybase_type = structs.ContinuousType("ybase")
    _tbase_type = structs.ContinuousType("tbase")
    _xbase_type.set_domain([-10, 10])
    _ybase_type.set_domain([-10, 10])
    _tbase_type.set_domain([-10, 10])

    # Constant world object
    _world = WORLD

    # Predicates
    Blocked0 = structs.Predicate("Blocked0", 1, [_obj_type])
    Cleared0 = structs.Predicate("Cleared0", 1, [_obj_type])
    Blocked1 = structs.Predicate("Blocked1", 1, [_obj_type])
    Cleared1 = structs.Predicate("Cleared1", 1, [_obj_type])
    Blocked2 = structs.Predicate("Blocked2", 1, [_obj_type])
    Cleared2 = structs.Predicate("Cleared2", 1, [_obj_type])
    Blocked3 = structs.Predicate("Blocked3", 1, [_obj_type])
    Cleared3 = structs.Predicate("Cleared3", 1, [_obj_type])
    Blocked4 = structs.Predicate("Blocked4", 1, [_obj_type])
    Cleared4 = structs.Predicate("Cleared4", 1, [_obj_type])
    Blocked5 = structs.Predicate("Blocked5", 1, [_obj_type])
    Cleared5 = structs.Predicate("Cleared5", 1, [_obj_type])
    Blocked6 = structs.Predicate("Blocked6", 1, [_obj_type])
    Cleared6 = structs.Predicate("Cleared6", 1, [_obj_type])
    Blocked7 = structs.Predicate("Blocked7", 1, [_obj_type])
    Cleared7 = structs.Predicate("Cleared7", 1, [_obj_type])
    Blocked8 = structs.Predicate("Blocked8", 1, [_obj_type])
    Cleared8 = structs.Predicate("Cleared8", 1, [_obj_type])
    Blocked9 = structs.Predicate("Blocked9", 1, [_obj_type])
    Cleared9 = structs.Predicate("Cleared9", 1, [_obj_type])
    GoalReach = structs.Predicate("GoalReach", 0, [])
    IsValidClear = structs.Predicate("IsValidClear", 4,
                                     [_xbase_type, _ybase_type, _tbase_type, _obj_type])
    IsValidPickPlace = structs.Predicate("IsValidPickPlace", 4,
                                         [_xbase_type, _ybase_type, _tbase_type, _target_type])
    _all_predicates = {Blocked0, Cleared0, Blocked1, Cleared1, Blocked2, Cleared2, Blocked3, Cleared3,
                       Blocked4, Cleared4, Blocked5, Cleared5, Blocked6, Cleared6, Blocked7, Cleared7,
                       Blocked8, Cleared8, Blocked9, Cleared9,
                       GoalReach, IsValidClear, IsValidPickPlace}
    _all_predicate_names_to_preds = {p.name: p for p in _all_predicates}
    _continuous_predicates = {pred for pred in _all_predicates
                              if any(t.is_continuous for t in pred.var_types)}

    # Actions
    Clear0 = structs.Predicate("Clear0", 4, [_obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear1 = structs.Predicate("Clear1", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear2 = structs.Predicate("Clear2", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear3 = structs.Predicate("Clear3", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear4 = structs.Predicate("Clear4", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear5 = structs.Predicate("Clear5", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear6 = structs.Predicate("Clear6", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear7 = structs.Predicate("Clear7", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear8 = structs.Predicate("Clear8", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    Clear9 = structs.Predicate("Clear9", 5, [_obj_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    PickPlace = structs.Predicate("PickPlace", 5,
                                  [_target_type, _obj_type, _xbase_type, _ybase_type, _tbase_type])
    action_predicates = {Clear0, Clear1, Clear2, Clear3, Clear4, Clear5, Clear6, Clear7, Clear8, Clear9, PickPlace}

    def __init__(self, num_objs, sim_id, seed):
        super().__init__(num_objs, seed)
        self.sim_id = sim_id

        self._num_targets = num_objs  # each object has a target
        self._objs = []
        for i in range(self._num_objs):
            self._objs.append(self._obj_type("obj{}".format(i)))
        self._objs_to_obj_ids = {}
        self._target = self._target_type("target")
        self._target_to_obj_id = {}

    def parse_state(self, state):
        # Build literal set.
        lits = set()
        predicate_names = self._all_predicate_names_to_preds.keys()

        for pred_name in predicate_names:
            pred = self._all_predicate_names_to_preds[pred_name]
            if pred_name == "Blocked0":
                if is_numerical_equal_two_tuples(state[self._objs[0]]["pose"], state[self._objs[0]]["init_pose"], 5):
                    lits.add(pred(self._objs[0]))
            if pred_name == "Cleared0":
                if not is_numerical_equal_two_tuples(state[self._objs[0]]["pose"], state[self._objs[0]]["init_pose"], 5):
                    lits.add(pred(self._objs[0]))
            if pred_name == "Blocked1":
                if is_numerical_equal_two_tuples(state[self._objs[1]]["pose"], state[self._objs[1]]["init_pose"], 5):
                    lits.add(pred(self._objs[1]))
            if pred_name == "Cleared1":
                if not is_numerical_equal_two_tuples(state[self._objs[1]]["pose"], state[self._objs[1]]["init_pose"], 5):
                    lits.add(pred(self._objs[1]))
            if pred_name == "Blocked2":
                if is_numerical_equal_two_tuples(state[self._objs[2]]["pose"], state[self._objs[2]]["init_pose"], 5):
                    lits.add(pred(self._objs[2]))
            if pred_name == "Cleared2":
                if not is_numerical_equal_two_tuples(state[self._objs[2]]["pose"], state[self._objs[2]]["init_pose"], 5):
                    lits.add(pred(self._objs[2]))
            if pred_name == "Blocked3":
                if is_numerical_equal_two_tuples(state[self._objs[3]]["pose"], state[self._objs[3]]["init_pose"], 5):
                    lits.add(pred(self._objs[3]))
            if pred_name == "Cleared3":
                if not is_numerical_equal_two_tuples(state[self._objs[3]]["pose"], state[self._objs[3]]["init_pose"], 5):
                    lits.add(pred(self._objs[3]))
            if pred_name == "Blocked4":
                if is_numerical_equal_two_tuples(state[self._objs[4]]["pose"], state[self._objs[4]]["init_pose"], 5):
                    lits.add(pred(self._objs[4]))
            if pred_name == "Cleared4":
                if not is_numerical_equal_two_tuples(state[self._objs[4]]["pose"], state[self._objs[4]]["init_pose"], 5):
                    lits.add(pred(self._objs[4]))
            if pred_name == "Blocked5":
                if is_numerical_equal_two_tuples(state[self._objs[5]]["pose"], state[self._objs[5]]["init_pose"], 5):
                    lits.add(pred(self._objs[5]))
            if pred_name == "Cleared5":
                if not is_numerical_equal_two_tuples(state[self._objs[5]]["pose"], state[self._objs[5]]["init_pose"], 5):
                    lits.add(pred(self._objs[5]))
            if pred_name == "Blocked6":
                if is_numerical_equal_two_tuples(state[self._objs[6]]["pose"], state[self._objs[6]]["init_pose"], 5):
                    lits.add(pred(self._objs[6]))
            if pred_name == "Cleared6":
                if not is_numerical_equal_two_tuples(state[self._objs[6]]["pose"], state[self._objs[6]]["init_pose"], 5):
                    lits.add(pred(self._objs[6]))
            if pred_name == "Blocked7":
                if is_numerical_equal_two_tuples(state[self._objs[7]]["pose"], state[self._objs[7]]["init_pose"], 5):
                    lits.add(pred(self._objs[7]))
            if pred_name == "Cleared7":
                if not is_numerical_equal_two_tuples(state[self._objs[7]]["pose"], state[self._objs[7]]["init_pose"], 5):
                    lits.add(pred(self._objs[7]))
            if pred_name == "Blocked8":
                if is_numerical_equal_two_tuples(state[self._objs[8]]["pose"], state[self._objs[8]]["init_pose"], 5):
                    lits.add(pred(self._objs[8]))
            if pred_name == "Cleared8":
                if not is_numerical_equal_two_tuples(state[self._objs[8]]["pose"], state[self._objs[8]]["init_pose"], 5):
                    lits.add(pred(self._objs[8]))
            if pred_name == "Blocked9":
                if is_numerical_equal_two_tuples(state[self._objs[9]]["pose"], state[self._objs[9]]["init_pose"], 5):
                    lits.add(pred(self._objs[9]))
            if pred_name == "Cleared9":
                if not is_numerical_equal_two_tuples(state[self._objs[9]]["pose"], state[self._objs[9]]["init_pose"], 5):
                    lits.add(pred(self._objs[9]))
            if pred_name == "GoalReach" and is_box_on_placement(self._target_to_obj_id, self.goal_placement):
                lits.add(pred())
        return lits

    def simulate(self, state, action, save_data):
        next_state = {k: v.copy() for k, v in state.items()}

        for i, obj in enumerate(self._objs):
            next_state[obj]["pose"] = get_pose(self.problem.movable[i])[0] + get_pose(self.problem.movable[i])[1]
        next_state[self._target]["pose"] = get_pose(self.problem.movable[-1])[0] + get_pose(self.problem.movable[-1])[1]
        next_state[self._world]["base_position"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        next_state[self._world]["joints"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['left_arm'])) + \
                                            get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['right_arm']))

        hl_next_state = self.parse_state(next_state)
        reward = int(self.literal_goal.issubset(hl_next_state))
        done = (reward == 1)

        return next_state, reward, done, save_data

    def initial_pybullet_setup(self, arm, grasp_type, save_data):
        self.problem = self._create_problem(grasp_type, save_data)
        self.save_path = SavePath(self.robot, None)

        for i, obj in enumerate(self._objs):
            self._objs_to_obj_ids[obj] = self.problem.movable[i]
        self._target_to_obj_id = self.problem.movable[-1]

        state = {}
        for i, obj in enumerate(self._objs):
            state[obj] = {"pose": get_pose(self.problem.movable[i])[0] + get_pose(self.problem.movable[i])[1],
                          "init_pose": get_pose(self.problem.movable[i])[0] + get_pose(self.problem.movable[i])[1]}
        state[self._target] = {"pose": get_pose(self.problem.movable[-1])[0] + get_pose(self.problem.movable[-1])[1]}
        world_state = {}
        world_state["base_position"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        world_state["joints"] = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['left_arm'])) + \
                                get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['right_arm']))
        state[self._world] = world_state

        save_data.init(sym_actions='init', base_states=world_state["base_position"],
                       arm_states=world_state["joints"], obj_states=[state[obj]['pose'] for obj in self._objs],
                       configs=None, hand_hold=None, feasibilities=None, steps=-1)
        return state, self.robot, save_data

    def get_save_path(self):
        return self.save_path

    def sample_IsValidClear(self, state, obj, rng=None, pre_saved_world=None):
        print('===Sampling in IsValidClear===')
        if not pre_saved_world:
            saved_world = WorldSaver()
        else:
            saved_world = pre_saved_world
            saved_world.restore()

        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        base_goal = get_goal_position(get_pose(self._objs_to_obj_ids[obj])[0][:2],
                                      get_euler(self._objs_to_obj_ids[obj])[-1],
                                      self.reachable_point)
        base_reach_path = base_motion(self.robot, base_start, base_goal,
                                      teleport=True,
                                      obstacles=self.fixed_obstacles + [self.target_box],
                                      custom_limits=self.custom_limits)
        if not base_reach_path:
            print('Plan fails: base reach motion in IsValidClear')
            saved_world.restore()
            return get_pose(self._objs_to_obj_ids[obj])
        set_joint_positions(self.robot, [0, 1, 2], base_reach_path[-1])

        set_arm_conf(self.robot, 'left', BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(BOX_HOLDING_LEFT_ARM))
        attachment = create_attachment(self.robot, get_gripper_link(self.robot, 'left'), self._objs_to_obj_ids[obj])
        set_arm_conf(self.robot, 'left', POST_BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(POST_BOX_HOLDING_LEFT_ARM))
        attachment.assign()

        # p = random.uniform(-3.14, 3.14)
        p = 0
        p_pose = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))[:2] + (p,)
        set_joint_positions(self.robot, [0, 1, 2], p_pose)

        set_arm_conf(self.robot, 'left', BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(BOX_HOLDING_LEFT_ARM))
        attachment.assign()

        # if any(pairwise_collision(self._objs_to_obj_ids[obj], o) for o in
        #        self.fixed_obstacles + [self.target_box] + [b for b in self.boxes if b != self._objs_to_obj_ids[obj]]):
        #     print('Plan fails: wrong placement of an object in IsValidClear')
        #     saved_world.restore()
        #     return p_pose
        set_arm_conf(self.robot, 'left', PRE_BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(PRE_BOX_HOLDING_LEFT_ARM))

        self.save_path.add(actions=['base'], paths=base_reach_path+[p_pose], attachments=[attachment])

        result_saved_world = WorldSaver()
        basex, basey, baset = p_pose
        return {'saved_world': result_saved_world, 'config': p_pose, 0: basex, 1: basey, 2: baset}

    def sample_IsValidPickPlace(self, state, obj, rng=None, pre_saved_world=None):
        print('===Sampling in IsValidPickPlace===')
        if not pre_saved_world:
            saved_world = WorldSaver()
        else:
            saved_world = pre_saved_world
            saved_world.restore()

        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        base_goal = get_goal_position(get_pose(self._target_to_obj_id)[0][:2],
                                      get_euler(self._target_to_obj_id)[-1],
                                      self.reachable_point)
        base_reach_path = base_motion(self.robot, base_start, base_goal,
                                      teleport=True,
                                      obstacles=self.fixed_obstacles, custom_limits=self.custom_limits)
        if not base_reach_path:
            print('Plan fails: base reach motion in IsValidPickPlace')
            saved_world.restore()
            return get_pose(self._target_to_obj_id)
        set_joint_positions(self.robot, [0, 1, 2], base_reach_path[-1])

        set_arm_conf(self.robot, 'left', BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(BOX_HOLDING_LEFT_ARM))
        attachment = create_attachment(self.robot, get_gripper_link(self.robot, 'left'), self._target_to_obj_id)
        set_arm_conf(self.robot, 'left', POST_BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(POST_BOX_HOLDING_LEFT_ARM))
        attachment.assign()

        rp_gen_fn = get_namo_rp_gen(self.fixed_obstacles + [self.robot] + self.boxes)
        rp_gen = rp_gen_fn(self._target_to_obj_id, self.goal_placement)
        (p,) = next(rp_gen)

        base_start = get_joint_positions(self.robot, joints_from_names(self.robot, PR2_GROUPS['base']))
        base_goal = get_goal_position(p.value[0][:2],
                                      euler_from_quaternion(p.value[1])[-1],
                                      self.reachable_point)
        base_clear_path = base_motion(self.robot, base_start, base_goal,
                                      attachments=[attachment], obstacles=self.fixed_obstacles + self.boxes,
                                      custom_limits=self.custom_limits)
        if not base_clear_path:
            print('Plan fails: base clear motion in IsValidClear')
            saved_world.restore()
            return p.value
        set_joint_positions(self.robot, [0, 1, 2], base_clear_path[-1])

        set_arm_conf(self.robot, 'left', BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(BOX_HOLDING_LEFT_ARM))
        attachment.assign()
        set_arm_conf(self.robot, 'left', PRE_BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(PRE_BOX_HOLDING_LEFT_ARM))

        self.save_path.add(actions=['base'], paths=base_reach_path+base_clear_path, attachments=[attachment])

        result_saved_world = WorldSaver()
        basex, basey, baset = base_clear_path[-1]
        return {'saved_world': result_saved_world, 'config': p.value, 0: basex, 1: basey, 2: baset}

    def _create_problem(self, grasp_type, save_data):
        plane = create_floor()
        room = load_pybullet(SINGLE_BIG_ROOM)
        set_point(room, (ROOM_WIDTH * 3 / 8, ROOM_HEIGHT * 3 / 8, 0))
        self.custom_limits = {0: (-1 * ROOM_WIDTH / 2 + ROOM_WIDTH * 3 / 8, ROOM_WIDTH / 2 + ROOM_WIDTH * 3 / 8),
                         1: (-1 * ROOM_HEIGHT / 2 + ROOM_HEIGHT * 3 / 8, ROOM_HEIGHT / 2 + ROOM_HEIGHT * 3 / 8)}

        self.robot = create_pr2()
        set_base_values(self.robot, (0.0, 0.0, 0.0))
        set_arm_conf(self.robot, 'left', PRE_BOX_HOLDING_LEFT_ARM)
        set_arm_conf(self.robot, 'right', rightarm_from_leftarm(PRE_BOX_HOLDING_LEFT_ARM))
        self.reachable_point = (-BOX_REACHABLE_MARGIN, 0.0, 0.0)

        # Goal placement
        self.goal_placement = create_shelf_placement(w=1.0, l=1.0, h=0.01, color=GREY)
        set_point(self.goal_placement, (0.0, 0.0, 0.0))

        # Box
        # TODO: currently the orientations of the boxes are carefully tuned.
        self.boxes = []
        for i in range(self._num_objs):
            self.boxes.append(create_box(BOX_SIZE, BOX_SIZE, BOX_SIZE, color=BLUE))
            if i == 0:
                set_point(self.boxes[i], (0.6, 1.4, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.0))
            elif i == 1:
                set_point(self.boxes[i], (0.8, 2.1, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.0))
            elif i == 2:
                set_point(self.boxes[i], (1.7, 2.4, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.5))
            elif i == 3:
                set_point(self.boxes[i], (1.3, 3.0, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.4))
            elif i == 4:
                set_point(self.boxes[i], (1.4, 3.8, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.57))
            elif i == 5:
                set_point(self.boxes[i], (2.0, 4.5, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.3))
            elif i == 6:
                set_point(self.boxes[i], (1.7, 5.2, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 2.07))
            elif i == 7:
                set_point(self.boxes[i], (2.4, 6.0, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.87))
            elif i == 8:
                set_point(self.boxes[i], (3.0, 6.9, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 1.6))
            elif i == 9:
                set_point(self.boxes[i], (3.8, 7.4, BOX_SIZE / 2))
                set_euler(self.boxes[i], (0.0, 0.0, 0.9))

        # Target box
        self.target_box = create_box(BOX_SIZE, BOX_SIZE, BOX_SIZE, color=RED)
        set_point(self.target_box, (ROOM_WIDTH / 2 + ROOM_WIDTH / 4, ROOM_HEIGHT / 2 + ROOM_HEIGHT / 4, BOX_SIZE / 2))

        # Fixed obstacles
        num_fixed_boxes = 25
        fixed_boxes = []
        for i in range(num_fixed_boxes):
            fixed_boxes.append(create_box(BOX_SIZE, BOX_SIZE, BOX_SIZE, color=WHITE))
            if i == 0:
                set_point(fixed_boxes[i], (0.0, 4.6, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.0))
            elif i == 1:
                set_point(fixed_boxes[i], (0.9, 5.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.5))
            elif i == 2:
                set_point(fixed_boxes[i], (4.0, 5.2, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.2))
            elif i == 3:
                set_point(fixed_boxes[i], (4.1, 6.6, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.4))
            elif i == 4:
                set_point(fixed_boxes[i], (3.9, 5.9, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.1))
            elif i == 5:
                set_point(fixed_boxes[i], (0.0, 7.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.3))
            elif i == 6:
                set_point(fixed_boxes[i], (0.9, 8.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, -0.2))
            elif i == 7:
                set_point(fixed_boxes[i], (1.8, 8.2, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.2))
            elif i == 8:
                set_point(fixed_boxes[i], (2.2, 7.5, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.4))
            elif i == 9:
                set_point(fixed_boxes[i], (1.4, 6.6, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.1))
            elif i == 10:
                set_point(fixed_boxes[i], (0.2, 3.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.3))
            elif i == 11:
                set_point(fixed_boxes[i], (-0.3, 3.2, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, -0.2))
            elif i == 12:
                set_point(fixed_boxes[i], (3.4, 2.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.2))
            elif i == 13:
                set_point(fixed_boxes[i], (2.9, 2.1, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.4))
            elif i == 14:
                set_point(fixed_boxes[i], (4.0, 4.2, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.1))
            elif i == 15:
                set_point(fixed_boxes[i], (3.1, 1.2, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.0))
            elif i == 16:
                set_point(fixed_boxes[i], (4.0, 1.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.8))
            elif i == 17:
                set_point(fixed_boxes[i], (2.6, 0.7, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, -0.4))
            elif i == 18:
                set_point(fixed_boxes[i], (0.1, 6.1, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.0))
            elif i == 19:
                set_point(fixed_boxes[i], (2.6, 8.6, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.8))
            elif i == 20:
                set_point(fixed_boxes[i], (4.1, 0.8, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, -0.4))
            elif i == 21:
                set_point(fixed_boxes[i], (4.4, 0.1, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 1.0))
            elif i == 22:
                set_point(fixed_boxes[i], (0.1, 7.1, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, 0.8))
            elif i == 23:
                set_point(fixed_boxes[i], (2.5, 9.6, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, -0.4))
            elif i == 24:
                set_point(fixed_boxes[i], (1.3, 0.5, BOX_SIZE / 2))
                set_euler(fixed_boxes[i], (0.0, 0.0, -0.4))
        self.fixed_obstacles = [room] + fixed_boxes

        return Problem(robot=self.robot, movable=self.boxes+[self.target_box])

    @property
    def literal_goal(self):
        return {self.GoalReach()}

    @property
    def discrete_predicates(self):
        return self._all_predicates - self._continuous_predicates