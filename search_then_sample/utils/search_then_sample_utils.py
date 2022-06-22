"""Generic utilities for PyBullet.
"""

import os
import math
import time
import pickle
import numpy as np
import pybullet as p
import random
from itertools import islice, count
from datetime import datetime, date, time

from pybullet_utils.transformations import quaternion_from_euler, euler_from_quaternion, quaternion_multiply
from pybullet_planning.pybullet_tools.ikfast.pr2.ik import is_ik_compiled, pr2_inverse_kinematics
from pybullet_planning.pybullet_tools.pr2_primitives import create_trajectory, iterate_approach_path, Commands, State, \
    SELF_COLLISIONS, Pose, Conf
from pybullet_planning.pybullet_tools.pr2_utils import get_gripper_link, get_arm_joints, arm_conf, open_arm, get_aabb, \
    get_disabled_collisions, get_group_joints, learned_pose_generator, PR2_GROUPS
from pybullet_planning.pybullet_tools.utils import is_placement, multiply, invert, set_joint_positions, \
    pairwise_collision, \
    get_joint_positions, plan_direct_joint_motion, plan_joint_motion, joint_from_name, all_between, BodySaver, \
    LockRenderer, get_bodies, get_joint_limits, set_joint_limits, get_default_resolution, uniform_pose_generator, Saver, \
    PoseSaver, ConfSaver, get_configuration, remove_body, inverse_kinematics_helper, get_movable_joints, get_link_pose, \
    is_pose_close, elapsed_time, irange, create_sub_robot, get_custom_limits, sub_inverse_kinematics, INF, \
    get_box_geometry, \
    create_shape, create_body, sample_placement, get_pose, get_euler, STATIC_MASS, RED, BROWN, join_paths, \
    get_parent_dir, \
    get_extend_fn, get_collision_fn, MAX_DISTANCE
from pybullet_planning.motion.motion_planners.utils import default_selector

MODEL_DIRECTORY = join_paths(get_parent_dir(__file__), os.pardir, '../pybullet_planning/models/')
ROOM_FLOOR = join_paths(MODEL_DIRECTORY, 'room_floor.urdf')
SHORT_FLOOR = join_paths(MODEL_DIRECTORY, 'short_floor.urdf')
ROOMS = join_paths(MODEL_DIRECTORY, 'rooms.urdf')
SINGLE_ROOM = join_paths(MODEL_DIRECTORY, 'single_room.urdf')
SINGLE_BIG_ROOM = join_paths(MODEL_DIRECTORY, 'single_big_room.urdf')
SINGLE_SMALL_ROOM = join_paths(MODEL_DIRECTORY, 'single_small_room.urdf')
NARROW_TABLE = join_paths(MODEL_DIRECTORY, 'narrow_table.urdf')


def base_motion(robot, base_start, base_goal, teleport=False, obstacles=[], attachments=[], custom_limits={}):
    disabled_collisions = get_disabled_collisions(robot)
    base_joints = [joint_from_name(robot, name) for name in PR2_GROUPS['base']]
    set_joint_positions(robot, base_joints, base_start)
    base_goal = base_goal[:len(base_joints)]
    # TODO: hardcoded values to increase resolutions used in extension fn

    if teleport:
        set_joint_positions(robot, base_joints, base_start)
        if any(pairwise_collision(robot, b) for b in obstacles):
            return None
        set_joint_positions(robot, base_joints, base_goal)
        if any(pairwise_collision(robot, b) for b in obstacles):
            return None
        return [base_start, base_goal]
    else:
        resolutions = np.array([2 * get_default_resolution(robot, 2), 2 * get_default_resolution(robot, 2),
                                get_default_resolution(robot, 2)])
        with LockRenderer(lock=False):
            base_path = plan_joint_motion(robot, base_joints, base_goal, obstacles=obstacles,
                                          attachments=attachments, disabled_collisions=disabled_collisions,
                                          resolutions=resolutions, custom_limits=custom_limits)
        if not base_path: set_joint_positions(robot, base_joints, base_start)
        return base_path


def arm_motion(robot, arm_start, arm_goal, arm, teleport=False, obstacles=[], attachments=[], custom_limits={}):
    disabled_collisions = get_disabled_collisions(robot)
    arm_joints = get_arm_joints(robot, arm)
    set_joint_positions(robot, arm_joints, arm_start)
    base_goal = arm_goal[:len(arm_joints)]
    # TODO: hardcoded values to increase resolutions used in extension fn

    if teleport:
        set_joint_positions(robot, arm_joints, arm_start)
        if any(pairwise_collision(robot, b) for b in obstacles):
            return None
        set_joint_positions(robot, arm_joints, base_goal)
        if any(pairwise_collision(robot, b) for b in obstacles):
            return None
        return [arm_start, arm_goal]
    else:
        resolutions = 0.05 ** np.ones(len(arm_joints))
        with LockRenderer(lock=False):
            arm_path = plan_joint_motion(robot, arm_joints, arm_goal, obstacles=obstacles,
                                         attachments=attachments, disabled_collisions=disabled_collisions,
                                         self_collisions=False,
                                         resolutions=resolutions, custom_limits=custom_limits,
                                         restarts=2, max_iterations=25, smooth=25)
        if not arm_path: set_joint_positions(robot, arm_joints, arm_start)
        return arm_path


def simple_direct_path_base_motion(robot, base_start, base_goal, teleport=False, obstacles=[], attachments=[],
                            custom_limits={}):
    disabled_collisions = get_disabled_collisions(robot)
    base_start = (base_start[0], base_start[1], -2.0)  # TODO: hard-coded
    base_joints = [joint_from_name(robot, name) for name in PR2_GROUPS['base']]
    set_joint_positions(robot, base_joints, base_start)
    # base_goal = base_goal[:len(base_joints)]
    base_goal = (0.0, 0.0, 0.0) # TODO: hard-coded. instead use the above
    resolutions = np.array([2 * get_default_resolution(robot, 2), 2 * get_default_resolution(robot, 2),
                            get_default_resolution(robot, 2)])
    self_collisions = True
    max_distance = MAX_DISTANCE
    use_aabb = False
    cache = True

    # if math.floor(abs((base_goal[0]-base_start[0]) / resolutions[0])) > \
    #     math.floor(abs((base_goal[1]-base_start[1]) / resolutions[1])):
    #     num_steps = math.floor(abs((base_goal[0]-base_start[0]) / resolutions[0]))
    #     resolutions[1] = abs((base_goal[1]-base_start[1])) / num_steps
    # else:
    #     num_steps = math.floor(abs((base_goal[1] - base_start[1]) / resolutions[1]))
    #     resolutions[0] = abs((base_goal[0] - base_start[0])) / num_steps
    num_steps = 10
    resolutions[0] = abs((base_goal[0] - base_start[0])) / num_steps
    resolutions[1] = abs((base_goal[1] - base_start[1])) / num_steps

    collision_fn = get_collision_fn(robot, base_joints, obstacles, attachments, self_collisions, disabled_collisions,
                                    custom_limits=custom_limits, max_distance=max_distance,
                                    use_aabb=use_aabb, cache=cache)
    base_path = []
    for t in range(num_steps):
        base_path.append((base_start[0] - t * resolutions[0],
                          base_start[1] - t * resolutions[1],
                          base_start[2]))
    if any(collision_fn(q) for q in default_selector(base_path)):
        base_path = None
    if not base_path: set_joint_positions(robot, base_joints, base_start)
    return base_path


def get_stable_manual_gen(problem, collisions=True, **kwargs):
    obstacles = problem.fixed if collisions else []
    manual_body_pose = []
    manual_body_pose.append(((0.11150483042001724, 1.8443114757537842, 0.7074999809265137), (0.0, 0.0, 0.15275177359580994, 0.9882646203041077)))
    manual_body_pose.append(((-0.024299629032611847, 1.8851918649673462, 0.7074999809265137), (0.0, 0.0, 0.5342151522636414, 0.8453485369682312)))
    manual_body_pose.append(((-0.1571858525276184, 1.8978041076660156, 0.7074999809265137), (0.0, 0.0, -0.6982966065406799, 0.7158085107803345)))
    manual_body_pose.append(((-0.1974816119670868, 1.7997376012802124, 0.7074999809265137), (0.0, 0.0, 0.8826384544372559, 0.4700525403022766)))
    manual_body_pose.append(((0.140473221316933632, 1.7847394847869873, 0.7074999809265137), (0.0, 0.0, -0.6524704694747925, 0.7578141689300537)))
    manual_body_pose.append(((-0.07064237773418427, 1.7635848426818848, 0.7074999809265137), (0.0, 0.0, -0.6774644255638123, 0.7355555295944214)))
    manual_body_pose.append(((-0.04096284881234169, 1.669626235961914, 0.7074999809265137), (0.0, 0.0, 0.9875448942184448, -0.15733756124973297)))
    manual_body_pose.append(((0.1010371595621109, 1.6853487396240234, 0.7074999809265137), (0.0, 0.0, 0.2628512978553772, 0.9648363590240479)))
    manual_body_pose.append(((0.135058495923876762, 1.618767490386963, 0.7074999809265137), (0.0, 0.0, 0.5342151522636414, 0.8453485369682312)))
    manual_body_pose.append(((-0.14256032705307007, 1.6276129484176636, 0.7074999809265137), (0.0, 0.0, -0.6982966065406799, 0.7158085107803345)))
    def gen(body, surface, index):
        # TODO: surface poses are being sampled in pr2_belief
        if surface is None:
            surfaces = problem.surfaces
        else:
            surfaces = [surface]
        while True:
            surface = random.choice(surfaces) # TODO: weight by area
            body_pose = manual_body_pose[index]
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)
    # TODO: apply the acceleration technique here
    return gen


def get_ir_sampler(problem, custom_limits={}, max_attempts=25, collisions=True, collision_objs=[], learned=True):
    robot = problem.robot
    obstacles = collision_objs if collisions else []
    gripper = problem.get_gripper()

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                # print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None

    return gen_fn


def get_ik_fn(problem, custom_limits={}, collisions=True, collision_objs=[], teleport=False):
    robot = problem.robot
    obstacles = collision_objs if collisions else []
    if is_ik_compiled():
        print('Using ikfast for inverse kinematics')
    else:
        print('Using pybullet for inverse kinematics')

    def fn(arm, obj, pose, grasp, base_conf):
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        # approach_pose = multiply(grasp.approach, gripper_pose)
        approach_pose = multiply(pose.value, invert(grasp.approach))
        arm_link = get_gripper_link(robot, arm)
        arm_joints = get_arm_joints(robot, arm)

        default_conf = arm_conf(arm, grasp.carry)
        # sample_fn = get_sample_fn(robot, arm_joints)
        pose.assign()
        base_conf.assign()
        open_arm(robot, arm)
        set_joint_positions(robot, arm_joints, default_conf)  # default_conf | sample_fn()
        grasp_conf = pr2_inverse_kinematics(robot, arm, gripper_pose,
                                            custom_limits=custom_limits)  # , upper_limits=USE_CURRENT)
        # nearby_conf=USE_CURRENT) # upper_limits=USE_CURRENT,
        if (grasp_conf is None) or any(pairwise_collision(robot, b) for b in obstacles):  # [obj]
            # print('Grasp IK failure', grasp_conf)
            # if grasp_conf is not None:
            #    print(grasp_conf)
            #    #wait_if_gui()
            return None
        # approach_conf = pr2_inverse_kinematics(robot, arm, approach_pose, custom_limits=custom_limits,
        #                                       upper_limits=USE_CURRENT, nearby_conf=USE_CURRENT)
        approach_conf = sub_inverse_kinematics(robot, arm_joints[0], arm_link, approach_pose,
                                               custom_limits=custom_limits)
        if (approach_conf is None) or any(pairwise_collision(robot, b) for b in obstacles + [obj]):
            # print('Approach IK failure', approach_conf)
            # wait_if_gui()
            return None
        approach_conf = get_joint_positions(robot, arm_joints)
        attachment = grasp.get_attachment(problem.robot, arm)
        attachments = {attachment.child: attachment}
        if teleport:
            path = [default_conf, approach_conf, grasp_conf]
        else:
            resolutions = 0.05 ** np.ones(len(arm_joints))
            set_joint_positions(robot, arm_joints, default_conf)
            approach_path = plan_joint_motion(robot, arm_joints, approach_conf, attachments=attachments.values(),
                                              obstacles=obstacles, self_collisions=SELF_COLLISIONS,
                                              custom_limits=custom_limits, resolutions=resolutions,
                                              restarts=2, max_iterations=25, smooth=25)
            if approach_path is None:
                print('Approach path failure')
                return None
            set_joint_positions(robot, arm_joints, approach_conf)
            grasp_path = plan_direct_joint_motion(robot, arm_joints, grasp_conf, attachments=attachments.values(),
                                                  obstacles=approach_obstacles, self_collisions=SELF_COLLISIONS,
                                                  custom_limits=custom_limits, resolutions=resolutions / 2.)
            if grasp_path is None:
                print('Grasp path failure')
                return None
            path = approach_path + grasp_path
        mt = create_trajectory(robot, arm_joints, path)
        cmd = Commands(State(attachments=attachments), savers=[BodySaver(robot)], commands=[mt])
        return (cmd, attachments, gripper_pose[0],)  # Only this line has been changed from the original code

    return fn


def get_ir_sampler_legacy(problem, custom_limits={}, max_attempts=25, collisions=True, learned=True):
    robot = problem.robot
    obstacles = problem.fixed if collisions else []
    gripper = problem.get_gripper()
    custom_limits = custom_limits

    def gen_fn(arm, obj, pose, grasp):
        pose.assign()
        approach_obstacles = {obst for obst in obstacles if not is_placement(obj, obst)}
        for _ in iterate_approach_path(robot, arm, gripper, pose, grasp, body=obj):
            if any(pairwise_collision(gripper, b) or pairwise_collision(obj, b) for b in approach_obstacles):
                return
        gripper_pose = multiply(pose.value, invert(grasp.value))  # w_f_g = w_f_o * (g_f_o)^-1
        default_conf = arm_conf(arm, grasp.carry)
        arm_joints = get_arm_joints(robot, arm)
        base_joints = get_group_joints(robot, 'base')
        if learned:
            base_generator = learned_pose_generator(robot, gripper_pose, arm=arm, grasp_type=grasp.grasp_type)
        else:
            base_generator = uniform_pose_generator(robot, gripper_pose)
        lower_limits, upper_limits = get_custom_limits(robot, base_joints, custom_limits)
        lower_limits = (custom_limits[0][0], custom_limits[1][0]) + lower_limits[2:]
        upper_limits = (custom_limits[0][1], custom_limits[1][1]) + upper_limits[2:]

        # lower_limit, upper_limit = get_custom_limits(robot, [2])
        # lower_limits = (custom_limits[0][0], custom_limits[1][0]) + lower_limit
        # upper_limits = (custom_limits[0][1], custom_limits[1][1]) + upper_limit
        while True:
            count = 0
            for base_conf in islice(base_generator, max_attempts):
                count += 1
                if not all_between(lower_limits, base_conf, upper_limits):
                    continue
                bq = Conf(robot, base_joints, base_conf)
                pose.assign()
                bq.assign()
                set_joint_positions(robot, arm_joints, default_conf)
                if any(pairwise_collision(robot, b) for b in obstacles + [obj]):
                    continue
                # print('IR attempts:', count)
                yield (bq,)
                break
            else:
                yield None

    return gen_fn


def get_ik_ir_gen(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        b, a, p, g = inputs
        ir_generator = ir_sampler(*inputs)
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                ir_outputs = next(ir_generator)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            # if not p.init:
            #    return

    return gen


def get_ik_ir_given_q_gen(problem, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    ir_sampler = get_ir_sampler(problem, learned=learned, max_attempts=max_attempts, **kwargs)
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        b, a, p, g, q = inputs
        ir_generator = ir_sampler(*inputs[:4])
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            ir_outputs = (q,)
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs[:4] + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            # if not p.init:
            #    return

    return gen


def get_ik_skip_ir_gen(problem, shelf, reachable_point, max_attempts=25, learned=True, teleport=False, **kwargs):
    # TODO: compose using general fn
    robot = problem.robot
    ik_fn = get_ik_fn(problem, teleport=teleport, **kwargs)

    def gen(*inputs):
        b, a, p, g = inputs
        attempts = 0
        while True:
            if max_attempts <= attempts:
                if not p.init:
                    return
                attempts = 0
                yield None
            attempts += 1
            try:
                base_joints = get_group_joints(robot, 'base')
                base_conf = get_goal_position(get_pose(shelf)[0][:2],
                                              get_euler(shelf)[-1] + 1.57,
                                              reachable_point)  # TODO: hard-coded rotation value (1.57: 90 degree rotation)
                ir_outputs = (Conf(robot, base_joints, base_conf),)
            except StopIteration:
                return
            if ir_outputs is None:
                continue
            ik_outputs = ik_fn(*(inputs + ir_outputs))
            if ik_outputs is None:
                continue
            print('IK attempts:', attempts)
            yield ir_outputs + ik_outputs
            return
            # if not p.init:
            #    return

    return gen


def get_namo_rp_gen(fixed_obstacles, collisions=True, **kwargs):
    # This generator is for reachable pose of the robot to place an object (pr is reachable placement)
    obstacles = fixed_obstacles if collisions else []

    def gen(body, surface):
        # TODO: surface poses are being sampled in pr2_belief
        surfaces = [surface]
        while True:
            surface = random.choice(surfaces)  # TODO: weight by area
            body_pose = sample_placement(body, surface, **kwargs)
            if body_pose is None:
                break
            p = Pose(body, body_pose, surface)
            p.assign()
            if not any(pairwise_collision(body, obst) for obst in obstacles if obst not in {body, surface}):
                yield (p,)

    # TODO: apply the acceleration technique here
    return gen


def get_goal_position(translate, rotate, reachable_point):
    tform = np.array([[math.cos(rotate), -math.sin(rotate), translate[0]],
                      [math.sin(rotate), math.cos(rotate), translate[1]],
                      [0, 0, 1]])
    reachable_point = [list(reachable_point)]
    reachable_point[0][-1] = 1.0
    return tuple(np.squeeze(tform @ np.transpose(np.array(reachable_point))))[:2] + (rotate,)


def is_box_on_placement(body, surface):
    if get_aabb(surface).lower[0] < get_aabb(body).lower[0] and get_aabb(surface).lower[1] < get_aabb(body).lower[1] and \
            get_aabb(surface).upper[0] > get_aabb(body).upper[0] and get_aabb(surface).upper[1] > get_aabb(body).upper[
        1]:
        return True
    else:
        return False


def choose_grasps(placement_pose, grasps):  # TODO: hacked to always choose the grasp towards the same direction
    for i in range(len(grasps)):
        rotate = math.degrees(euler_from_quaternion(multiply(placement_pose.value, invert(grasps[i][0].value))[-1])[-1])
        if rotate > 45 and rotate < 135:
            return grasps[i]


def is_numerical_equal_two_tuples(tuple1, tuple2, precision):
    for i in range(len(tuple1)):
        if round(tuple1[i], precision) != round(tuple2[i], precision):
            return False
    return True


def get_custom_limits_legacy(robot, room_floors, target=None):
    if isinstance(room_floors, int):
        limits = get_aabb(room_floors)
        return (limits.lower[0], limits.upper[0]), (limits.lower[1], limits.upper[1])
    else:
        limits = None
        for room_floor in room_floors:
            if is_placement(robot, room_floor):
                limits = get_aabb(room_floor)
                if not target:
                    return (limits.lower[0], limits.upper[0]), (limits.lower[1], limits.upper[1])
                else:
                    break
        if not limits:  # This sometimes occurs when the robot is placed near the boundary of adjacent rooms.
            total_limit_lower0, total_limit_lower1 = float('inf'), float('inf')
            total_limit_upper0, total_limit_upper1 = float('-inf'), float('-inf')
            for room_floor in room_floors:
                limits = get_aabb(room_floor)
                if limits.lower[0] < total_limit_lower0: total_limit_lower0 = limits.lower[0]
                if limits.upper[0] > total_limit_upper0: total_limit_upper0 = limits.upper[0]
                if limits.lower[1] < total_limit_lower1: total_limit_lower1 = limits.lower[1]
                if limits.upper[1] > total_limit_upper1: total_limit_upper1 = limits.upper[1]
            return (total_limit_lower0, total_limit_upper0), (total_limit_lower1, total_limit_upper1)
        else:
            target_limits = get_aabb(target)
            if target_limits.lower[0] < limits.lower[0]:
                total_limit_lower0 = target_limits.lower[0]
            else:
                total_limit_lower0 = limits.lower[0]
            if target_limits.upper[0] < limits.upper[0]:
                total_limit_upper0 = limits.upper[0]
            else:
                total_limit_upper0 = target_limits.upper[0]
            if target_limits.lower[1] < limits.lower[1]:
                total_limit_lower1 = target_limits.lower[1]
            else:
                total_limit_lower1 = limits.lower[1]
            if target_limits.upper[1] < limits.upper[1]:
                total_limit_upper1 = limits.upper[1]
            else:
                total_limit_upper1 = target_limits.upper[1]
            return (total_limit_lower0, total_limit_upper0), (total_limit_lower1, total_limit_upper1)


def apply_margin(base_goal, custom_limits, margin_to_walls):
    safe_base_goal = ()
    for i in range(2):  # We do this for base x and base y only
        if base_goal.value[0][i] < custom_limits[i][0] + margin_to_walls:
            safe_base_goal += (custom_limits[i][0] + margin_to_walls,)
        elif base_goal.value[0][i] > custom_limits[i][1] - margin_to_walls:
            safe_base_goal += (custom_limits[i][1] - margin_to_walls,)
        else:
            safe_base_goal += (base_goal.value[i][0],)
    safe_base_goal += (base_goal.value[0][2],)
    return (safe_base_goal, base_goal.value[1])


def plan_cartesian_motion_legacy(robot, first_joint, target_link, waypoint_poses,
                                 max_iterations=200, max_time=INF, custom_limits={}, **kwargs):
    # TODO: fix stationary joints
    # TODO: pass in set of movable joints and take least common ancestor
    # TODO: update with most recent bullet updates
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics.py
    # https://github.com/bulletphysics/bullet3/blob/master/examples/pybullet/examples/inverse_kinematics_husky_kuka.py
    # TODO: plan a path without needing to following intermediate waypoints

    lower_limits, upper_limits = get_custom_limits(robot, get_movable_joints(robot), custom_limits)
    lower_limits = (custom_limits[0][0], custom_limits[1][0]) + lower_limits[2:]
    upper_limits = (custom_limits[0][1], custom_limits[1][1]) + upper_limits[2:]
    sub_robot, selected_joints, sub_target_link = create_sub_robot(robot, first_joint, target_link)
    sub_joints = get_movable_joints(sub_robot)
    # null_space = get_null_space(robot, selected_joints, custom_limits=custom_limits)
    null_space = None

    solutions = []
    for target_pose in waypoint_poses:
        start_time = time.time()
        for iteration in irange(max_iterations):
            if elapsed_time(start_time) >= max_time:
                remove_body(sub_robot)
                return None
            sub_kinematic_conf = inverse_kinematics_helper(sub_robot, sub_target_link, target_pose,
                                                           null_space=null_space)
            if sub_kinematic_conf is None:
                remove_body(sub_robot)
                return None
            set_joint_positions(sub_robot, sub_joints, sub_kinematic_conf)
            if is_pose_close(get_link_pose(sub_robot, sub_target_link), target_pose, **kwargs):
                set_joint_positions(robot, selected_joints, sub_kinematic_conf)
                kinematic_conf = get_configuration(robot)
                if not all_between(lower_limits, kinematic_conf, upper_limits):
                    # movable_joints = get_movable_joints(robot)
                    # print([(get_joint_name(robot, j), l, v, u) for j, l, v, u in
                    #       zip(movable_joints, lower_limits, kinematic_conf, upper_limits) if not (l <= v <= u)])
                    # print("Limits violated")
                    # wait_if_gui()
                    remove_body(sub_robot)
                    return None
                # print("IK iterations:", iteration)
                solutions.append(kinematic_conf)
                break
        else:
            remove_body(sub_robot)
            return None
    # TODO: finally:
    remove_body(sub_robot)
    return solutions


def pause_pybullet(physics_client_id, secs=float("inf")):
    """Pause the simulation at any point so that if you have the viewer
    on, you can look around. Useful for development of environments.
    """
    start_time = time.time()
    while True:
        p.setGravity(0., 0., -10., physicsClientId=physics_client_id)
        time.sleep(0.1)
        if time.time() - start_time > secs:
            break


def get_move_action(gripper_position, target_position, gain=5,
                    max_vel_norm=1, close_gripper=False):
    """Move an end effector to a position.
    """
    # Get the currents
    target_position = np.array(target_position)
    gripper_position = np.array(gripper_position)
    action = gain * (target_position - gripper_position)
    action_norm = np.linalg.norm(action)
    if action_norm > max_vel_norm:
        action = action * max_vel_norm / action_norm

    if close_gripper:
        gripper_action = -0.1
    else:
        gripper_action = 0.
    action = np.hstack((action, gripper_action))

    return action


def inverse_kinematics(body_id, end_effector_id, target_position,
                       target_orientation, joint_indices, physics_client_id=-1):
    """
    Parameters
    ----------
    body_id : int
    end_effector_id : int
    target_position : (float, float, float)
    target_orientation : (float, float, float, float)
    joint_indices : [ int ]

    Returns
    -------
    joint_poses : [ float ] * len(joint_indices)
    """
    lls, uls, jrs, rps = get_joint_ranges(body_id, joint_indices,
                                          physics_client_id=physics_client_id)

    all_joint_poses = p.calculateInverseKinematics(
        body_id, end_effector_id, target_position,
        targetOrientation=target_orientation,
        lowerLimits=lls, upperLimits=uls, jointRanges=jrs, restPoses=rps,
        physicsClientId=physics_client_id)

    # Find the free joints
    free_joint_indices = []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)
    for idx in range(num_joints):
        joint_info = p.getJointInfo(body_id, idx,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            free_joint_indices.append(idx)

    # Find the poses for the joints that we want to move
    joint_poses = []

    for idx in joint_indices:
        free_joint_idx = free_joint_indices.index(idx)
        joint_pose = all_joint_poses[free_joint_idx]
        joint_poses.append(joint_pose)

    return joint_poses


def get_joint_ranges(body_id, joint_indices, physics_client_id=-1):
    """
    Parameters
    ----------
    body_id : int
    joint_indices : [ int ]

    Returns
    -------
    lower_limits : [ float ] * len(joint_indices)
    upper_limits : [ float ] * len(joint_indices)
    joint_ranges : [ float ] * len(joint_indices)
    rest_poses : [ float ] * len(joint_indices)
    """
    lower_limits, upper_limits, joint_ranges, rest_poses = [], [], [], []

    num_joints = p.getNumJoints(body_id, physicsClientId=physics_client_id)

    for i in range(num_joints):
        joint_info = p.getJointInfo(body_id, i,
                                    physicsClientId=physics_client_id)

        # Fixed joint so ignore
        qIndex = joint_info[3]
        if qIndex <= -1:
            continue

        ll, ul = -2., 2.
        jr = 2.

        # For simplicity, assume resting state == initial state
        rp = p.getJointState(body_id, i, physicsClientId=physics_client_id)[0]

        # Fix joints that we don't want to move
        if i not in joint_indices:
            ll, ul = rp - 1e-8, rp + 1e-8
            jr = 1e-8

        lower_limits.append(ll)
        upper_limits.append(ul)
        joint_ranges.append(jr)
        rest_poses.append(rp)

    return lower_limits, upper_limits, joint_ranges, rest_poses


def get_kinematic_chain(robot_id, end_effector_id, physics_client_id=-1):
    """
    Get all of the free joints from robot base to end effector.

    Includes the end effector.

    Parameters
    ----------
    robot_id : int
    end_effector_id : int
    physics_client_id : int

    Returns
    -------
    kinematic_chain : [ int ]
        Joint ids.
    """
    kinematic_chain = []
    while end_effector_id > 0:
        joint_info = p.getJointInfo(robot_id, end_effector_id,
                                    physicsClientId=physics_client_id)
        if joint_info[3] > -1:
            kinematic_chain.append(end_effector_id)
        end_effector_id = joint_info[-1]
    return kinematic_chain


def create_shelf(w, l, h, set_point, sim_id):
    link_vis = []
    link_cols = []
    link_pos = []

    # Left side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0] - w / 2, set_point[1], set_point[2] + h / 2])
    # Right side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[0.01 / 2, l / 2, h / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0] + w / 2, set_point[1], set_point[2] + h / 2])
    # Back side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[w / 2, 0.01 / 2, h / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[w / 2, 0.01 / 2, h / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0], set_point[1] + l / 2, set_point[2] + h / 2])
    # Top side
    link_cols.append(p.createCollisionShape(
        p.GEOM_BOX, halfExtents=[w / 2, l / 2, 0.01 / 2],
        physicsClientId=sim_id))
    link_vis.append(p.createVisualShape(
        p.GEOM_BOX, halfExtents=[w / 2, l / 2, 0.01 / 2],
        rgbaColor=(0.6, 0.3, 0.0, 0.5),
        physicsClientId=sim_id))
    link_pos.append([set_point[0], set_point[1], set_point[2] + h])

    return p.createMultiBody(
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
        physicsClientId=sim_id)


def create_shelf_placement(w, l, h, mass=STATIC_MASS, color=RED, **kwargs):
    collision_id, visual_id = create_shape(get_box_geometry(w, l, h), color=color, **kwargs)
    return create_body(collision_id, visual_id, mass=mass)


def store_path(path=None, robot=None, env_name=None, arm=None, grasp_type=None, num_objs=None):
    db = {}
    db['path'] = path
    db['robot'] = robot
    db['env_name'] = env_name
    db['arm'] = arm
    db['grasp_type'] = grasp_type
    db['num_objs'] = num_objs

    path = join_paths(get_parent_dir(__file__), os.pardir, '../')
    dbfile = open(path + '/data/path_{}'.format(datetime.now()), 'ab')
    pickle.dump(db, dbfile)
    dbfile.close()


def store_data(data, opt):
    db = {}
    db['sym_actions'] = data._tot_sym_actions
    db['base_states'] = data._tot_base_states
    db['arm_states'] = data._tot_arm_states
    db['obj_states'] = data._tot_obj_states
    db['obj_ids'] = data._tot_obj_ids
    db['configs'] = data._tot_configs
    db['hand_hold'] = data._tot_hand_hold
    db['feasibilities'] = data._tot_feasibilities
    db['steps'] = data._tot_steps

    path = join_paths(get_parent_dir(__file__), os.pardir, os.pardir, "data")
    if opt.env_name == "namo":
        env_info = opt.env_name
    else:
        env_info = "{}_obj".format("rand" if opt.generalization else opt.num_objs)
    dbfile = open(path + '/data_{}_seed_{}'.format(env_info, opt.seed), 'ab')
    pickle.dump(db, dbfile)
    dbfile.close()


class SaveData(object):
    def __init__(self):
        self._tot_sym_actions = []
        self._tot_base_states = []
        self._tot_arm_states = []
        self._tot_obj_states = []
        self._tot_obj_ids = []
        self._tot_configs = []
        self._tot_hand_hold = []
        self._tot_feasibilities = []
        self._tot_worlds = []
        self._tot_steps = []

    def init(self, sym_actions, base_states, arm_states, obj_states, obj_ids, configs, hand_hold, feasibilities, steps):
        self._sym_actions = []
        self._base_states = []
        self._arm_states = []
        self._obj_states = []
        self._obj_ids = []
        self._configs = []
        self._hand_hold = []
        self._feasibilities = []
        self._steps = []
        self._sym_actions.append(sym_actions)
        self._base_states.append(base_states)
        self._arm_states.append(arm_states)
        self._obj_states.append(obj_states)
        self._obj_ids.append(obj_ids)
        self._configs.append(configs)
        self._hand_hold.append(hand_hold)
        self._feasibilities.append(feasibilities)
        self._steps.append(steps)

    def add_init(self, sym_actions, configs, steps):
        self._sym_actions.append(sym_actions)
        self._configs.append(configs)
        self._steps.append(steps)

    def add_rest(self, base_states, arm_states, obj_states, obj_ids, hand_hold, feasibilities):
        self._base_states.append(base_states)
        self._arm_states.append(arm_states)
        self._obj_states.append(obj_states)
        self._obj_ids.append(obj_ids)
        self._hand_hold.append(hand_hold)
        self._feasibilities.append(feasibilities)

    def tot_add(self):
        self._tot_sym_actions.append(self._sym_actions)
        self._tot_base_states.append(self._base_states)
        self._tot_arm_states.append(self._arm_states)
        self._tot_obj_states.append(self._obj_states)
        self._tot_obj_ids.append(self._obj_ids)
        self._tot_configs.append(self._configs)
        self._tot_hand_hold.append(self._hand_hold)
        self._tot_feasibilities.append(self._feasibilities)
        self._tot_steps.append(self._steps)


class SavePath(object):
    def __init__(self, robot, arm):
        self._robot = robot
        self._arm = arm

        self._actions = []
        self._paths = []
        self._attachments = []

    def add(self, actions=[], paths=[], attachments=[]):
        self._actions.append(actions)
        self._paths.append(paths)
        self._attachments.append(attachments)

    def delete(self):
        self._actions.pop()
        self._paths.pop()
        self._attachments.pop()


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
