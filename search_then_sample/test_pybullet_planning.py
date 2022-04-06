#!/usr/bin/env python

import path
import sys

directory = path.Path(__file__).abspath()
sys.path.append(directory.parent.parent)

import time
import random
import pybullet as p

from pybullet_planning.pybullet_tools.utils import connect, disable_real_time, load_model, Pose, Point, \
    set_default_camera, stable_z, dump_world, wait_if_gui, plan_joint_motion, set_joint_positions, \
    joints_from_names, get_link_pose, link_from_name, BLOCK_URDF
from pybullet_planning.pybullet_tools.pr2_utils import DRAKE_PR2_URDF, set_pose, get_disabled_collisions, \
    rightarm_from_leftarm, open_arm, TOP_HOLDING_LEFT_ARM, SIDE_HOLDING_LEFT_ARM, REST_LEFT_ARM, PR2_GROUPS
from pybullet_planning.pybullet_tools.pr2_primitives import get_grasp_gen, get_ik_fn, get_motion_gen, get_ik_ir_gen, \
    get_stable_gen, Conf
from pybullet_planning.pybullet_tools.pr2_problems import holding_problem


def test_arm_motion(pr2, left_joints, arm_goal):
    disabled_collisions = get_disabled_collisions(pr2)
    wait_if_gui('Plan Arm?')
    arm_path = plan_joint_motion(pr2, left_joints, arm_goal, disabled_collisions=disabled_collisions)
    if arm_path is None:
        print('Unable to find an arm path')
        return
    print(len(arm_path))
    for q in arm_path:
        set_joint_positions(pr2, left_joints, q)
        #wait_if_gui('Continue?')
        time.sleep(0.01)


def test_ikfast(pr2):
    from pybullet_planning.pybullet_tools.ikfast.pr2.ik import get_tool_pose, get_ik_generator
    left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
    #right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
    torso_joints = joints_from_names(pr2, PR2_GROUPS['torso'])
    torso_left = torso_joints + left_joints
    print(get_link_pose(pr2, link_from_name(pr2, 'l_gripper_tool_frame')))
    # print(forward_kinematics('left', get_joint_positions(pr2, torso_left)))
    print(get_tool_pose(pr2, 'left'))

    arm = 'left'
    pose = get_tool_pose(pr2, arm)
    generator = get_ik_generator(pr2, arm, pose, torso_limits=False)
    for i in range(100):
        solutions = next(generator)
        print(i, len(solutions))
        for q in solutions:
            set_joint_positions(pr2, torso_left, q)
            wait_if_gui()


def plan(problem):
    robot = problem.robot
    arm = problem.arms[0]
    body = problem.movable[0]
    table = problem.fixed[1]

    grasp_gen_fn = get_grasp_gen(problem, collisions=True)
    # motion_fn = get_motion_gen(problem)
    placement_gen_fn = get_stable_gen(problem)
    ik_ir_fn = get_ik_ir_gen(problem)

    placement_gen = placement_gen_fn(body, table)
    grasps = list(grasp_gen_fn(body))

    (p,) = next(placement_gen)
    (g,) = random.choice(grasps)
    # for i in range(len(grasps)):
    # (g,) = grasps[i]
    output = next(ik_ir_fn(arm, body, p, g), None)
    if output:
        print('output found!')
    else:
        print('no output exists...')

    if output is None:
        print('Failed to find a solution.')
    # else:
    #     (_, ac) = output
    #     [at, ] = ac.commands
    #     at.path[-1].assign()
    #     gripper_from_base = multiply(invert(get_link_pose(robot, tool_link)), get_base_pose(robot))
    #     gripper_from_base_list.append(gripper_from_base)
    #     print('{} / {} [{:.3f}]'.format(
    #         len(gripper_from_base_list), num_samples, elapsed_time(start_time)))
    #     wait_if_gui()


def main():
    connect(use_gui=True)
    disable_real_time()
    # pr2 = load_model(DRAKE_PR2_URDF, fixed_base=True)
    # plane = p.loadURDF("plane.urdf")
    # block = load_model(BLOCK_URDF, fixed_base=False)
    # set_pose(block, Pose(Point(x=5., z=stable_z(block, plane))))
    # set_default_camera()
    # dump_world()
    #
    # arm_start = SIDE_HOLDING_LEFT_ARM
    # arm_goal = TOP_HOLDING_LEFT_ARM
    #
    # left_joints = joints_from_names(pr2, PR2_GROUPS['left_arm'])
    # right_joints = joints_from_names(pr2, PR2_GROUPS['right_arm'])
    # torso_joints = joints_from_names(pr2, PR2_GROUPS['torso'])
    # set_joint_positions(pr2, left_joints, arm_start)
    # set_joint_positions(pr2, right_joints, rightarm_from_leftarm(REST_LEFT_ARM))
    # set_joint_positions(pr2, torso_joints, [0.2])
    # open_arm(pr2, 'left')

    problem = holding_problem()
    plan(problem)


if __name__ == '__main__':
    main()