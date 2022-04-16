#!/usr/bin/env python

import argparse
import time
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment
from search_then_sample.utils.planner import Planner, PlanningExhausted, PlanningTimeout
import search_then_sample.utils.ground_truth_ndrs as ground_truth_ndrs
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import storeData
from pybullet_planning.pybullet_tools.utils import connect, disable_real_time, set_camera_pose, WorldSaver


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="pickplace")
parser.add_argument('--arm', type=str, default='left')
parser.add_argument('--grasp_type', type=str, default='side')
parser.add_argument('--num_objs', type=int, default=2, help='number of movable objects')
parser.add_argument('--num_samples_per_step', type=int, default=10)
parser.add_argument('--use_gui', action='store_true')

opt = parser.parse_args()
print(opt)


connect(use_gui=opt.use_gui)
disable_real_time()
set_camera_pose(camera_point=(-1.0, 0, 3), target_point=(0, 0, 0))

if opt.env_name == "pickplace":
    env = PickPlaceEnvironment(num_objs=opt.num_objs, seed=0)

planner = Planner(seed=0, timeout=constants.PLANNER_TIMEOUT,
                  heuristic_name="PyperplanHAddHeuristic",
                  num_samples_per_step=opt.num_samples_per_step)
all_ndrs = ground_truth_ndrs.get_groundtruth_ndrs(opt.env_name, env)

state, robot = env.initial_pybullet_setup(opt.arm, opt.grasp_type)
start_time = time.time()
try:
    plan, merged_path = planner.plan(env, state, all_ndrs)
    storeData(path=merged_path, robot=robot, env_name=opt.env_name, arm=opt.arm,
              grasp_type=opt.grasp_type, num_objs=opt.num_objs)
except (PlanningExhausted, PlanningTimeout) as e:
    print(f"planning failed with error: {e}")
print("finished in {:.5f} seconds".format(time.time()-start_time))