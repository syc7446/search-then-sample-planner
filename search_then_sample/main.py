#!/usr/bin/env python

import time
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment
from search_then_sample.utils.planner import Planner, PlanningExhausted, PlanningTimeout
import search_then_sample.utils.ground_truth_ndrs as ground_truth_ndrs
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import storeData
from pybullet_planning.pybullet_tools.utils import connect, disable_real_time, set_camera_pose, WorldSaver

### Parameters
env_name = "pickplace" # Options: pickplace
arm = 'left'
grasp_type = 'side'
num_objs = 1
# num_rooms = 3
num_sample_trials = 20 # number of trials in sampling
num_samples_per_step = 10 # related to backtracking
margin_to_walls = .5
is_gui_debug = True
###


connect(use_gui=is_gui_debug)
disable_real_time()
set_camera_pose(camera_point=(-1.0, 0, 3), target_point=(0, 0, 0))

if env_name == "pickplace":
    env = PickPlaceEnvironment(num_objs=num_objs, num_sample_trials=num_sample_trials,
                               margin_to_walls=margin_to_walls, seed=0)

planner = Planner(seed=0, timeout=constants.PLANNER_TIMEOUT,
                  heuristic_name="PyperplanHAddHeuristic",
                  num_samples_per_step=num_samples_per_step)
all_ndrs = ground_truth_ndrs.get_groundtruth_ndrs(env_name, env)

state, robot = env.initial_pybullet_setup(arm, grasp_type)
start_time = time.time()
try:
    plan, merged_path = planner.plan(env, state, all_ndrs)
    storeData(path=merged_path, robot=robot)
except (PlanningExhausted, PlanningTimeout) as e:
    print(f"planning failed with error: {e}")
print("finished in {:.5f} seconds".format(time.time()-start_time))