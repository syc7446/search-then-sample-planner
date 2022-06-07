#!/usr/bin/env python

import sys
  
# setting path
sys.path.append('..')

import argparse
import time
import torch
import random
import numpy as np
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment
from search_then_sample.examples.pack_in_shelf_env import PackInShelfEnvironment
from search_then_sample.examples.namo_env import NAMOEnvironment
from search_then_sample.planner.backtrack_newsample_planner import BacktrackNewsamplePlanner
from search_then_sample.planner.backtrack_resample_planner import BacktrackResamplePlanner
from search_then_sample.planner.backjump_newsample_planner import BackjumpNewsamplePlanner, PlanningExhausted, PlanningTimeout
import search_then_sample.utils.ground_truth_ndrs as ground_truth_ndrs
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import SaveData, store_path, store_data
from search_then_sample.constants import PACKING_CAMERA_POINT, PACKING_TARGET_POINT, NAMO_CAMERA_POINT, NAMO_TARGET_POINT
from pybullet_planning.pybullet_tools.utils import connect, disconnect, disable_real_time, set_camera_pose

np.set_printoptions(precision=3, suppress=True)
torch.set_printoptions(precision=3, sci_mode=False)

'''
Arguments
env_name options: 'pickplace', 'packinshelf', 'namo'
planner_name options: 'backtrack_newsample', 'backtrack_resample', 'backjump_newsample', 'backjump_resample'
learner_name options: 'plan_feasibility', 'imitation'
'''

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="packinshelf")
parser.add_argument('--planner_name', type=str, default='backjump_newsample')
parser.add_argument('--learner_name', type=str, default='plan_feasibility')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--arm', type=str, default='left')
parser.add_argument('--grasp_type', type=str, default='side')
parser.add_argument('--num_objs', type=int, default=10, help='number of movable objects')
parser.add_argument('--num_resamples', type=int, default=1, help='number of resamples to increase samples')
parser.add_argument('--num_samples_per_step', type=int, default=10, help='number of samples per step in each skeleton')
parser.add_argument('--num_probs', type=int, default=1, help='number of problems to solve to gather data')
parser.add_argument('--generalization', action='store_true')
parser.add_argument('--use_gui', action='store_true')
parser.add_argument('--save_merged_path', action='store_true', help='save merged path to visualize')
parser.add_argument('--save_data', action='store_true', help='save backtracking data')

opt = parser.parse_args()
print(opt)

random.seed(opt.seed)
np.random.seed(opt.seed)
save_data = SaveData()

for i in range(opt.num_probs):
    print("{}/{}".format(i + 1, opt.num_probs))

    sim_id = connect(use_gui=opt.use_gui)
    disable_real_time()

    if opt.env_name == "pickplace":
        set_camera_pose(camera_point=PACKING_CAMERA_POINT, target_point=PACKING_TARGET_POINT)
        env = PickPlaceEnvironment(num_objs=opt.num_objs, seed=opt.seed)
    elif opt.env_name == "packinshelf":
        num_objs = np.random.randint(1, opt.num_objs + 1) if opt.generalization else opt.num_objs
        set_camera_pose(camera_point=PACKING_CAMERA_POINT, target_point=PACKING_TARGET_POINT)
        env = PackInShelfEnvironment(num_objs=num_objs, sim_id=sim_id, seed=opt.seed)
    elif opt.env_name == "namo":
        set_camera_pose(camera_point=NAMO_CAMERA_POINT, target_point=NAMO_TARGET_POINT)
        env = NAMOEnvironment(num_objs=opt.num_objs, sim_id=sim_id, seed=opt.seed)

    if opt.planner_name == 'backtrack_newsample':
        planner = BacktrackNewsamplePlanner(seed=opt.seed, timeout=constants.PLANNER_TIMEOUT,
                                            heuristic_name="PyperplanHAddHeuristic",
                                            num_samples_per_step=opt.num_samples_per_step)
    elif opt.planner_name == 'backtrack_resample':
        planner = BacktrackResamplePlanner(seed=opt.seed, timeout=constants.PLANNER_TIMEOUT,
                                           heuristic_name="PyperplanHAddHeuristic",
                                           num_samples_per_step=opt.num_samples_per_step,
                                           num_resamples=opt.num_resamples)
    elif opt.planner_name == 'backjump_newsample':
        planner = BackjumpNewsamplePlanner(seed=opt.seed, timeout=constants.PLANNER_TIMEOUT,
                                           heuristic_name="PyperplanHAddHeuristic",
                                           num_samples_per_step=opt.num_samples_per_step,
                                           learner_name=opt.learner_name)
    elif opt.planner_name == 'backjump_resample':
        raise NotImplementedError

    all_ndrs = ground_truth_ndrs.get_groundtruth_ndrs(opt.env_name, env)

    state, robot, save_data, saved_world = env.initial_pybullet_setup(opt.arm, opt.grasp_type, save_data)
    start_time = time.time()
    try:
        plan, save_path, save_data = planner.plan(env, state, all_ndrs, save_data, saved_world)
        if opt.save_merged_path:
            store_path(path=save_path, robot=robot, env_name=opt.env_name, arm=opt.arm,
                      grasp_type=opt.grasp_type, num_objs=opt.num_objs)
        if opt.save_data:
            save_data.tot_add()
    except (PlanningExhausted, PlanningTimeout) as e:
        print(f'planning failed with error: {e}')
    print('finished in {:.5f} seconds'.format(time.time()-start_time))
    del env
    del planner
    disconnect()
    print('problem {} is solved'.format(i))

if opt.save_data:
    store_data(data=save_data, opt=opt)