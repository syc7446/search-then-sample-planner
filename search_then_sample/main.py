#!/usr/bin/env python

import argparse
import time
import random
import numpy as np
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment
from search_then_sample.examples.pack_in_shelf_env import PackInShelfEnvironment
from search_then_sample.utils.planner import Planner, PlanningExhausted, PlanningTimeout
import search_then_sample.utils.ground_truth_ndrs as ground_truth_ndrs
import search_then_sample.utils.constants as constants
from search_then_sample.utils.search_then_sample_utils import SaveData, store_path, store_data
from pybullet_planning.pybullet_tools.utils import connect, disconnect, disable_real_time, set_camera_pose, WorldSaver


parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default="packinshelf") # options: 'pickplace', 'packinshelf'
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--arm', type=str, default='left')
parser.add_argument('--grasp_type', type=str, default='side')
parser.add_argument('--num_objs', type=int, default=2, help='number of movable objects')
parser.add_argument('--num_samples_per_step', type=int, default=10)
parser.add_argument('--num_probs', type=int, default=1, help='number of problems to solve to gather data')
parser.add_argument('--use_gui', action='store_true')
parser.add_argument('--save_merged_path', action='store_true', help='save merged path to visualize')
parser.add_argument('--save_data', action='store_true', help='save backtracking data')

opt = parser.parse_args()
print(opt)


save_data = SaveData()
for i in range(opt.num_probs):
    random.seed(opt.seed)
    np.random.seed(opt.seed)

    sim_id = connect(use_gui=opt.use_gui)
    disable_real_time()
    set_camera_pose(camera_point=(-1.0, 0, 3), target_point=(0, 0, 0))

    if opt.env_name == "pickplace":
        env = PickPlaceEnvironment(num_objs=opt.num_objs, seed=opt.seed)
    elif opt.env_name == "packinshelf":
        env = PackInShelfEnvironment(num_objs=opt.num_objs, sim_id=sim_id, seed=opt.seed)

    planner = Planner(seed=opt.seed, timeout=constants.PLANNER_TIMEOUT,
                      heuristic_name="PyperplanHAddHeuristic",
                      num_samples_per_step=opt.num_samples_per_step)
    all_ndrs = ground_truth_ndrs.get_groundtruth_ndrs(opt.env_name, env)

    state, robot, save_data = env.initial_pybullet_setup(opt.arm, opt.grasp_type, save_data)
    start_time = time.time()
    try:
        plan, save_path, save_data = planner.plan(env, state, all_ndrs, save_data)
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
    store_data(data=save_data)