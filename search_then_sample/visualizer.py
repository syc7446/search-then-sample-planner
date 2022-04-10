import pickle
import os

from pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir, set_joint_positions, joint_from_name, \
    wait_for_duration, WorldSaver, connect, disable_real_time, set_camera_pose
from pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment


### Parameters
env_name = "pickplace" # Options: pickplace
arm = 'left'
grasp_type = 'side'
num_objs = 1
num_rooms = 3
num_sample_trials = 20
margin_to_walls = .5
###

connect(use_gui=True)
disable_real_time()
set_camera_pose(camera_point=(-0.4, 3, 5), target_point=(0, 3, 0))

env = PickPlaceEnvironment(num_objs=num_objs, num_rooms=num_rooms,
                           num_sample_trials=num_sample_trials, margin_to_walls=margin_to_walls,
                           seed=0)
state = env.initial_pybullet_setup(arm, grasp_type)

path = join_paths(get_parent_dir(__file__), os.pardir, '.')
dbfile = open(path+'/data/save_data', 'rb')
db = pickle.load(dbfile)
for keys in db:
    print(keys, '=>', db[keys])

paths = db['path']._paths
attachments = db['path']._attachments
actions = db['path']._actions
robot = db['robot']

for i in range(len(actions)):
    for j in range(len(actions[i])):
        if actions[i][j] == 'base':
            for q in paths[i][j]:
                set_joint_positions(robot, [joint_from_name(robot, name) for name in PR2_GROUPS['base']], q)
                if attachments[i][j] is not None:
                    attachments[i][j].assign()
                wait_for_duration(0.02)
        elif actions[i][j] == 'arm':
            for q in paths[i][j]:
                set_joint_positions(robot, [joint_from_name(robot, name) for name in PR2_GROUPS['left_arm']], q)
                if attachments[i][j] is not None:
                    attachments[i][j].assign()
                wait_for_duration(0.01)

dbfile.close()