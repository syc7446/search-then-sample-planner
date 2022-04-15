import pickle
import os

from pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir, set_joint_positions, joint_from_name, \
    wait_for_duration, WorldSaver, connect, disable_real_time, set_camera_pose
from pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment

path = join_paths(get_parent_dir(__file__), os.pardir, '.')
dbfile = open(path+'/data/save_data2022-04-14 21:58:01.820924', 'rb')
db = pickle.load(dbfile)
for keys in db:
    print(keys, '=>', db[keys])

paths = db['path']._paths
attachments = db['path']._attachments
actions = db['path']._actions
robot = db['robot']
env_name = db['env_name']
arm = db['arm']
grasp_type = db['grasp_type']
num_objs = db['num_objs']
num_sample_trials = db['num_sample_trials']
margin_to_walls = db['margin_to_walls']


connect(use_gui=True)
disable_real_time()
set_camera_pose(camera_point=(-1, 0, 3), target_point=(0, 0, 0))

if env_name == "pickplace":
    env = PickPlaceEnvironment(num_objs=num_objs, num_sample_trials=num_sample_trials,
                               margin_to_walls=margin_to_walls, seed=0)
state = env.initial_pybullet_setup(arm, grasp_type)

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