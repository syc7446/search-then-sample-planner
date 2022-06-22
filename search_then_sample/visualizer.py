import pickle
import os
import argparse

from pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir, set_joint_positions, joint_from_name, \
    wait_for_duration, WorldSaver, connect, disable_real_time, set_camera_pose, VideoSaver
from pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS
from search_then_sample.utils.search_then_sample_utils import SaveData
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment
# from search_then_sample.examples.pack_in_shelf_env import PackInShelfEnvironment
from search_then_sample.examples.pack_in_shelf_env_for_video import PackInShelfEnvironment
# from search_then_sample.examples.namo_env import NAMOEnvironment
from search_then_sample.examples.namo_env_for_video import NAMOEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('--load_file', type=str, required=True, help="file path and name")

opt = parser.parse_args()
print(opt)

path = join_paths(get_parent_dir(__file__), os.pardir, '.')
dbfile = open(path+'/data/'+opt.load_file, 'rb')
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


sim_id = connect(use_gui=True)
disable_real_time()
set_camera_pose(camera_point=(2, 4.5, 5), target_point=(2.25, 4.5, 0))

if env_name == "pickplace":
    env = PickPlaceEnvironment(num_objs=num_objs, seed=0)
elif env_name == "packinshelf":
    env = PackInShelfEnvironment(num_objs=num_objs, sim_id=sim_id, seed=0)
elif env_name == "namo":
    env = NAMOEnvironment(num_objs=num_objs, sim_id=sim_id, seed=0)
save_data = SaveData()  # TODO: we do not save data here
state, _, _, _ = env.initial_pybullet_setup(arm, grasp_type, save_data)

video_saver = VideoSaver(env_name+'.mp4')

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
        elif actions[i][j] == 'both_arms':
            set_joint_positions(robot, [joint_from_name(robot, name) for name in PR2_GROUPS['left_arm']], paths[i][j][0])
            set_joint_positions(robot, [joint_from_name(robot, name) for name in PR2_GROUPS['right_arm']], paths[i][j][1])
            if attachments[i][j] is not None:
                attachments[i][j].assign()
            wait_for_duration(0.01)

video_saver.restore()
dbfile.close()