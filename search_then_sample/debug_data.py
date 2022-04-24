import pickle
import os
import argparse

from pybullet_planning.pybullet_tools.utils import join_paths, get_parent_dir, set_joint_positions, joint_from_name, \
    wait_for_duration, WorldSaver, connect, disable_real_time, set_camera_pose
from pybullet_planning.pybullet_tools.pr2_utils import PR2_GROUPS
from search_then_sample.examples.pick_place_env import PickPlaceEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('--load_file', type=str, required=True, help="file path and name")

opt = parser.parse_args()
print(opt)

path = join_paths(get_parent_dir(__file__), os.pardir, '.')
dbfile = open(path+'/data/'+opt.load_file, 'rb')
db = pickle.load(dbfile)
for keys in db:
    print(keys, '=>', db[keys])

sym_actions = db['sym_actions']
base_states = db['base_states']
arm_states = db['arm_states']
obj_states = db['obj_states']
configs = db['configs']
hand_hold = db['hand_hold']
feasibilities = db['feasibilities']
steps = db['steps']
