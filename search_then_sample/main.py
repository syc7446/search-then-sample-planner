#!/usr/bin/env python

from search_then_sample.examples.pick_place_env import PickPlaceEnvironment
from pybullet_planning.pybullet_tools.utils import connect, disable_real_time

env_name = "pickplace" # Options: pickplace

connect(use_gui=True)
disable_real_time()

if env_name == "pickplace":
    num_samples_per_step = 10 # related to backtracking
    env = PickPlaceEnvironment(num_objs=1, seed=0)
    a = 1