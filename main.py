"""Top-level script.
"""

import time
from blocks_env import BlocksEnvironment
from bins_env import BinsEnvironment
import ground_truth_ndrs
from planner import Planner, PlanningExhausted, PlanningTimeout
import constants


env_name = "blocks"
# env_name = "bins"

if env_name == "blocks":
    num_samples_per_step = 1  # no backtracking required
    env = BlocksEnvironment(num_objs=4, seed=0)
if env_name == "bins":
    num_samples_per_step = 10
    env = BinsEnvironment(num_objs=3, seed=0)
planner = Planner(seed=0, timeout=constants.PLANNER_TIMEOUT,
                  heuristic_name="PyperplanHAddHeuristic",
                  num_samples_per_step=num_samples_per_step)
all_ndrs = ground_truth_ndrs.get_groundtruth_ndrs(env_name, env)
for i, state in enumerate(env.get_train_initial_states()):
    print(f"\nRunning problem {i}")
    start_time = time.time()
    try:
        plan = planner.plan(env, state, all_ndrs)
    except (PlanningExhausted, PlanningTimeout) as e:
        print(f"planning failed with error: {e}")
        continue
    print("finished in {:.5f} seconds".format(time.time()-start_time))
