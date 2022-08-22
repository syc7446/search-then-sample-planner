import os
from subprocess import Popen

path_filter = "imit_att_10_obj_500_task_seed_0"

generalization = False
save_data = False
env_name = "packinshelf"                # namo, packinshelf
num_objs = 10
num_probs = 1
num_processes = 1
seed_offset = 100008            # 100000, 100020, 100040, 100060
num_resamples = 30
num_samples_per_step = 30
planner_name = "backjump_forgetting"    # backtrack_forgetting, backtrack_batch_sampling, backjump_forgetting, backjump_batch_sampling
learner_base = "/scratch/cluster/zzwang/culprit_detection_learner/interesting_models/"

cuda_id = 0
for dirname in os.listdir(learner_base):
    if path_filter not in dirname:
        continue
    for i in range(num_processes):
        command = "nohup python main.py"
        command += " --env_name={}".format(env_name)
        command += " --seed={}".format(seed_offset)
        command += " --num_objs={}".format(num_objs)
        command += " --num_probs={}".format(num_probs)
        command += " --num_resamples={}".format(num_resamples)
        command += " --num_samples_per_step={}".format(num_samples_per_step)
        command += " --planner_name={}".format(planner_name)
        if "backjump" in planner_name:
            if "imit" in dirname:
                learner_name = "imitation"
            elif "pfl" in dirname:
                learner_name = "plan_feasibility"
            else:
                raise NotImplementedError
            command += " --learner_name={}".format(learner_name)
            command += " --learner_path={}".format(learner_base + dirname)
            command += " --cuda_id={}".format(cuda_id)
        if save_data:
            command += " --save_data"
        if generalization:
            command += " --generalization"
        print(command)
        print()
        # Popen(command, shell=True)

    if "backtrack" in planner_name:
        break
    else:
        cuda_id += 1
