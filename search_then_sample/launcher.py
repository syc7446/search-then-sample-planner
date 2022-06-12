from subprocess import Popen

generalization = False
save_data = True
env_name = "namo"                       # namo, packinshelf
num_objs = 10
num_probs = 4
num_processes = 50
seed_offset = 210
num_samples_per_step = 3
planner_name = "backtrack_newsample"     # backjump_newsample, backtrack_newsample
learner_name = "imitation"              # imitation, plan_feasibility
learner_path = "/scratch/cluster/zzwang/culprit_detection_learner/interesting_models/old_results/imit_rnn_rand_obj_seed_0"
cuda_id = 2

for i in range(num_processes):
    command = "nohup python main.py"
    command += " --env_name={}".format(env_name)
    command += " --seed={}".format(i + seed_offset)
    command += " --num_objs={}".format(num_objs)
    command += " --num_probs={}".format(num_probs)
    command += " --num_samples_per_step={}".format(num_samples_per_step)
    command += " --planner_name={}".format(planner_name)
    if "backjump" in planner_name:
        command += " --learner_name={}".format(learner_name)
        command += " --learner_path={}".format(learner_path)
        command += " --cuda_id={}".format(cuda_id)
    if save_data:
        command += " --save_data"
    if generalization:
        command += " --generalization"
    print(command)
    Popen(command, shell=True)
