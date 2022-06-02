from subprocess import Popen

generalization = False
num_objs = 10
num_probs = 10
num_processes = 50
seed_offset = 0
num_samples_per_step = 30

for i in range(num_processes):
    command = "nohup python main.py"
    command += " --seed={}".format(i + seed_offset)
    command += " --num_objs={}".format(num_objs)
    command += " --num_probs={}".format(num_probs)
    command += " --num_samples_per_step={}".format(num_samples_per_step)
    command += " --save_data"
    if generalization:
        command += " --generalization"
    # print(command)
    Popen(command, shell=True)
