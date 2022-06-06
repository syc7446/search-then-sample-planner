import os
import pickle
import numpy as np
import networkx as nx
from scipy.spatial.transform import Rotation as R

from joblib import Parallel, delayed
from torch_geometric.utils import from_networkx


np.set_printoptions(precision=2, linewidth=np.inf, suppress=True)

BOX_SIZE = np.array([.07, .05])
TABLE_POSE_X, TABLE_POSE_Y = 0.0, 1.8

fname_filter = "data_10_obj"
data_to_prepare = ["pfl_test"]       # "pfl", "imit", "pfl_test"
dirname = "data"


def get_state_graph(obj_state, init_obj_state):
    moved_obj = []
    for obj_state_i, init_obj_state_i in zip(obj_state, init_obj_state):
        obj_state_i, init_obj_state_i = np.array(obj_state_i), np.array(init_obj_state_i)
        if np.linalg.norm(obj_state_i - init_obj_state_i) > 1e-5:
            moved_obj.append(obj_state_i)

    processed_obj = []
    for obj_state_i in moved_obj:
        x = obj_state_i[0] - TABLE_POSE_X
        y = obj_state_i[1] - TABLE_POSE_Y
        quat = obj_state_i[-4:]
        z_rot = R.from_quat(quat).as_euler('zyx')[0]
        processed_obj.append(np.array([x, y, np.cos(z_rot), np.sin(z_rot)], dtype=np.float32))

    graph = nx.DiGraph()
    for i, obj_state_i in enumerate(processed_obj):
        graph.add_node(i, x=obj_state_i)

    for i, obj_state_i in enumerate(processed_obj):
        for j, obj_state_j in enumerate(processed_obj):
            if i == j:
                graph.add_edge(i, j, edge_attr=np.zeros(4, dtype=np.float32))
                continue
            x1, y1, zcos1, zsin1 = obj_state_i
            x2, y2, zcos2, zsin2 = obj_state_j
            graph.add_edge(i, j, edge_attr=np.array([x2 - x1, y2 - y1,
                                                     zcos2 * zcos1 + zsin2 * zsin1,
                                                     zsin2 * zcos1 - zcos2 * zsin1],
                                                    dtype=np.float32))
    return from_networkx(graph)


def get_feasibility_likelihood(step, num_remaining_obj, steps, feasibilities):
    num_feas_at_steps = [0. for _ in range(num_remaining_obj)]
    num_infeas_at_steps = [0. for _ in range(num_remaining_obj)]
    for step_i, feas_i in zip(steps, feasibilities):
        if step_i <= step:
            break
        if feas_i:
            num_feas_at_steps[step_i - step - 1] += 1
        else:
            num_infeas_at_steps[step_i - step - 1] += 1

    fl = [num_feas_at_step_i / (num_feas_at_step_i + sum(num_infeas_at_steps[:i + 1]))
          for i, num_feas_at_step_i in enumerate(num_feas_at_steps)]

    return fl


def get_imit_label(step, steps, feasibilities):
    backjump_step = step
    for step_i, feas_i in zip(steps, feasibilities):
        if step_i == step and feas_i:
            break
        if step_i < backjump_step:
            backjump_step = step_i
    return backjump_step


def post_process_file(fname):
    print(fname)
    with open(fname, "rb") as f:
        db = pickle.load(f)

    print_brief = False
    if print_brief:
        for key in db:
            print(key, "->", len(db[key]))
            length = len(db[key][0])
        print("task steps", [s[-1] for s in db["steps"]])
        print()

        # for i in range(length):
        #     print(i)
        #     for key in db:
        #         print(key, "->", db[key][0][i])
        #     print()

    sym_actions = db["sym_actions"]
    base_states = db["base_states"]
    arm_states = db["arm_states"]
    obj_states = db["obj_states"]
    configs = db["configs"]
    hand_holds = db["hand_hold"]
    feasibilities = db["feasibilities"]
    steps = db["steps"]

    pfl_states = []
    pfl_obj_infos = []
    pfl_labels = []

    pfl_test_state_trajs = []
    pfl_test_obj_infos = []
    pfl_test_labels = []

    imit_state_trajs = []
    imit_obj_infos = []
    imit_labels = []

    num_tree = len(sym_actions)
    assert len(sym_actions) == len(base_states) == len(arm_states) == len(obj_states) == len(configs) == len(hand_holds) \
           == len(feasibilities) == len(steps)
    for i in range(num_tree):
        imit_obj_state_traj = []

        assert len(sym_actions[i]) == len(base_states[i]) == len(arm_states[i]) == len(obj_states[i]) == len(configs[i]) \
            == len(hand_holds[i]) == len(feasibilities[i]) == len(steps[i])

        tree_len = len(sym_actions[i])
        init_obj_state = obj_states[i][0]
        num_obj = len(init_obj_state)

        for j in range(tree_len):
            obj_state = obj_states[i][j]
            feasible = feasibilities[i][j]
            step = steps[i][j]

            # for plan feasibility likelihood data
            if "pfl" in data_to_prepare and step >= 0 and feasible:
                state_graph = get_state_graph(obj_state, init_obj_state)
                num_remaining_obj = num_obj - step - 1
                if num_remaining_obj == 0:
                    continue
                fl = get_feasibility_likelihood(step, num_remaining_obj, steps[i][j + 1:], feasibilities[i][j + 1:])
                pfl_states.append(state_graph)
                pfl_obj_infos.append(np.array([BOX_SIZE] * num_remaining_obj))
                pfl_labels.append(fl)

            # for imitation learning and plan feasibility likelihood TEST data
            if step >= 0 and ("pfl_test" in data_to_prepare or "imit" in data_to_prepare):
                imit_obj_state_traj = imit_obj_state_traj[:step]
                if feasible:
                    imit_obj_state_traj.append(obj_state)
                else:
                    imit_label = get_imit_label(step, steps[i][j + 1:], feasibilities[i][j + 1:])

                    # can succeed at the same step after a few more trials, which is not what backjumping aims for
                    if imit_label == step:
                        continue
                    assert imit_label < len(imit_obj_state_traj)

                    state_traj = [get_state_graph(obj_state_t, init_obj_state) for obj_state_t in imit_obj_state_traj]
                    imit_state_trajs.append(state_traj)
                    imit_obj_infos.append(BOX_SIZE)
                    imit_labels.append(imit_label)

                    t = len(state_traj)
                    pfl_test_state_trajs.append(state_traj)
                    pfl_test_obj_infos.append([np.array([BOX_SIZE] * (t - k)) for k in range(t)])
                    pfl_test_labels.append(imit_label)

    data = {"pfl_state_graphs": pfl_states,
            "pfl_obj_infos": pfl_obj_infos,
            "feasibility_likelihood": pfl_labels,
            "imit_state_graphs": imit_state_trajs,
            "imit_obj_infos": imit_obj_infos,
            "imitation_label": imit_labels,
            "pfl_test_state_trajs": pfl_test_state_trajs,
            "pfl_test_obj_infos": pfl_test_obj_infos,
            "pfl_test_labels": pfl_test_labels}

    print_data = False
    if print_data:
        for i in np.random.randint(len(data["feasibility_likelihood"]), size=5):
            print(i)
            for k, v in data.items():
                print(k)
                v = v[i]
                if isinstance(v, nx.DiGraph):
                    for n in v.nodes(data=True):
                        print(n)
                    for e in v.edges(data=True):
                        print(e)
                elif isinstance(v, list) and isinstance(v[0], nx.DiGraph):
                    for v_ in v:
                        for n in v_.nodes(data=True):
                            print(n)
                        for e in v_.edges(data=True):
                            print(e)
                        print()
                else:
                    print(v)
            print()

    return data

if __name__ == "__main__":
    fnames = [os.path.join(dirname, fname)
              for fname in os.listdir(dirname) if fname_filter in fname]
    fnames = [fname for fname in fnames if os.path.isfile(fname)]
    n_jobs = min(len(fnames), 80)
    filedatas = Parallel(n_jobs=n_jobs)(delayed(post_process_file)(fname) for fname in fnames)

    data = {}
    # list of dict to dict of concatenated list
    for filedata in filedatas:
        for key in filedata:
            if key not in data:
                data[key] = []
            data[key].extend(filedata[key])

    if "pfl" in data_to_prepare:
        with open(fname_filter + "_pfl", "wb") as f:
            print("pfl data points", len(data["feasibility_likelihood"]))
            pickle.dump({"state_graphs": data["pfl_state_graphs"],
                         "obj_infos": data["pfl_obj_infos"],
                         "feasibility_likelihood": data["feasibility_likelihood"]},
                        f)

    if "pfl_test" in data_to_prepare:
        with open(fname_filter + "_pfl_test", "wb") as f:
            print("pfl test data points", len(data["pfl_test_state_trajs"]))
            pickle.dump({"state_graphs": data["pfl_test_state_trajs"],
                         "obj_infos": data["pfl_test_obj_infos"],
                         "backjump_label": data["pfl_test_labels"]},
                        f)

    if "imit" in data_to_prepare:
        with open(fname_filter + "_imit", "wb") as f:
            print("imitation data points", len(data["imitation_label"]))
            pickle.dump({"state_graphs": data["imit_state_graphs"],
                         "obj_infos": data["imit_obj_infos"],
                         "imitation_label": data["imitation_label"]},
                        f)

    print("done!")
