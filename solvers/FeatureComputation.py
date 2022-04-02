import math
import os
import numpy as np
import matplotlib.pyplot as plt

class Features():
    def __init__(self, plan_time, step_size):
        self.plan_time = plan_time
        self.step_size = step_size
        self.k = [3, 4, 5]

        # Feature-related parameters
        self.feature_time = np.zeros(plan_time)
        self.feature_convex_val = np.zeros(len(self.k))
        self.feature_convex = np.zeros((len(self.k), plan_time))
        self.feature_entire_convex_val = 0
        self.feature_entire_convex = np.zeros(plan_time)
        self.feature_smaller_step_size_val = 0
        self.feature_smaller_step_size = np.zeros(plan_time)
        self.feature_dist_bet_trees = np.zeros(plan_time)

    def add(self, total_trials, cur_tree, q_near, q_next, status):
        self.cur_tree = cur_tree
        self.q_near = q_near
        self.q_next = q_next
        self.status = status
        self.update_value(total_trials)

    def update_value(self, total_trials):
        if total_trials <= self.plan_time:
            # print('q_near: (%.2f, %.2f), q_next: (%.2f, %.2f), status: %s'
            #       % (self.q_near[0], self.q_near[1], self.q_next[0], self.q_next[1], self.status))
            if self.status != 'sample collision' and self.status != 'path collision':
                # Convexity
                for i in range(len(self.k)):
                    CH = ConvexHull(self.q_next, self.cur_tree.kNearestNode(self.q_next, k=self.k[i]))
                    self.feature_convex_val[i] += CH.get_convexity()
                # Convexity of the entire tree
                CH = ConvexHull(self.q_next, self.cur_tree.kNearestNode(self.q_next, k=self.cur_tree.length+1))
                self.feature_entire_convex_val += CH.get_convexity()
                # Closer than step size
                if math.hypot(self.q_near[0]-self.q_next[0], self.q_near[1]-self.q_next[1]) < self.step_size:
                    self.feature_smaller_step_size_val += 1

            self.feature_time[total_trials-1] = total_trials / self.plan_time
            for i in range(len(self.k)): self.feature_convex[i, total_trials-1] = self.feature_convex_val[i]
            self.feature_entire_convex[total_trials-1] = self.feature_entire_convex_val
            self.feature_smaller_step_size[total_trials-1] = self.feature_smaller_step_size_val
            self.feature_dist_bet_trees[total_trials-1] = self.cur_tree.getDistBetTrees()   # getting normalized distance

    def set_solution(self, total_trials):
        self.solution_time = total_trials

    def save_value(self, save_at_file_name, save_policy_file_name):
        # Feature normalization
        self.feature_convex /= self.plan_time
        self.feature_entire_convex /= self.plan_time
        self.feature_smaller_step_size /= self.plan_time

        # Training data of learning a policy directly
        # Feature saving
        train_data_filepath = os.path.join(os.path.dirname(__file__), save_policy_file_name)
        with open(train_data_filepath, "a") as train_data_file:
            # Timestep
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_time[i]) + ':')
                else: train_data_file.write(str(self.feature_time[i]) + ',')
            # Convexity
            for i in range(len(self.k)):
                for j in range(self.plan_time):
                    if j == self.plan_time-1: train_data_file.write(str(self.feature_convex[i,j]) + ':')
                    else: train_data_file.write(str(self.feature_convex[i,j])+',')
            # Convexity of the entire tree
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_entire_convex[i]) + ':')
                else: train_data_file.write(str(self.feature_entire_convex[i]) + ',')
            # Closer than step size
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_smaller_step_size[i]) + ':')
                else: train_data_file.write(str(self.feature_smaller_step_size[i]) + ',')
            # Distance between two trees
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_dist_bet_trees[i]) + ';')
                else: train_data_file.write(str(self.feature_dist_bet_trees[i]) + ',')
            # Make sure for the last feature we must finish with ';' instead of ':'

            # Output
            for i in range(self.plan_time): # solution: 1 (continue), no solution: 0 (stop)
                if self.solution_time > self.plan_time: # no solution found within planning time
                    if i == self.plan_time-1: train_data_file.write(str(0) + '\n')
                    else: train_data_file.write(str(0) + ',')
                else:
                    if i == self.plan_time-1: train_data_file.write(str(1) + '\n')
                    else: train_data_file.write(str(1) + ',')

        # Training data of additional time to find a solution
        # Feature saving
        train_data_filepath = os.path.join(os.path.dirname(__file__), save_at_file_name)
        with open(train_data_filepath, "a") as train_data_file:
            # Timestep
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_time[i]) + ':')
                else: train_data_file.write(str(self.feature_time[i]) + ',')
            # Convexity
            for i in range(len(self.k)):
                for j in range(self.plan_time):
                    if j == self.plan_time-1: train_data_file.write(str(self.feature_convex[i, j]) + ':')
                    else: train_data_file.write(str(self.feature_convex[i, j]) + ',')
            # Convexity of the entire tree
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_entire_convex[i]) + ':')
                else: train_data_file.write(str(self.feature_entire_convex[i]) + ',')
            # Closer than step size
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_smaller_step_size[i]) + ':')
                else: train_data_file.write(str(self.feature_smaller_step_size[i]) + ',')
            # Distance between two trees
            for i in range(self.plan_time):
                if i == self.plan_time-1: train_data_file.write(str(self.feature_dist_bet_trees[i]) + ';')
                else: train_data_file.write(str(self.feature_dist_bet_trees[i]) + ',')
            # Make sure for the last feature we must finish with ';' instead of ':'

            # Output
            for i in range(self.plan_time):
                if self.solution_time > self.plan_time: # no solution found within planning time
                    if i == self.plan_time-1: train_data_file.write(str(self.plan_time-i) + '\n')
                    else: train_data_file.write(str(self.plan_time-i) + ',')
                else:
                    if self.solution_time-i > 0:
                        if i == self.plan_time-1: train_data_file.write(str(self.solution_time-i) + '\n')
                        else: train_data_file.write(str(self.solution_time-i) + ',')
                    else:
                        if i == self.plan_time-1: train_data_file.write(str(0) + '\n')
                        else: train_data_file.write(str(0) + ',')

def plot_feature_result(f_convex_list, f_dist_list, total_trials, inst):
    plt.figure(1)
    plt.savefig('./plot/final_'+str(inst)+'.png')
    plt.close(1)

    plt.figure(2)
    plt.plot(f_convex_list)
    plt.legend(['Cumulative number of convexity'])
    plt.xlabel('Steps')
    plt.ylabel('Count')
    plt.title('Convexity feature (solution found at '+str(total_trials)+' steps)')
    plt.savefig('./plot/convex_'+str(inst)+'.png')
    plt.close(2)

    plt.figure(3)
    plt.plot(f_dist_list)
    plt.legend(['Distance between q_next and q_near'])
    plt.xlabel('Steps')
    plt.ylabel('Distance')
    plt.title('Distance feature (solution found at '+str(total_trials)+' steps)')
    plt.savefig('./plot/distance_'+str(inst)+'.png')
    plt.close(3)

    with open('./plot/data_'+str(inst)+'.txt', "a") as f:
        for i in range(len(f_convex_list[:-1])):
            f.write(str(i)+",")
        f.write(str(len(f_convex_list) - 1)+"\n")
        for i in range(len(f_convex_list[:-1])):
            f.write(str(f_convex_list[i])+",")
        f.write(str(f_convex_list[-1])+"\n")
        for i in range(len(f_dist_list[:-1])):
            f.write(str(f_dist_list[i])+",")
        f.write(str(f_dist_list[-1])+"\n")

class ConvexHull:
    def __init__(self, new, nodes):
        self.convexity = None
        points = [list(nodes[i]) for i in range(len(nodes))]
        try:
            self.convexity = self._in_hull(new, points)
        except:
            print('Condition for convexity computation is not ready.')

    def _in_hull(self, p, hull):
        """
        Test if points in `p` are in `hull`

        `p` should be a `NxK` coordinates of `N` points in `K` dimensions
        `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the
        coordinates of `M` points in `K`dimensions for which Delaunay triangulation
        will be computed
        """
        from scipy.spatial import Delaunay
        if not isinstance(hull, Delaunay):
            hull = Delaunay(hull)

        return hull.find_simplex(p) >= 0

    def get_convexity(self):
        # print('convexity', self.convexity)
        if self.convexity: return 1
        else: return 0