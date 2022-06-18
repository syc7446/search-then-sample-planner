"""Srivastava-style sampling-based TAMP planner.
"""

import time
import heapq as hq
import numpy as np
from pddlgym.utils import get_object_combinations
from pddlgym.structs import ground_literal, LiteralConjunction
from ndr.ndrs import NOISE_OUTCOME
from search_then_sample.utils.structs import Operator
import search_then_sample.utils.planner_heuristics as planner_heuristics
from search_then_sample.utils.env_base import EnvironmentFailure
from search_then_sample.utils.utils import compute_static_preds, compute_delete_relax_reachable_lits
import search_then_sample.utils.constants as constants


class BacktrackBatchSamplingPlanner:
    """Definition of planner.
    """
    def __init__(self, seed, timeout, heuristic_name, num_samples_per_step, num_resamples, learner_name=None):
        self._seed = seed
        self._timeout = timeout  # in seconds
        self._heuristic_name = heuristic_name  # from planner_heuristics.py
        self._num_samples_per_step = num_samples_per_step
        self._num_resamples = num_resamples
        self._num_calls = 0
        self._ground_operators = None  # cache for planning
        self._count_motion_prob_solving = 0

    def plan(self, env, state, all_ndrs, save_data, saved_world):
        """Return a plan given an env, low-level state, and NDR dictionary.
        """
        # Parse the state to get the set of discrete objects.
        lits = env.parse_state(state)
        discrete_objects = sorted({o for o in state if not o.is_continuous})
        self._init_saved_world = saved_world
        # Cache the operators ground with the discrete objects
        # These will have placeholders for continuous objects that will be
        # filled dynamically during search (to add a timestep index)
        self._ground_operators, lifted_operators = self._precompute_operators(
            all_ndrs, discrete_objects)
        static_preds = compute_static_preds(self._ground_operators,
                                            env.discrete_predicates)
        static_facts = {lit for lit in lits if lit.predicate in static_preds}
        self._ground_operators = [op for op in self._ground_operators
                                  if self._static_facts_satisfied(
                                      op, static_preds, static_facts)]
        self._ground_operators = [op for op in self._ground_operators
                                  if self._continuous_args_constrained(op)]
        # If the goals are not reachable, fail early
        dr_reachable_lits = compute_delete_relax_reachable_lits(
            lits, self._ground_operators)
        goal_literals = set(env.literal_goal)
        if not goal_literals.issubset(dr_reachable_lits):
            raise PlanningExhausted(
                f"Goals {goal_literals - dr_reachable_lits} not reachable")
        heuristic = getattr(planner_heuristics, self._heuristic_name)(
            env, lifted_operators, discrete_objects)

        return self._find_plan(env, state, lits, heuristic, save_data)

    def _find_plan(self, env, state, lits, heuristic, save_data):
        # Do search over skeletons.
        start_time = time.time()
        queue = []
        visited = {(frozenset(lits), tuple())}
        root_node = Node(lits=lits, skeleton=[], constraints=[],
                         lits_sequence=[lits])
        rng_prio = np.random.RandomState(self._seed+self._num_calls)
        rng_sampler = np.random.RandomState(self._seed+self._num_calls)
        self._num_calls += 1
        hq.heappush(queue, (heuristic(root_node),
                            rng_prio.uniform(),
                            root_node))
        num_expanded = 0
        num_sampling = 0
        while queue and (time.time()-start_time < self._timeout):
            _, _, node = hq.heappop(queue)
            # Good debug point #1: print node.skeleton and node.constraints
            # here to see what the high-level search is doing.
            num_expanded += 1
            # If this skeleton satisfies the high-level goal, run sampling.
            if env.literal_goal.issubset(node.lits):
                num_sampling += 1
                assert node.lits == node.lits_sequence[-1]
                plan, save_data = self._sample_continuous_values(
                    env, state, node.skeleton, node.constraints,
                    node.lits_sequence, rng_sampler, start_time, save_data)
                if plan is not None:
                    print("Success! expanded {} skeletons (sampled for {}), "
                          "found plan of length {}: {}".format(
                              num_expanded, num_sampling, len(plan), plan))
                    print('Total number of motion problems solved', self._count_motion_prob_solving)
                    return plan, env.get_save_path(), save_data, self._count_motion_prob_solving
                else:
                    print('Symbolic plan failed so move onto the next one.')
            else:
                # Generate successors.
                for child_node in self._get_successors(node):
                    visited_key = (frozenset(child_node.lits),
                                   tuple(map(
                                       frozenset, child_node.constraints)))
                    if visited_key in visited:
                        continue
                    visited.add(visited_key)
                    # Do A*.
                    priority = len(child_node.skeleton)+heuristic(child_node)
                    hq.heappush(queue, (priority,
                                        rng_prio.uniform(),
                                        child_node))
        if not queue:
            raise PlanningExhausted("Ran out of skeletons!")
        else:
            assert time.time()-start_time > self._timeout
            raise PlanningTimeout("Timed out!")

    def _sample_continuous_values(self, env, start_state, skeleton,
                                  constraints, lits_sequence, rng, start_time, save_data):
        """Backtracking search over continuous values.
        """
        for nr in range(self._num_resamples):
            save_sampled_config = [[None for _ in range(self._num_samples_per_step)] for _ in range(len(skeleton))]
            assert len(skeleton) == len(constraints)
            num_sample_tries = 0
            cur_idx = 0
            num_tries = [0 for _ in skeleton]
            saved_worlds = [None for _ in skeleton]
            idx_to_max_num_tries = [self._num_samples_per_step \
                if any(v.is_continuous for v in a.variables) \
                else 1 for a in skeleton]
            plan = [None for _ in skeleton]
            traj = [start_state]+[None for _ in skeleton]
            while cur_idx < len(skeleton):
                if time.time()-start_time > self._timeout:
                    raise PlanningTimeout("Timed out!")
                assert num_tries[cur_idx] < idx_to_max_num_tries[cur_idx]
                print("Planner at {} step: num trials {}".format(cur_idx, num_tries))
                # Good debug point #2: if you have a skeleton that you think is
                # reasonable, but sampling isn't working, print num_tries here to
                # see at what step the backtracking search is getting stuck.
                num_tries[cur_idx] += 1
                state = traj[cur_idx]
                skel_act = skeleton[cur_idx]
                constr_set = constraints[cur_idx]
                if cur_idx > 0:
                    act_args, saved_world, sampled_config = self._sample_action_args(env, state, skel_act,
                                                                                     constr_set, rng, saved_worlds[cur_idx - 1],
                                                                                     save_sampled_config[cur_idx][num_tries[cur_idx] - 1])
                else:
                    act_args, saved_world, sampled_config = self._sample_action_args(env, state, skel_act,
                                                                                     constr_set, rng, self._init_saved_world,
                                                                                     save_sampled_config[cur_idx][num_tries[cur_idx] - 1])
                if not save_sampled_config[cur_idx][num_tries[cur_idx] - 1]:
                    save_sampled_config[cur_idx][num_tries[cur_idx] - 1] = sampled_config
                self._count_motion_prob_solving += 1
                save_data.add_init(sym_actions=skel_act.predicate.__str__(), configs=sampled_config, steps=cur_idx)
                num_sample_tries += 1
                if act_args: # Motion level is feasible
                    saved_worlds[cur_idx] = saved_world
                    ground_act = skel_act.predicate(*act_args)
                    plan[cur_idx] = ground_act
                    try:
                        traj[cur_idx+1], _, _, save_data = env.simulate(state, ground_act, save_data)
                    except EnvironmentFailure as e:
                        print(f'WARNING: env failure in planning: {e.args[0]}')
                        traj[cur_idx+1] = state
                        save_data.add_rest(base_states=None, arm_states=None, obj_states=None,
                                           hand_hold=None, feasibilities=False)
                    cur_idx += 1
                    # Check literal sequence constraint. Backtrack if failed.
                    assert len(traj) == len(lits_sequence)
                    lits = env.parse_state(traj[cur_idx])
                    if lits == lits_sequence[cur_idx]:
                        if cur_idx == len(skeleton):  # success!
                            print(f'Total number of motion planning tries: {num_sample_tries}')
                            return plan, save_data
                        continue  # all good, no need to backtrack
                    cur_idx -= 1
                else:
                    save_data.add_rest(base_states=None, arm_states=None, obj_states=None,
                                       hand_hold=None, feasibilities=False)

                # Do backtracking
                while num_tries[cur_idx] == idx_to_max_num_tries[cur_idx]:
                    num_tries[cur_idx] = 0
                    traj[cur_idx] = None
                    if cur_idx > 0:
                        saved_worlds[cur_idx - 1] = None
                        plan[cur_idx - 1] = None
                    cur_idx -= 1
                    if cur_idx < 0:
                        if nr == self._num_resamples - 1:
                            return None, save_data  # backtracking exhausted
                    else:
                        env.save_path.delete()
            # Should only get here if the skeleton was empty
            assert not skeleton
            print(f'Total number of motion planning tries: {num_sample_tries}')
            return plan, save_data

    @staticmethod
    def _sample_action_args(env, state, skel_act, constr_set, rng, pre_saved_world, save_sampled_config):
        params_to_samples = {}
        for constr in constr_set:
            # Get & evaluate the sampler for this constraint.
            func = getattr(env, "sample_"+constr.predicate.name)
            sampler_args = [state]
            # All samplers are conditioned on their discrete variables.
            for var in constr.variables:
                if not var.is_continuous:
                    sampler_args.append(var)
            sampler_args.append(rng)
            sampler_args.append(pre_saved_world)
            sampler_args.append(save_sampled_config)
            r_sampler_args = func(*sampler_args)
            if isinstance(r_sampler_args, dict): # Sampling succeeded
                for ind, sample in r_sampler_args.items():
                    if ind == 'saved_world':
                        cur_saved_world = sample
                        continue
                    if ind == 'config':
                        sampled_config = sample
                        continue
                    param = constr.variables[ind]
                    assert param.is_continuous
                    if param in params_to_samples:
                        raise Exception("Multiple samplers for one parameter?!")
                    params_to_samples[param] = sample
            else: # Sampling failed
                return [], None, r_sampler_args
        # Construct arguments for ground action.
        act_args = []
        for var in skel_act.variables:
            if var.is_continuous:
                assert var in params_to_samples, "Missing a sampler?"
                act_args.append(var.var_type(
                    f"optresult_{var.name}", params_to_samples[var]))
            else:
                if var in params_to_samples:
                    raise Exception(f"Unexpected sample for argument {var}")
                act_args.append(var)
        return act_args, cur_saved_world, sampled_config

    @staticmethod
    def _static_facts_satisfied(op, static_preds, static_facts):
        """Return whether the given GROUND operator could possibly satisfy
        the given static facts.
        """
        for pre in op.preconds.literals:
            assert not pre.is_negative
            if pre.predicate in static_preds and pre not in static_facts:
                return False
        return True

    @staticmethod
    def _continuous_args_constrained(op):
        """Check whether all continuous action arguments are constrained
        by at least one precondition
        """
        unconstrained_args = {v for v in op.action.variables \
                              if v.is_continuous}
        for lit in op.preconds.literals:
            unconstrained_args -= set(lit.variables)
        return len(unconstrained_args) == 0

    def _get_successors(self, node):
        # Check whether each ground operator has satisfied the
        # DISCRETE preconditions.
        timestep = len(node.skeleton)
        for operator in self._get_applicable_operators(node.lits, timestep):
            child_lits = self._apply_operator(operator, node.lits)
            child_skeleton = node.skeleton+[operator.action]
            # Add the CONTINUOUS preconditions as constraints, to be
            # dealt with by the optimizer.
            child_constraints = set()
            for lit in operator.preconds.literals:
                if lit == operator.action:
                    continue
                if any(o.is_continuous for o in lit.variables):
                    child_constraints.add(lit)
            yield Node(lits=child_lits, skeleton=child_skeleton,
                       constraints=node.constraints+[child_constraints],
                       lits_sequence=node.lits_sequence+[child_lits],
                       parent=node)

    def _get_applicable_operators(self, lits, timestep):
        for operator in self._ground_operators:
            # Create a copy of the operator so that we can make it
            # timestep-specific, i.e., add time indices to all pose variables.
            sigma = {o: o if not o.is_continuous
                        else o.var_type(f"{o.name}_t{timestep}")
                     for o in operator.params}
            operator = self._create_ground_operator(operator, sigma)
            applicable = True
            for lit in operator.preconds.literals:
                # Ignore all preconditions involving continuous arguments
                if any(o.is_continuous for o in lit.variables):
                    continue
                # Ignore action precondition
                if lit == operator.action:
                    continue
                # Check if lit is in the given lits
                if lit not in lits:
                    applicable = False
                    break
            if applicable:
                yield operator

    @staticmethod
    def _apply_operator(operator, lits):
        add_effects, delete_effects = set(), set()
        for effect in operator.effects.literals:
            assert not any(o.is_continuous for o in effect.variables)
            if effect.is_anti:
                delete_effects.add(effect.inverted_anti)
            else:
                add_effects.add(effect)
        new_lits = lits.copy()
        new_lits -= delete_effects
        new_lits |= add_effects
        return new_lits

    def _precompute_operators(self, guidance, discrete_objects):
        ground_operators = []
        lifted_operators = []
        for action in sorted(guidance):
            ndr_set = guidance[action]
            # Determinize the NDRs into operators
            cnt = 0
            for ndr in ndr_set:
                operator = ndr.determinize(name_suffix=cnt)
                cnt += 1
                operator.action = action
                # We'll never want to use an operator with noisy effects
                if NOISE_OUTCOME in operator.effects.literals:
                    continue
                ground_operators += self._precompute_single_ground_operators(
                    operator, discrete_objects)
                lifted_operators.append(operator)
        return ground_operators, lifted_operators

    def _precompute_single_ground_operators(self, operator, discrete_objects):
        params = operator.params
        objects = discrete_objects+[p for p in params if p.is_continuous]
        choices = list(get_object_combinations(
            objects, len(params), var_types=[p.var_type for p in params],
            allow_duplicates=True))
        for ground_params in choices:
            sigma = dict(zip(params, ground_params))
            ground_operator = self._create_ground_operator(operator, sigma)
            # If the ground operator has non-unique action arguments, do not
            # consider (this is explicitly disallowed)
            action_args = ground_operator.action.variables
            if len(action_args) != len(set(action_args)):
                continue
            yield ground_operator

    @staticmethod
    def _create_ground_operator(operator, sigma):
        """Substitute the params of the operator.
        """
        assert set(operator.params) == set(sigma)
        ground_params = [sigma[p] for p in operator.params]
        ground_action = ground_literal(operator.action, sigma)
        ground_preconds = LiteralConjunction([ground_literal(lit, sigma) \
            for lit in operator.preconds.literals])
        ground_effects = LiteralConjunction([ground_literal(lit, sigma) \
            for lit in operator.effects.literals])
        ground_operator = Operator(ground_action, operator.name, ground_params,
                                   ground_preconds, ground_effects)
        return ground_operator


class Node:
    """A node for the search over skeletons.
    """
    def __init__(self, lits, skeleton, constraints, lits_sequence, parent=None):
        self.lits = lits
        self.skeleton = skeleton
        self.constraints = constraints
        self.lits_sequence = lits_sequence
        self.parent = parent
        self.continuous_values = None  # can set this after optimization


class PlanningExhausted(Exception):
    """Exception raised when planning runs out of things to try.
    """
    pass


class PlanningTimeout(Exception):
    """Exception raised when planning times out.
    """
    pass
