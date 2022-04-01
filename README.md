# Search-Then-Sample TAMP Planner

## Usage
Install dependencies listed below and run `python main.py`. Expected output is that it should run 10 planning problems on the blocks environment and succeed on all. Each call to planning should take 3-10 seconds.

If you want to watch it go, you can turn on `USE_VIEWER` in `constants.py`. You can also change `env_name` to the bins environment in `main.py`.

## Notes
* The TAMP planner is defined in `planner.py` and uses heuristics defined in `planner_heuristics.py`. This planner is built off [Srivastava 2014](https://people.eecs.berkeley.edu/~russell/papers/icra14-planrob.pdf). It uses a high-level A* search over plan skeletons, informed by a domain-independent classical heuristic such as hadd. For any skeleton which reaches the goal at the symbolic level, a backtracking search is conducted over calls to the environment-defined samplers for values of the continuous action parameters in the skeleton.
* Each sampler is associated with a continuous predicate in the environment, and is a function of the state and the values of that predicate's discrete parameters.
* Two environments are provided, blocks and bins. Blocks is a pybullet version of the standard STRIPS blocks domain. Blocks never requires backtracking since each sampler is deterministic, so there is no point in invoking a sampler multiple times with the same arguments. In bins, the robot must place objects into the shelf and box, but to place into the shelf it must be side-grasping an object, and to place into the box it must be top-grasping an object. The grasp sampler given to the robot randomly produces either top-grasps or side-grasps, so backtracking is almost certainly needed for planning.
* The bins environment is fickle due to noise in the simulator and so will not succeed on all planning problems. It should solve around 6-8 of the 10 problems.

## Dependencies
* [NDR](https://github.com/tomsilver/ndr)
* [pyperplan](https://github.com/aibasel/pyperplan/)
* [pddlgym](https://github.com/tomsilver/pddlgym/)
* [pybullet](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit)
