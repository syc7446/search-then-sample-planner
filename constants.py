"""Constants used throughout the code.
"""

USE_VIEWER = True

PLANNER_TIMEOUT = 20
if USE_VIEWER:
    PLANNER_TIMEOUT = 999999

BLOCKS_BLOCK_HEIGHT = 0.075
BINS_OBJ_HEIGHT = 0.13
BINS_OBJ_RADIUS = 0.03
BINS_ROBOT_X = 0.8
BINS_ROBOT_Z = -0.25
BINS_SHELF_L = 0.8  # length
BINS_SHELF_D = 0.15  # depth
BINS_SHELF_H = 0.18  # height
BINS_SHELF_SLABS = 2  # number of slabs
BINS_BOX_Y = 0.5  # y coordinate
BINS_BOX_S = 0.08  # side length
BINS_SHELF_BOX_T = 0.5  # transparency (0-1)
VEL_TOL = 0.02
GRASPING_TOL = 5e-3
