"""Implementation of various controllers for PyBullet.
"""

import abc
import numpy as np
from pybullet_utils import get_move_action
import constants

ATOL = 1e-4


class Controller:
    """A generic controller implemented as a state machine.
    """
    def __init__(self, objects, names_to_predicates, world):
        # Objects is a bit of a misnomer, it's actually the arguments
        # to this controller.
        self._objects = objects
        self._names_to_predicates = names_to_predicates
        self._world = world
        self._stage = 0
        self._stages = self._get_stages()
        assert self._stages
        self._noop_action = self._get_noop_action()
        self._done = False
        self._timesteps_this_stage = 0

    def __repr__(self):
        return "{}({})".format(self.__class__.__name__, self._objects)

    def __str__(self):
        return repr(self)

    @abc.abstractmethod
    def _get_stages(self):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _get_noop_action(self):
        raise NotImplementedError("Override me!")

    @abc.abstractmethod
    def _terminate_early(self, objects, logical_state, object_state):
        raise NotImplementedError("Override me!")

    def step(self, logical_state, object_state):
        """Given the state, return (an action, done).
        The interpretation of done=True is that the returned action is the
        final one outputted by this controller.
        """
        assert not self._done, "Tried to call step on a done controller"
        if self._stage == 0 and self._terminate_early(
                self._objects, logical_state, object_state):
            self._done = True
            return self._noop_action, self._done
        action, stage_done = self._stages[self._stage](
            self._objects, logical_state, object_state)
        self._timesteps_this_stage += 1
        if stage_done or self._timesteps_this_stage > 100:
            self._timesteps_this_stage = 0
            self._stage += 1
        if self._stage >= len(self._stages):
            self._done = True
        return action, self._done


class BlocksPickupFromBlockController(Controller):
    """Controller for picking up a block that's on another block.
    """
    def _get_stages(self):
        return [
            self._move_to_above_target1,
            self._move_to_above_target2,
            self._open_grippers,
            self._move_down,
            # self._close_grippers,  # removed to prevent aggressive grasps
            self._close_grippers,
            self._bring_to_pick_position,
            self._move_up,
            self._move_up,
            self._move_up,
            self._move_up,
            self._move_up,
        ]

    def _get_noop_action(self):
        return np.zeros(4)

    def _terminate_early(self, objects, logical_state, object_state):
        clear = self._names_to_predicates["Clear"]
        holding = self._names_to_predicates["Holding"]
        on = self._names_to_predicates["On"]
        ontable = self._names_to_predicates["OnTable"]
        # Check that the object is clear
        block = objects[0]
        if clear(block) not in logical_state and \
           holding(block) not in logical_state:
            return True
        if ontable(block) in logical_state:
            return True
        if on(block, objects[1]) not in logical_state:
            return True
        # If we're holding something, terminate
        for lit in logical_state:
            if lit.predicate == holding:
                return True
        # If the block is below the table, terminate
        z = object_state[block]["pose"][2]
        if z < 0.1:
            return True
        return False

    def _move_to_above_target(self, objects, object_state, above_amount):
        target_position = objects[-3:]
        gripper_position = object_state[self._world]["gripper_position"]
        target_position[2] += above_amount
        if ((gripper_position[0]-target_position[0])**2+
                (gripper_position[1]-target_position[1])**2) < ATOL:
            return np.zeros(4), True
        return get_move_action(gripper_position, target_position), False

    def _move_to_above_target1(self, objects, _, object_state):
        return self._move_to_above_target(objects, object_state, 0.15)

    def _move_to_above_target2(self, objects, _, object_state):
        return self._move_to_above_target(objects, object_state, 0.05)

    @staticmethod
    def _open_grippers(_1, _2, _3):
        return np.array([0., 0., 0., 1.]), True

    def _move_down(self, objects, _, object_state):
        target_position = objects[-3:]
        gripper_position = object_state[self._world]["gripper_position"]
        done = ((gripper_position[2]-target_position[2])**2 < ATOL)
        return np.array([0., 0., -0.1, 0.]), done

    @staticmethod
    def _close_grippers(_1, _2, _3):
        return np.array([0., 0., 0., -0.25]), True

    def _bring_to_pick_position(self, _1, _2, object_state):
        pick_height = 0.5
        gripper_position = object_state[self._world]["gripper_position"]
        done = (gripper_position[2] > pick_height)
        return np.array([0., 0., 0.5, -0.005]), done

    @staticmethod
    def _move_up(_1, _2, _3):
        return np.array([0., 0., 0.5, -0.005]), True


class BlocksPickupFromTableController(BlocksPickupFromBlockController):
    """Controller for picking up a block that's on the table.
    """
    def _terminate_early(self, objects, logical_state, object_state):
        clear = self._names_to_predicates["Clear"]
        holding = self._names_to_predicates["Holding"]
        ontable = self._names_to_predicates["OnTable"]
        block = objects[0]
        if clear(block) not in logical_state and \
           holding(block) not in logical_state:
            return True
        if ontable(block) not in logical_state:
            return True
        # If we're holding something, terminate
        for lit in logical_state:
            if lit.predicate == holding:
                return True
        # If the block is below the table, terminate
        z = object_state[block]["pose"][2]
        if z < 0.1:
            return True
        return False


class BlocksPutOnBlockController(Controller):
    """Controller for putting a block on another.
    """
    def _get_stages(self):
        return [
            self._move_to_above_target1,
            self._move_to_above_target2,
            self._open_grippers,
            self._move_up,
            self._move_up,
            self._move_up,
            self._move_up,
            self._move_up,
        ]

    def _get_noop_action(self):
        return np.zeros(4)

    def _terminate_early(self, objects, logical_state, object_state):
        clear = self._names_to_predicates["Clear"]
        holding = self._names_to_predicates["Holding"]
        handfull = self._names_to_predicates["HandFull"]
        # Check that the block to place onto is clear
        block = objects[1]
        if clear(block) not in logical_state:
            return True
        if handfull() not in logical_state:
            return True
        if holding(objects[0]) not in logical_state:
            return True
        # If the block to place onto is below the table, terminate
        z = object_state[block]["pose"][2]
        if z < 0.1:
            return True
        return False

    def _move_to_target(self, objects, _, object_state, offset):
        target_position = objects[-3:]
        gripper_position = object_state[self._world]["gripper_position"]
        target_position[2] += offset
        if np.sum(np.subtract(target_position, gripper_position)**2) < ATOL:
            return np.array([0., 0., 0., -0.005]), True
        move_action = get_move_action(gripper_position, target_position)
        move_action = np.array(move_action)
        move_action[3] = -0.005
        return move_action, False

    def _move_to_above_target1(self, objects, _, object_state):
        return self._move_to_target(objects, _, object_state, offset=0.1)

    def _move_to_above_target2(self, objects, _, object_state):
        return self._move_to_target(objects, _, object_state, offset=0)

    @staticmethod
    def _open_grippers(_1, _2, _3):
        return np.array([0., 0., 0., 1.]), True

    @staticmethod
    def _move_up(_1, _2, _3):
        return np.array([0., 0., 1., 0.]), True


class BlocksPutOnTableController(BlocksPutOnBlockController):
    """Controller for putting a block on the table.
    """
    def _terminate_early(self, objects, logical_state, object_state):
        handfull = self._names_to_predicates["HandFull"]
        holding = self._names_to_predicates["Holding"]
        if handfull() not in logical_state:
            return True
        if holding(objects[0]) not in logical_state:
            return True
        return False

    def _move_to_above_target1(self, objects, _, object_state):
        return self._move_to_target(objects, _, object_state, offset=0.5)

    def _move_to_above_target2(self, objects, _, object_state):
        return self._move_to_target(objects, _, object_state, offset=0.15)


class BinsPickController(Controller):
    """Controller for picking up an object in the bins environment.
    Handles both top and side grasps.
    """
    def _get_stages(self):
        return [
            self._raise_up,
            self._move_to_target,
            self._open_grippers,
            self._move_in,
            self._close_grippers,
            self._raise_up,
        ]

    def _get_noop_action(self):
        return np.zeros(4)

    def _terminate_early(self, objects, logical_state, object_state):
        holding_side = self._names_to_predicates["HoldingSide"]
        holding_top = self._names_to_predicates["HoldingTop"]
        ontable = self._names_to_predicates["OnTable"]
        # Check that the object is on the table
        obj = objects[0]
        if ontable(obj) not in logical_state:
            return True
        # If we're holding something, terminate
        for lit in logical_state:
            if lit.predicate in (holding_side, holding_top):
                return True
        # If the object is below the table, terminate
        z = object_state[obj]["pose"][2]
        if z < 0.1:
            return True
        return False

    def _move_to_target(self, objects, _, object_state, offset=0):
        target_position = objects[-3:]
        gripper_position = object_state[self._world]["gripper_position"]
        if target_position[2] > 0.2+constants.BINS_OBJ_HEIGHT:  # top grasping
            target_position[2] += offset
            if offset != 0:
                target_position[2] += 0.04
        else:  # we're side grasping
            target_position[0] -= offset
        if ((gripper_position[0]-target_position[0])**2+
                (gripper_position[1]-target_position[1])**2+
                (gripper_position[2]-target_position[2])**2) < ATOL:
            return np.zeros(4), True
        return get_move_action(gripper_position, target_position), False

    def _move_in(self, objects, _, object_state):
        return self._move_to_target(objects, None, object_state, -0.15)

    @staticmethod
    def _open_grippers(_1, _2, _3):
        return np.array([0., 0., 0., 1.]), True

    @staticmethod
    def _close_grippers(_1, _2, _3):
        return np.array([0., 0., 0., -0.25]), True

    def _raise_up(self, _1, _2, object_state):
        pick_height = 0.4
        gripper_position = object_state[self._world]["gripper_position"]
        done = (gripper_position[2] > pick_height)
        return np.array([0., 0., 0.5, -0.005]), done


class BinsPlaceController(Controller):
    """Controller for placing an object in the bins environment.
    """
    def _get_stages(self):
        return [
            self._move_to_target,
            self._move_in,
            self._open_grippers,
            self._move_out,
            self._move_up,  # used by side place only
        ]

    def _get_noop_action(self):
        return np.zeros(4)

    def _terminate_early(self, objects, logical_state, object_state):
        holding_side = self._names_to_predicates["HoldingSide"]
        holding_top = self._names_to_predicates["HoldingTop"]
        handfull = self._names_to_predicates["HandFull"]
        istargetforobj = self._names_to_predicates["IsTargetForObj"]
        if istargetforobj(objects[0], objects[1]) not in logical_state:
            return True
        # Check that the object is being held
        if holding_side(objects[1]) not in logical_state and \
           holding_top(objects[1]) not in logical_state:
            return True
        if handfull() not in logical_state:
            return True
        return False

    def _move_to_target(self, objects, _, object_state, offset=0):
        target_position = objects[-3:]
        gripper_position = object_state[self._world]["gripper_position"]
        if target_position[1] < 1:  # top-place into the box
            target_position[2] += offset
            if offset < 0:
                target_position[2] += 0.06
            if offset > 0:
                target_position[0] -= 0.02
            tol = ATOL
        else:  # side-place into the shelf
            target_position[0] -= offset
            tol = ATOL*10  # finicky, increase tolerance
        if ((gripper_position[0]-target_position[0])**2+
                (gripper_position[1]-target_position[1])**2+
                (gripper_position[2]-target_position[2])**2) < tol:
            return np.zeros(4), True
        return get_move_action(gripper_position, target_position), False

    def _move_in(self, objects, _, object_state):
        return self._move_to_target(objects, None, object_state, -0.25)

    def _move_out(self, objects, _, object_state):
        return self._move_to_target(objects, None, object_state, 0.05)

    def _move_up(self, objects, _, object_state):
        target_position = objects[-3:]
        gripper_position = object_state[self._world]["gripper_position"]
        if target_position[1] < 1:  # do nothing for top-place into the box
            return np.zeros(4), True
        target_position[2] += 0.15
        tol = ATOL
        if ((gripper_position[0]-target_position[0])**2+
                (gripper_position[1]-target_position[1])**2+
                (gripper_position[2]-target_position[2])**2) < tol:
            return np.zeros(4), True
        return get_move_action(gripper_position, target_position), False

    @staticmethod
    def _open_grippers(_1, _2, _3):
        return np.array([0., 0., 0., 1.]), True


class TampnamoReachGoalController(Controller):
    """Controller for placing an object in the bins environment.
    """