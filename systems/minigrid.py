from gym_minigrid.minigrid import (MiniGridEnv, Grid, Lava, Floor,
                                   Ball, Key, Door, Goal, Wall, Box)
from gym_minigrid.minigrid import WorldObj as BaseWorldObj
from gym_minigrid.wrappers import (ReseedWrapper, FullyObsWrapper,
                                   ViewSizeWrapper)
from gym_minigrid.minigrid import (IDX_TO_COLOR, STATE_TO_IDX, COLORS, COLOR_TO_IDX,
                                   DIR_TO_VEC)
from gym_minigrid.register import register
from gym_minigrid.rendering import (point_in_rect, point_in_circle,
                                    point_in_triangle, rotate_fn,
                                    fill_coords, highlight_img, downsample)

import matplotlib.pyplot as plt
import gym
import math
import numpy as np
import re
import queue
import warnings
import itertools
from collections import defaultdict
from bidict import bidict
from typing import Type, List, Tuple, Any, Dict, Union, Set
from gym import wrappers, spaces
from gym.wrappers import Monitor
from gym.wrappers.monitor import disable_videos
from enum import IntEnum, Enum

import warnings
import matplotlib.cbook
warnings.filterwarnings("ignore",category=matplotlib.cbook.mplDeprecation)

# define these type defs for method annotation type hints
EnvObs = np.ndarray
CellObs = Tuple[int, int, int]
ActionsEnum = MiniGridEnv.Actions
Reward = float
Done = bool
StepData = Tuple[EnvObs, Reward, Done, dict]
Agent = Any
AgentPos = Tuple[int, int]
AgentDir = int
MultiAgentAction = List[ActionsEnum]
MultiAgentState = List[Tuple[AgentPos, AgentDir]]
MultiStepMultiAgentAction = Tuple[MultiAgentAction, ...]
MultiAgentActions = List[MultiAgentAction]
MultiAgentStates = List[MultiAgentState]
EnvType = Type[MiniGridEnv]
MultiAgentEnvType = Any #Type[MultiAgentMiniGridEnv]
Minigrid_TSNode = Tuple[AgentPos, AgentDir]
Minigrid_TSEdge = Tuple[Minigrid_TSNode, Minigrid_TSNode]
Minigrid_Edge_Unpacked = Tuple[Minigrid_TSNode, Minigrid_TSNode, AgentPos,
                               AgentDir, AgentPos, AgentDir, str, str]

MINIGRID_TO_GRAPHVIZ_COLOR = {'red': 'firebrick',
                              'green': 'darkseagreen1',
                              'blue': 'steelblue1',
                              'purple': 'mediumpurple1',
                              'yellow': 'yellow',
                              'grey': 'gray60'}

# Map of object type to integers
OBJECT_TO_IDX = {
    'unseen': 0,
    'empty': 1,
    'wall': 2,
    'floor': 3,
    'door': 4,
    'key': 5,
    'ball': 6,
    'box': 7,
    'goal': 8,
    'lava': 9,
    'agent': 10,
    'carpet': 11,
    'water': 12,
}
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))
IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))

GYM_MONITOR_LOG_DIR_NAME = 'minigrid_env_logs'

WHITE = 230
TILE_PIXELS = 128

FRANKA_ACTION_ABR = {
    "Ready": "Rd",
    "Transit": "Ts",
    "Grasp": "G",
    "Transfer": "Tf",
    "Release": "Rl",
    "Intervene": "I",
}
ABR_TO_FRANKA_ACTION = dict(zip(FRANKA_ACTION_ABR.values(), FRANKA_ACTION_ABR.keys()))


def custom_enum(typename, items_dict):
    class_definition = """\nfrom enum import IntEnum\n\nclass {}(IntEnum):\n\t{}"""\
        .format(typename, '\n\t'.join(['{} = {}'.format(k, v) for k, v in items_dict.items()]))
    namespace = dict(__name__='enum_%s' % typename)
    exec(class_definition, namespace)
    result = namespace[typename]
    result._source = class_definition

    return result


class BooleanAction(IntEnum):
    off = 0
    on = 1


class BooleanState(IntEnum):
    off = 0
    on = 1


class WorldObj(BaseWorldObj):
    """    Base class for grid world objects
    """

    def __init__(self, type, color):
        assert type in OBJECT_TO_IDX, type
        assert color in COLOR_TO_IDX, color
        self.type = type
        self.color = color
        self.contains = None

        # Initial position of the object
        self.init_pos = None

        # Current position of the object
        self.cur_pos = None

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @staticmethod
    def decode(type_idx, color_idx, state):
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == 'empty' or obj_type == 'unseen':
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == 'wall':
            v = Wall(color)
        elif obj_type == 'floor':
            v = Floor(color)
        elif obj_type == 'ball':
            v = Ball(color)
        elif obj_type == 'key':
            v = Key(color)
        elif obj_type == 'box':
            v = Box(color)
        elif obj_type == 'door':
            v = Door(color, is_open, is_locked)
        elif obj_type == 'goal':
            v = Goal()
        elif obj_type == 'lava':
            v = Lava()
        elif obj_type == 'carpet':
            v = Carpet()
        elif obj_type == 'water':
            v = Water()
        else:
            assert False, "unknown object type in decode '%s'" % obj_type

        return v


class Carpet(WorldObj):
    """
    Yellow carpet (floor) tile the agent can walk over
    """

    def __init__(self):
        super().__init__('carpet', color='yellow')

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        # color = COLORS[self.color] / 2
        color = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Water(WorldObj):
    """
    A floor tile with water on it that the agent can walk over
    """

    def __init__(self):
        super().__init__('water', color='blue')

    def can_overlap(self):
        return True

    def render(self, img):
        # Give the floor a pale color
        # color = COLORS[self.color] / 3
        color = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)


class Agent(WorldObj):

    DIR_TO_STRING = bidict({0: 'right', 1: 'down', 2: 'left', 3: 'up'})

    def __init__(self, name: str, type: str='agent', color='red', view_size: int=7,
                 actions_type: str = 'simple_static'):
        super(Agent, self).__init__(type, color)
        self.name = name
        self.view_size = view_size

        self.pos: List[float] = None
        self.dir: List[float] = None
        self.carrying: WorldObj = None

        self.set_actions_type(actions_type)

    def set_actions_type(self, actions_type: str):
        self._allowed_actions_types = set(['static', 'simple_static',
                                           'diag_static', 'default'])
        if actions_type not in self._allowed_actions_types:
            msg = f'actions_type ({actions_type}) must be one of: ' + \
                  f'{actions_type}'
            raise ValueError(msg)

        if actions_type == 'simple_static' or actions_type == 'diag_static':
            self.directionless_agent = True
        elif actions_type == 'static' or actions_type == 'default':
            self.directionless_agent = False

        self._actions_type = actions_type

        if actions_type == 'static':
            actions = ModifyActionsWrapper.StaticActions
            step_function = self._step_default
        elif actions_type == 'simple_static':
            actions = ModifyActionsWrapper.SimpleStaticActions
            step_function = self._step_simple_static
        elif actions_type == 'diag_static':
            actions = ModifyActionsWrapper.DiagStaticActions
            step_function = self._step_diag_static
        elif actions_type == 'default':
            actions = MiniGridEnv.Actions
            step_function = self._step_default

        self.actions = actions
        num_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(num_actions)
        self._step = step_function

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action \
                                   for action in self.actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

    def step(self, action: ActionsEnum, grid: Grid) -> Tuple[Done, Reward]:
        if isinstance(action, str):
            action = self.ACTION_STR_TO_ENUM[action]
        return self._step(action, grid)

    def _step_default(self, action: IntEnum, grid: Grid) -> Tuple[Done, Reward]:
        """
        """
        reward = 0
        done = False

        if action is None or np.isnan(action):
            return reward, done

        # Get the position in front of the agent
        fwd_pos = self.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = grid.get(*fwd_pos)

        # Rotate left
        if action == self.actions.left:
            self.dir -= 1
            if self.dir < 0:
                self.dir += 4

        # Rotate right
        elif action == self.actions.right:
            self.dir = (self.dir + 1) % 4

        # Move forward
        elif action == self.actions.forward:
            if fwd_cell == None or fwd_cell.can_overlap():
                grid.update(self.pos, fwd_pos, self)
                self.pos = fwd_pos
            if fwd_cell != None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell != None and fwd_cell.type == 'lava':
                done = True

        else:
            assert False, "unknown action"

        return reward + 1, done

    def _step_diag_static(self, action: IntEnum, grid: Grid) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        if action is None or np.isnan(action):
            return reward, done

        start_pos = self.pos

        # a diagonal action is really just two simple actions :)
        pos_delta = ModifyActionsWrapper.DIAG_ACTION_TO_POS_DELTA[action]

        # Get the contents of the new cell of the agent
        new_pos = tuple(np.add(start_pos, pos_delta))
        new_cell = grid.get(*new_pos)

        if new_cell is None or new_cell.can_overlap():
            grid.update(self.pos, new_pos, self)
            self.pos = new_pos
        if new_cell is not None and new_cell.type == 'goal':
            done = True
            reward = self._reward()
        if new_cell is not None and new_cell.type == 'lava':
            done = True

        return reward + 1, done

    def _step_simple_static(self, action: IntEnum, grid: Grid) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        if action is None or np.isnan(action):
            return reward, done

        # save the original direction so we can reset it after moving
        old_dir = self.dir
        new_dir = ModifyActionsWrapper.SIMPLE_ACTION_TO_DIR_IDX[action]
        self.dir = new_dir

        # Get the contents of the cell in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            grid.update(self.pos, fwd_pos, self)
            self.pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            done = True
            reward = self._reward()
        if fwd_cell is not None and fwd_cell.type == 'lava':
            done = True

        # reset the direction of the agent, as it really cannot change
        # direction
        self.dir = old_dir

        return reward + 1, done

    def _get_action_str(self, action_enum: ActionsEnum) -> str:
        """
        Gets a string representation of the action enum constant

        :param      action_enum:  The action enum constant to convert

        :returns:   The action enum's string representation
        """

        return self.actions._member_names_[action_enum]

    def can_overlap(self):
        return True

    def _reward(self):
        return 0

    def render(self, img):
        c = COLORS[self.color]

        if self.directionless_agent:

            cir_fn = point_in_circle(cx=0.5, cy=0.5, r=0.3)
            fill_coords(img, cir_fn, c)

        else:

            tri_fn = point_in_triangle(
                (0.12, 0.19),
                (0.87, 0.50),
                (0.12, 0.81),
            )
            # Rotate the agent based on its direction
            tri_fn = rotate_fn(tri_fn, cx=0.5, cy=0.5, theta=0.5 * math.pi * self.dir)
            fill_coords(img, tri_fn, c)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # If we want to encode more information other than self.dir,
        # we can use mapping (an attribute to a number) or assign a digit for each attribute
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], self.dir)

    def encode_as_str(self):
        obj_type, obj_color, obj_state = self.encode()
        return '_'.join([IDX_TO_OBJECT[obj_type],
                         IDX_TO_COLOR[obj_color],
                         self.DIR_TO_STRING[obj_state]])

    def __eq__(self, other):
        if not isinstance(other, Agent):
            return False

        if self.type == other.type and \
            self.color == other.color and \
                self.name == other.name:
                return True

        return False

    @property
    def dir_vec(self):
        """
        Get the direction vector for the agent, pointing in the direction
        of forward movement.
        """

        assert self.dir >= 0 and self.dir < 4
        return DIR_TO_VEC[self.dir]

    @property
    def right_vec(self):
        """
        Get the vector pointing to the right of the agent.
        """

        dx, dy = self.dir_vec
        return np.array((-dy, dx))

    @property
    def front_pos(self):
        """
        Get the position of the cell that is right in front of the agent
        """

        return self.pos + self.dir_vec

    @property
    def state(self) -> Tuple:
        return (tuple(self.pos), self.dir)

    def set_state(self, state, grid: Grid) -> None:
        position, direction = state

        if position is not None:
            grid.update(self.pos, position, self)
            self.pos = position

        if direction is not None:
            self.dir = direction

    def state_str(self, state=None) -> str:
        if state is None:
            state = self.state

        return (state[0], self.DIR_TO_STRING[state[1]])

    def get_view_coords(self, i, j):
        """
        Translate and rotate absolute grid coordinates (i, j) into the
        agent's partially observable view (sub-grid). Note that the resulting
        coordinates may be negative or outside of the agent's view size.
        """

        ax, ay = self.pos
        dx, dy = self.dir_vec
        rx, ry = self.right_vec

        # Compute the absolute coordinates of the top-left view corner
        sz = self.view_size
        hs = self.view_size // 2
        tx = ax + (dx * (sz - 1)) - (rx * hs)
        ty = ay + (dy * (sz - 1)) - (ry * hs)

        lx = i - tx
        ly = j - ty

        # Project the coordinates of the object relative to the top-left
        # corner onto the agent's own coordinate system
        vx = (rx * lx + ry * ly)
        vy = -(dx * lx + dy * ly)

        return vx, vy

    def get_view_exts(self):
        """
        Get the extents of the square set of tiles visible to the agent
        Note: the bottom extent indices are not included in the set
        """

        # Facing right
        if self.dir == 0:
            topX = self.pos[0]
            topY = self.pos[1] - self.view_size // 2
        # Facing down
        elif self.dir == 1:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1]
        # Facing left
        elif self.dir == 2:
            topX = self.pos[0] - self.view_size + 1
            topY = self.pos[1] - self.view_size // 2
        # Facing up
        elif self.dir == 3:
            topX = self.pos[0] - self.view_size // 2
            topY = self.pos[1] - self.view_size + 1
        else:
            assert False, "invalid agent direction"

        botX = topX + self.view_size
        botY = topY + self.view_size

        return (topX, topY, botX, botY)

    def relative_coords(self, x, y):
        """
        Check if a grid position belongs to the agent's field of view, and returns the corresponding coordinates
        """

        vx, vy = self.get_view_coords(x, y)

        if vx < 0 or vy < 0 or vx >= self.view_size or vy >= self.view_size:
            return None

        return vx, vy

    def in_view(self, x, y):
        """
        check if a grid position is visible to the agent
        """

        return self.relative_coords(x, y) is not None

    def str_to_state(self, state_str: str):
        state_str = list(state_str)
        state = state_str
        state[0] = state_str[0]
        state[1] = self.DIR_TO_STRING.inverse[state_str[1]]

        return tuple(state)


class ConstrainedAgent(Agent):

    Position = Tuple[int, int]
    def __init__(self,
        restricted_objs: List[WorldObj] = None,
        restricted_positions: List[Position] = None, **kwargs):
        super(ConstrainedAgent, self).__init__(**kwargs)

        self._restricted_objs = restricted_objs
        self._restricted_positions = restricted_positions

    def _step_simple_static(self, action: IntEnum, grid: Grid) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        if action is None or np.isnan(action):
            return reward, done

        # save the original direction so we can reset it after moving
        old_dir = self.dir
        new_dir = ModifyActionsWrapper.SIMPLE_ACTION_TO_DIR_IDX[action]
        self.dir = new_dir

        # Get the contents of the cell in front of the agent
        fwd_pos = self.front_pos
        fwd_cell = grid.get(*fwd_pos)

        # Don't move if the next state is in the restricted area
        if self._restricted_positions and tuple(fwd_pos) in self._restricted_positions:
            pass
        # Allow
        else:
            if fwd_cell is None or (fwd_cell.can_overlap() and self._is_safe(fwd_cell)):
                grid.update(self.pos, fwd_pos, self)
                self.pos = fwd_pos

            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = self._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        self.dir = old_dir

        return reward + 1, done

    def _is_safe(self, fwd_cell):
        if fwd_cell is None or self._restricted_objs is None:
            return True

        if not any([fwd_cell.type == o for o in self._restricted_objs]):
            return True

        return False


class BooleanAgent(Agent):

    def __init__(self, name: str, type: str = 'agent', colors=['grey', 'blue']):
        super().__init__(name, type, colors[0])
        self.name = name
        self.colors = colors
        self.actions = BooleanAction
        self.states = BooleanState
        self._state = self.states.off
        self._step = self._step_default
        num_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(num_actions)
        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action \
                                   for action in self.actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

    def _step_default(self, action: IntEnum, grid: Grid) -> Tuple[Done, Reward]:
        reward = 0
        done = False

        if action is None or np.isnan(action):
            return reward, done

        # Rotate left
        if action == self.actions.on:
            self._state = self.states.on
            self.color = self.colors[self._state]

        # Rotate right
        elif action == self.actions.off:
            self._state = self.states.off
            self.color = self.colors[self._state]

        else:
            assert False, "unknown action"

        return reward + 1, done

    def render(self, img):
        # Give the floor a pale color
        # color = COLORS[self.color] / 3
        color = COLORS[self.color]
        fill_coords(img, point_in_rect(0.031, 1, 0.031, 1), color)

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # If we want to encode more information other than self.dir,
        # we can use mapping (an attribute to a number) or assign a digit for each attribute
        return (OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], 0)

    @property
    def state(self):
        return int(self._state)

    def set_state(self, state, grid: Grid) -> None:
        self._state = state

    def state_str(self, state=None):
        if state is None:
            state = self.state

        return list(self.states)[state]


class FrankaWorldConfig:

    class Object:

        class ObjectState(IntEnum):
            Ready = 0
            Transit = 1
            Grasp = 2
            Transfer = 3
            Release = 4
            Intervene = 5

        def __init__(self,
            name: str,
            location: str,
            max_intervention_step: int,
            max_intervention_count: int = 1):
            """
            :arg max_intervention_step:     Maximum No. of steps the object can be intervened
                                            for one time
            :arg max_intervention_count:    Maximum No. of times human can intervene the object
            """
            self.name = name
            self.location = location
            self.max_intervention_step = max_intervention_step
            self.max_intervention_count = max_intervention_count
            self.num_intervention_step = 0
            self.num_intervention_count = 0
            self.state = FrankaWorldConfig.Object.ObjectState.Ready

            self.STATE_STR_TO_ENUM = {
                FrankaWorldConfig.Object.ObjectState._member_names_[s]: s \
                                    for s in list(FrankaWorldConfig.Object.ObjectState)}
            self.STATE_ENUM_TO_STR = dict(zip(self.STATE_STR_TO_ENUM.values(),
                                            self.STATE_STR_TO_ENUM.keys()))

        def is_intervened(self):
            return self.state == FrankaWorldConfig.Object.ObjectState.Intervene

        def can_intervene(self):
            if self.num_intervention_count != self.max_intervention_count and \
                (self.state == FrankaWorldConfig.Object.ObjectState.Ready or\
                self.state == FrankaWorldConfig.Object.ObjectState.Transit):
                return True
            return False

        def get_state(self):
            return self.state, self.location, self.num_intervention_step, self.num_intervention_count

        def state_str(self, state=None):
            if state is None:
                state = self.get_state()

            state = list(state)
            state[0] = FRANKA_ACTION_ABR[self.STATE_ENUM_TO_STR[state[0]]]
            state[2] = str(state[2])
            state[3] = str(state[3])

            return '.'.join(state)

        def str_to_state(self, state_str: str):
            state = state_str.split('.')
            state[0] = self.STATE_STR_TO_ENUM[ABR_TO_FRANKA_ACTION[state[0]]]
            state[2] = int(state[2])
            state[3] = int(state[3])

            return tuple(state)

        def set_state(self, state):
            state = list(state)
            self.state = state[0]
            self.location = state[1]
            self.num_intervention_step = state[2]
            self.num_intervention_count = state[3]

        def must_return(self):
            return self.num_intervention_step == self.max_intervention_step

        def increment(self):
            self.num_intervention_step += 1

        def reset_state(self):
            self.state = FrankaWorldConfig.Object.ObjectState.Ready
            self.num_intervention_count += 1

    def __init__(self,
        locations: List[str],
        object_locations: Dict[str, str],
        target_locations: List[str],
        max_intervention_step: int = 2,
        distance_mappings: Dict[str, Dict[str, float]] = None,
        init_player: str = 'sys',
        can_obj_be_in_same_location: bool = True,
        minimum_actions: bool = True,
        waiting_cost: float = 0):

        self.locations = locations
        self.objects = [FrankaWorldConfig.Object(o, l, max_intervention_step) \
            for o, l in object_locations.items()]
        self.target_locations = target_locations
        self.curr_player = init_player
        self.NAME_TO_IDX = {obj.name: i for i, obj in enumerate(self.objects)}
        self.distance_mappings = distance_mappings
        self.can_obj_be_in_same_location = can_obj_be_in_same_location
        self.minimum_actions = minimum_actions
        self.waiting_cost = waiting_cost

    def update(self, object) -> float:

        if object is not None:
            self.set_object(object)

        for o in self.objects:
            if o.is_intervened():
                if self.curr_player == 'sys':
                    o.increment()

            if self.curr_player == 'env' and \
                o.state == FrankaWorldConfig.Object.ObjectState.Release:
                o.state = FrankaWorldConfig.Object.ObjectState.Ready

        self.curr_player = 'env' if self.curr_player == 'sys' else 'sys'

    def set_object(self, object):
        idx = self.NAME_TO_IDX[object.name]
        self.objects[idx] = object

    def get_object(self, object_name):
        idx = self.NAME_TO_IDX[object_name]
        return self.objects[idx]

    def get_object_at(self, location, state):
        for object in self.objects:
            if object.location == location and object.state == state:
                return object
        return None

    def get_object_in_intervention(self):
        for o in self.objects:
            if o.state ==  FrankaWorldConfig.Object.ObjectState.Intervene:
                return o

        return None

    def get_state(self):
        return tuple(o.get_state() for o in self.objects), self.curr_player

    def state_str(self, state=None):
        if state is None:
            state = self.get_state()

        state = list(state)
        object_states = state[0]

        return tuple([o.state_str(s) for o, s in zip(self.objects, object_states)]), state[1]

    def str_to_state(self, state_str: str):
        state_str = list(state_str)
        state = state_str

        object_states = list(state[0])
        state[0] = tuple([o.str_to_state(s) for o, s in zip(self.objects, object_states)])

        return tuple(state)

    def set_state(self, state):
        state = list(state)
        self.curr_player= state[1]

        for o, s in zip(self.objects, state[0]):
            o.set_state(s)

    def encode(self):
        """
        Return locations of objects that are not in action
        (Grasp, Transfer, Release, Intervene)
        """
        observations = []
        for i, o in enumerate(self.objects):
            if o.state == FrankaWorldConfig.Object.ObjectState.Ready or \
                o.state == FrankaWorldConfig.Object.ObjectState.Transit:

                if self.target_locations is None:
                    observations.append(o.location)
                elif o.location == self.target_locations[i]:
                    observations.append(o.name)

        observations.sort()

        if len(observations) == 0:
            return 'lambda'

        return ''.join(observations)

    def get_cost(self, from_location: str, to_location: str):
        if self.distance_mappings is not None:
            if from_location in self.distance_mappings:
                if to_location in self.distance_mappings[from_location]:
                    return self.distance_mappings[from_location][to_location]

        return 0

    def can_intervene(self):
        """
        If all object states are "Ready", the human cannot intervene.
        In other words, at least one object must be in transit!
        """
        prohibited_obj_states = [FrankaWorldConfig.Object.ObjectState.Release,
                                 FrankaWorldConfig.Object.ObjectState.Intervene]
        if any([o.state in prohibited_obj_states for o in self.objects]):
            return False
        return True


class FrankaAgent(Agent):

    class FrankaState(IntEnum):
        Ready = 0
        Transit = 1
        Grasp = 2
        Transfer = 3
        Release = 4

    def __init__(self, world_config, name: str = 'franka', init_location: str = 'H'):
        super().__init__(name)

        self.states = FrankaAgent.FrankaState
        self.actions = self._construct_actions(world_config)
        self.location = init_location

        num_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(num_actions)

        self._step = self._step_default
        self._state = self.states.Ready

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action \
                                   for action in self.actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))
        self.STATE_STR_TO_ENUM = {
            FrankaAgent.FrankaState._member_names_[s]: s \
                                for s in list(FrankaAgent.FrankaState)}
        self.STATE_ENUM_TO_STR = dict(zip(self.STATE_STR_TO_ENUM.values(),
                                        self.STATE_STR_TO_ENUM.keys()))

    def _construct_actions(self, world_config):
        """
        transit_OBJ
        grasp (if an object is at the current location)
        transfer_TARGETLOCATION
        release (if object is in gripper)
        wait
        """
        count = 0
        actions = {}

        if world_config.minimum_actions:
            for object in world_config.objects:
                actions[f'transitGraspTo{object.name}'] = count
                count += 1

            for location in world_config.locations:
                actions[f'transferReleaseTo{location}'] = count
                count += 1

        else:

            for object in world_config.objects:
                actions[f'transitTo{object.name}'] = count
                count += 1

            for location in world_config.locations:
                actions[f'transferTo{location}'] = count
                count += 1

            actions['grasp'] = count + 1
            actions['release'] = count + 2

        actions['wait'] = count

        return custom_enum('FrankaAction', actions)

    def _step_default(self, action: IntEnum, world_config) -> Tuple[Done, Reward]:
        """
        1. TransitGrasp & Intervene:
                        Ready, (R, R, ..., R)
            Transit:    Ts, (Ts, R, ..., R)
            Intervene:  Ts, (I, R, ..., R)
        2. TransitGrasp & Intervened & TransitGrasp & Wait:
                        Ts, (I, R, ..., R)
            Transit:    Ts, (I, Ts, R, ..., R)
            Wait:       Ts, (I, Ts, R, ..., R)  Human cannot intervene while intervention
        3. TransitGrasp & Intervened & TransferRelease & Wait:
                        Ts, (I, Ts, R, ..., R)
            Transfer:   Rl, (I, Rl, R, ..., R)
            Wait:       R, (I, R, R, ..., R)
        4. TransitGrasp againt & Wait:
                        Ready, (I, R, R, ..., R)
            Transit:    Ts, (I, Ts, R, ..., R)
            Wait:       Ts, (I, Ts, R, ..., R)  Human cannot intervene while intervention
        5. TransitGrasp again & Return:
                        Ready, (I, R, R, ..., R)
            Transit:    Ts, (I, Ts, R, ..., R)
            Return:     Ts, (R, Ts, R, ..., R)  Human cannot intervene while intervention
        6. TransferRelease & Return:            Human cannot intervene while transfer
                        Ts, (I, Ts, R, ..., R)
            Transfer:   Rl, (I, Rl, R, ..., R)
            Return:     R, (R, R, R, ..., R)
        7. TransferRelease & Wait:              Human cannot intervene while transfer
                        Ts, (R, Ts, R, ..., R)
            Transfer:   Rl, (R, Rl, R, ..., R)
            Wait:       R, (R, R, R, ..., R)
        8. Wait until Return:
                        Ready/Ts, (I, R, ..., R)
            Return:     Ready/Ts, (R, R, ..., R)
        """

        cost = 0
        done = False

        if action is None or np.isnan(action):
            if self._state == FrankaAgent.FrankaState.Release:
                self._state = FrankaAgent.FrankaState.Ready
            return cost, False

        if world_config.curr_player != 'sys':
            return cost, True

        action_str = self.ACTION_ENUM_TO_STR[action]
        action_info = action_str.split('To')

        object = None
        next_location = None

        if action_info[0] in ['transit', 'transitGrasp']:

            # If the object is in intervention, it can never be moved
            object_name = action_info[1]
            object = world_config.get_object(object_name)
            if object.is_intervened():
                return cost, True

            # Franka must be in Ready or in Transit mode
            allowed_states = [FrankaAgent.FrankaState.Ready, FrankaAgent.FrankaState.Transit]
            if self._state not in allowed_states:
                return cost, True

            # Franka can transit to ANY objects (other than I) if Ready
            # So we only consider when Franka is in Transit.
            # If any object is in Transit mode, Franka cannot take action transit
            if self._state is FrankaAgent.FrankaState.Transit:
                object_states = [o.state for o in world_config.objects]
                if FrankaWorldConfig.Object.ObjectState.Transit in object_states:
                    return cost, True

            # # Both the robot and the object must be Ready to be picked up
            # # to prevent the robot from picking multiple objects at once
            # # (which is impossible)
            # if object.is_intervened() or \
            #     self._state not in allowed_states or \
            #     object.state != FrankaWorldConfig.Object.ObjectState.Ready:
            #     return cost, True

            # # Franka can transit to other obj, if prev the transition failed
            # if self._state is FrankaAgent.FrankaState.Transit and \
            #     world_config.get_object_in_intervention() is None:
            #     return cost, True

            # update robot's state and the object state
            next_location = object.location
            self._state = FrankaAgent.FrankaState.Transit
            object.state = FrankaWorldConfig.Object.ObjectState.Transit

        elif action_info[0] == "grasp":
            # Find the object at the current location
            object = world_config.get_object_at(self.location,
                FrankaWorldConfig.Object.ObjectState.Transit)

            if object is None or object.is_intervened() or \
                self._state != FrankaAgent.FrankaState.Transit or \
                object.state != FrankaWorldConfig.Object.ObjectState.Transit:
                return cost, True

            # self.location stays the same
            self._state = FrankaAgent.FrankaState.Grasp
            object.state = FrankaWorldConfig.Object.ObjectState.Grasp

        elif action_info[0] == "transfer":
            # Find the object at the current location
            object = world_config.get_object_at(self.location,
                FrankaWorldConfig.Object.ObjectState.Grasp)
            # If object in hand (Grasp)
            if object is None or object.is_intervened() or \
                self._state != FrankaAgent.FrankaState.Grasp or \
                object.state != FrankaWorldConfig.Object.ObjectState.Grasp:
                return cost, True

            # Check if there is no object at the destination
            if not world_config.can_obj_be_in_same_location:
                object_locations = [o.location for o in world_config.objects]
                if action_info[1] in object_locations and\
                    action_info[1] != object.location:
                    return cost, True

            next_location = action_info[1]
            object.location = action_info[1]
            self._state = FrankaAgent.FrankaState.Transfer
            object.state = FrankaWorldConfig.Object.ObjectState.Transfer

        elif action_info[0] == "release":
            # Find the object at the current location
            object = world_config.get_object_at(self.location,
                FrankaWorldConfig.Object.ObjectState.Transfer)
            # If object in hand (inTranfer)
            if object is None or object.is_intervened() or \
                self._state != FrankaAgent.FrankaState.Transfer or \
                object.state != FrankaWorldConfig.Object.ObjectState.Transfer:
                return cost, True

            # self.location stays the same
            self._state = FrankaAgent.FrankaState.Release
            object.state = FrankaWorldConfig.Object.ObjectState.Release

        elif action_info[0] == "wait":

            if self._state in [
                FrankaAgent.FrankaState.Grasp,
                FrankaAgent.FrankaState.Transfer,
                FrankaAgent.FrankaState.Release]:
                return cost, True

            # Robot can wait if the object was intervened while the robot was in transit
            obj = world_config.get_object_in_intervention()
            allowed_states = [FrankaAgent.FrankaState.Ready, FrankaAgent.FrankaState.Transit]
            if self._state in allowed_states and \
                obj is not None and \
                obj.location == self.location:
                # pass

                # Only allow the robot to wait if the LAST obj was intervened.
                if world_config.target_locations is None:
                    return cost, True

                in_same_location = [int(o.location == t) for o, t in \
                    zip(world_config.objects, world_config.target_locations)]
                notin_target_locations = sum(in_same_location) < len(world_config.objects) - 1

                if notin_target_locations:
                    return cost, True

                cost = world_config.waiting_cost
            else:
                return cost, True

        elif action_info[0] == "transferRelease":
            # Find the object at the current location
            object = world_config.get_object_at(self.location,
                FrankaWorldConfig.Object.ObjectState.Transit)
            # If object in hand (Grasp)
            if object is None or object.is_intervened() or \
                self._state != FrankaAgent.FrankaState.Transit or \
                object.state != FrankaWorldConfig.Object.ObjectState.Transit:
                return cost, True

            # Check if there is no object at the destination
            if not world_config.can_obj_be_in_same_location:
                object_locations = [o.location for o in world_config.objects]
                if action_info[1] in object_locations and\
                    action_info[1] != object.location:
                    return cost, True

            next_location = action_info[1]
            object.location = action_info[1]
            self._state = FrankaAgent.FrankaState.Release
            object.state = FrankaWorldConfig.Object.ObjectState.Release

        else:
            assert False, f"unknown action {action_str}"

        # Increment the intervention counts
        world_config.update(object) # increment the interventions

        cost = world_config.get_cost(self.location, next_location)
        if next_location is not None:
            self.location = next_location

        return cost, done

    def render(self, img):
        raise NotImplementedError('')

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # If we want to encode more information other than self.dir,
        # we can use mapping (an attribute to a number) or assign a digit for each attribute

        # TODO: Include world configuration
        return [OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], int(self.state)]

    @property
    def state(self):
        return self._state, self.location

    def set_state(self, state, world_config) -> None:
        self._state = state[0]
        self.location = state[1]

    def state_str(self, state=None):
        if state is None:
            state = self.state
        state = list(state)
        state[0] = FRANKA_ACTION_ABR[self.STATE_ENUM_TO_STR[state[0]]]

        return tuple(state)

    def str_to_state(self, state_str: str):
        state_str = list(state_str)
        state = state_str
        state[0] = self.STATE_STR_TO_ENUM[ABR_TO_FRANKA_ACTION[state_str[0]]]

        return tuple(state)


class HumanAgent(Agent):

    class HumanState(IntEnum):
        Ready = 0
        Intervene = 1

    def __init__(self, world_config, name: str = 'human', init_location: str = 'base'):
        super().__init__(name)

        self.states = HumanAgent.HumanState
        self.actions = self._construct_actions(world_config)

        num_actions = len(self.actions)
        self.action_space = gym.spaces.Discrete(num_actions)

        self._step = self._step_default
        self._state = self.states.Ready

        self.world_config = world_config
        self.world_state  = world_config.get_state()

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action \
                                   for action in self.actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))
        self.STATE_STR_TO_ENUM = {
            HumanAgent.HumanState._member_names_[s]: s \
                                for s in list(HumanAgent.HumanState)}
        self.STATE_ENUM_TO_STR = dict(zip(self.STATE_STR_TO_ENUM.values(),
                                        self.STATE_STR_TO_ENUM.keys()))

    def _construct_actions(self, world_config):
        """
        transit_OBJ
        grasp (if an object is at the current location)
        transfer_TARGETLOCATION
        release (if object is in gripper)
        wait
        """
        # actions =
        count = 0
        actions = {}
        for object in world_config.objects:
            actions[f'interveneIn{object.name}'] = count
            count += 1

        actions['wait'] = count
        actions['returnObj'] = count + 1

        return custom_enum('HumanAction', actions)

    def _step_default(self, action: IntEnum, world_config) -> Tuple[Done, Reward]:
        cost = 0
        done = False

        if action is None or np.isnan(action):
            self.world_state = world_config.get_state()
            return cost, done

        if world_config.curr_player != 'env':
            return cost, True

        action_str = self.ACTION_ENUM_TO_STR[action]
        action_info = action_str.split('In')

        # Human can only intervene one object at the time
        object = None

        if action_info[0] == 'intervene':
            # Get the object information
            object_name = action_info[1]
            object = world_config.get_object(object_name)

            if not world_config.can_intervene() or \
                not object.can_intervene() or \
                self._state == HumanAgent.HumanState.Intervene:
                return cost, True

            # update robot's state and the object state
            self._state = HumanAgent.HumanState.Intervene
            object.state = FrankaWorldConfig.Object.ObjectState.Intervene

        elif action_info[0] == "returnObj":

            if self._state is not HumanAgent.HumanState.Intervene:
                return cost, True

            object = world_config.get_object_in_intervention()

            if not object.must_return():
                return cost, True

            self._state = HumanAgent.HumanState.Ready
            object.reset_state()

        elif action_info[0] == "wait":
            # The world doesn't change at all
            if self._state is HumanAgent.HumanState.Intervene:

                object = world_config.get_object_in_intervention()

                if object.must_return():
                    return cost, True

        else:
            assert False, f"unknown action {action}"

        # Increment the intervention counts
        world_config.update(object) # increment the interventions

        # Keep the world's state so that it can be included as a part
        # of franka's state (which isn't actullay franka's state)
        self.world_state = world_config.get_state()

        return cost, done

    def render(self, img):
        raise NotImplementedError('')

    def encode(self):
        """Encode the a description of this object as a 3-tuple of integers"""
        # If we want to encode more information other than self.dir,
        # we can use mapping (an attribute to a number) or assign a digit for each attribute
        return [OBJECT_TO_IDX[self.type], COLOR_TO_IDX[self.color], int(self.state)]

    @property
    def state(self):
        return self._state, self.world_state

    def set_state(self, state, world_config) -> None:

        self._state = state[0]
        world_config.set_state(state[1])

    def state_str(self, state=None):
        if state is None:
            state = self.state

        state = list(state)
        state[0] = FRANKA_ACTION_ABR[self.STATE_ENUM_TO_STR[state[0]]]
        state[1] = self.world_config.state_str(state[1])

        return tuple(state)

    def str_to_state(self, state_str: str):
        state_str = list(state_str)
        state = state_str
        state[0] = self.STATE_STR_TO_ENUM[ABR_TO_FRANKA_ACTION[state_str[0]]]
        state[1] = self.world_config.str_to_state(state_str[1])

        return tuple(state)


class Flood(BooleanAgent):

    def __init__(self, name: str = 'flood', type: str='water'):
        super().__init__(name, type)

    def encode_as_str(self):
        obj_type, obj_color, obj_state = self.encode()
        return '_'.join([IDX_TO_OBJECT[obj_type],
                         IDX_TO_COLOR[obj_color],
                         IDX_TO_STATE[obj_state]])


class MultiAgentMiniGridEnv(MiniGridEnv):
    """
    Multi-Agent 2D grid world game environment
    """

    def __init__(
        self,
        grid_size: int=None,
        width: int=None,
        height: int=None,
        max_steps: int=100,
        see_through_walls: bool=False,
        seed: int=1337,
        agent_view_size=7,
        concurrent: bool = False,
        only_reward_system: bool = True):

        # Can't set both grid_size and width/height
        if grid_size:
            assert width == None and height == None
            width = grid_size
            height = grid_size

        # Number of cells (width and height) in the agent view
        assert agent_view_size % 2 == 1
        assert agent_view_size >= 3
        self.view_size = agent_view_size

        # Observations are dictionaries containing an
        # encoding of the grid and a textual 'mission' string
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )
        self.observation_space = spaces.Dict({
            'image': self.observation_space
        })

        # Range of possible rewards
        self.reward_range = (0, 1)

        # Window to use for human rendering mode
        self.window = None

        # Environment configuration
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.see_through_walls = see_through_walls

        # Multi-Agent Configuration
        self.concurrent = concurrent
        self.sys_agent = None
        self.only_reward_system = only_reward_system

        # Initialize the RNG
        self.seed(seed=seed)

        # Initialize the state
        # Assume grid is generated and agents/objects are placed
        # by calling self.reset()
        # Use grid = MultiObjGrid(Grid(width, height))
        self.reset()

        # MultiActions PER agent
        agents_actions = [a.actions for a in self.agents]
        self.construct_multiactions(agents_actions, concurrent)

        self.multiactions = np.concatenate([a.multiactions for a in self.agents], axis=0)
        # Concatenated MultiActions as an environment
        self.ACTION_STR_TO_ENUM = {''.join(str(action)): tuple(action)
                                   for action in self.multiactions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

        action_dict = {s: i for i, s in enumerate(self.ACTION_STR_TO_ENUM)}
        self.actions = Enum('MultiAgentActions', action_dict)
        self.unwrapped.actions = self.actions
        self.unwrapped.action_space = gym.spaces.Discrete(len(self.multiactions))

    def _get_action_str(self, action: MultiAgentAction) -> str:
        return ''.join(str(action))

    def reset(self):

        self.agents = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid(self.width, self.height)

        if self.agents is None:
            raise Exception('There is no agent in the grid')

        # These fields should be defined by _gen_grid
        for a in self.agents:
            assert a.pos is not None
            assert a.dir is not None

        # Item picked up, being carried, initially nothing
        for a in self.agents:
            a.carrying = None

        # Step  since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def construct_multiactions(self, agents_actions, concurrent: bool = False) -> MultiAgentActions:
        """
        Given each agent's actions, take a combination of them
        """

        # If it's a concurrent-based game, return as it is
        if concurrent:
            # Take combinations of agents actions
            actions_list = [a.actions for a in self.agents]
            action_products = list(itertools.product(*actions_list))
            return action_products

        # For each column, store each agent's actions
        for i in range(self.n_agent):
            # Create an empty array
            n_action = len(agents_actions[i])
            multi_actions = np.empty((n_action, self.n_agent), dtype=object)
            multi_actions.fill(None)

            # Only fill the current agent's column
            multi_actions[:, i] = agents_actions[i]
            self.agents[i].multiactions = multi_actions.tolist()

    def step(self, action: Union[MultiAgentAction, IntEnum]) -> StepData:
        """
        Multi-Agent MiniGrid can take a list of actions or an integer.
        """

        self.step_count += 1

        reward = 0
        done = False

        if isinstance(action, IntEnum):
            action = self.multiactions[action]

        # Concat rewards and dones
        for agent, a in zip(self.agents, action):
            # pass grid by reference so that each agent can
            # manipulate the grid
            reward_, done_ = agent.step(a, self.grid)

            if self.only_reward_system:
                if agent == self.sys_agent:
                    reward += reward_
            else:
                reward += reward_

            done = bool(done + done_)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def render(self, mode='human', close=False, highlight=True, tile_size=TILE_PIXELS):
        """
        Render the whole-grid human view
        """
        # tmp 
        highlight=True

        if close:
            if self.window:
                self.window.close()
            return

        if mode == 'human' and not self.window:
            import gym_minigrid.window
            self.window = gym_minigrid.window.Window('gym_minigrid')
            self.window.show(block=False)

        # Compute which cells are visible to the agent
        _, vis_mask = self.gen_obs_grid()

        # Compute the world coordinates of the bottom-left corner
        # of the agent's view area
        f_vec = self.dir_vec
        r_vec = self.right_vec
        top_left = self.agent_pos + f_vec * (self.agent_view_size-1) - r_vec * (self.agent_view_size // 2)

        # Mask of which cells to highlight
        highlight_mask = np.zeros(shape=(self.width, self.height), dtype=bool)

        # For each cell in the visibility mask
        for vis_j in range(0, self.agent_view_size):
            for vis_i in range(0, self.agent_view_size):
                # If this cell is not visible, don't highlight it
                if not vis_mask[vis_i, vis_j]:
                    continue

                # Compute the world coordinates of this cell
                abs_i, abs_j = top_left - (f_vec * vis_j) + (r_vec * vis_i)

                if abs_i < 0 or abs_i >= self.width:
                    continue
                if abs_j < 0 or abs_j >= self.height:
                    continue

                # Mark this cell to be highlighted
                highlight_mask[abs_i, abs_j] = True

        # Render the whole grid
        img = self.grid.render(
            tile_size,
            agent_pos=None,
            agent_dir=None,
            highlight_mask=highlight_mask if highlight else None
        )

        if mode == 'human':
            self.window.show_img(img)
            self.window.set_caption(self.mission)

        return img

    def put_agent(self, agent, i, j, direction, sys_agent: bool = False):
        agent.pos = np.array([i, j])
        agent.dir = direction

        if self.agents is None:
            self.agents = [agent]
        else:
            self.agents.append(agent)

        if sys_agent and self.sys_agent is None:
            self.sys_agent = agent

        return super().put_obj(agent, i, j)

    def place_agent(self, top=None, size=None, rand_dir=True, max_tries=math.inf,
                    sys_agent: bool = False):
        """
        Set the agent's starting point at an empty position in the grid
        """

        if self.agents is None:
            agent = Agent(name='agent1', view_size=self.view_size)
        else:
            n_agent = len(self.agents)
            agent = Agent(name=f'agent{n_agent+1}', view_size=self.view_size)

        pos = self.place_obj(agent, top, size, max_tries=max_tries)
        agent.pos = pos

        if rand_dir:
            agent.dir = self._rand_int(0, 4)

        if self.agents is None:
            self.agents = [agent]
        else:
            self.agents.append(agent)

        if sys_agent and self.sys_agent is None:
            self.sys_agent = agent

        return pos

    def gen_obs_grid(self):
        """
        Generate the sub-grid observed by the agent.
        This method also outputs a visibility mask telling us which grid
        cells the agent can actually see.
        """

        topX, topY, botX, botY = self.get_view_exts()

        grid = self.grid.slice(topX, topY, self.agent_view_size, self.agent_view_size)

        for i in range(self.agent_dir + 1):
            grid = grid.rotate_left()

        # Process occluders and visibility
        # Note that this incurs some performance cost
        if not self.see_through_walls:
            vis_mask = grid.process_vis(agent_pos=(self.agent_view_size // 2 , self.agent_view_size - 1))
        else:
            vis_mask = np.ones(shape=(grid.width, grid.height), dtype=bool)

        # Make it so the agent sees what it's carrying
        # We do this by placing the carried object at the agent's position
        # in the agent's partially observable view
        agent_pos = grid.width // 2, grid.height - 1

        # if self.carrying:
        #     grid.set(*agent_pos, self.carrying)
        # else:
        #     grid.set(*agent_pos, None)

        return grid, vis_mask

    def gen_obs(self):
        """
        Generate the agent's view (partially observable, low-resolution encoding)
        """

        grid, vis_mask = self.gen_obs_grid()

        # Encode the partially observable view into a numpy array
        image = grid.encode(vis_mask)

        assert hasattr(self, 'mission'), "environments must define a textual mission string"

        # Observations are dictionaries containing:
        # - an image (partially observable view of the environment)
        # - the agent's direction/orientation (acting as a compass)
        # - a textual mission string (instructions for the agent)
        obs = {
            'image': image,
            'direction': self.agent_dir,
            'mission': self.mission
        }

        return obs

    @property
    def agent_pos(self):
        return self.agents[0].pos

    @property
    def agent_dir(self):
        return self.agents[0].dir

    @property
    def agent_view_size(self):
        return self.agents[0].view_size

    @agent_view_size.setter
    def agent_view_size(self, view_size):
        self.view_size = view_size
        for agent in self.agents:
            agent.view_size = view_size

    @property
    def carrying(self):
        return self.agents[0].carrying

    @property
    def dir_vec(self):
        return self.agents[0].dir_vec

    @property
    def right_vec(self):
        return self.agents[0].right_vec

    @property
    def n_agent(self):
        if self.agents is None:
            return 0
        return len(self.agents)


class MultiAgentEnv(gym.Env):
    """
    Multi-Agent 2D grid world game environment
    """

    def __init__(
        self,
        max_steps: int=100,
        seed: int=1337,
        concurrent: bool = False,
        only_reward_system: bool = True):

        # Window to use for human rendering mode
        self.window = None
        self.max_steps = max_steps

        # Multi-Agent Configuration
        self.concurrent = concurrent
        self.sys_agent = None
        self.only_reward_system = only_reward_system

        # Initialize the RNG
        self.width = 2
        self.height = 2
        self.seed(seed=seed)

        # Initialize the state
        self.reset()

        # MultiActions PER agent
        agents_actions = [a.actions for a in self.agents]
        self.construct_multiactions(agents_actions, concurrent)

        self.multiactions = np.concatenate([a.multiactions for a in self.agents], axis=0).tolist()

        # Concatenated MultiActions as an environment
        self.ACTION_STR_TO_ENUM = {}
        for multiaction in self.multiactions:

            action_strs = [agent.ACTION_ENUM_TO_STR[action] if action is not None else 'None'\
                for agent, action in zip(self.agents, multiaction)]

            # Connect each agent's action with __ to make it one string
            action_str = '__'.join(action_strs)

            self.ACTION_STR_TO_ENUM[action_str] = tuple(multiaction)

        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

        action_dict = {s: i for i, s in enumerate(self.ACTION_STR_TO_ENUM)}
        self.actions = Enum('MultiAgentActions', action_dict)
        # self.unwrapped.actions = self.actions
        # self.unwrapped.action_space = gym.spaces.Discrete(len(self.multiactions))

    def _get_action_str(self, action: MultiAgentAction) -> str:
        return ''.join(str(action))

    def reset(self):

        self.agents = None

        # Generate a new random grid at the start of each episode
        # To keep the same grid for each episode, call env.seed() with
        # the same seed before calling env.reset()
        self._gen_grid()

        if self.agents is None:
            raise Exception('There is no agent in the grid')

        # Step  since episode start
        self.step_count = 0

        # Return first observation
        obs = self.gen_obs()

        return obs

    def _gen_grid(self):
        raise NotImplementedError('')

    def construct_multiactions(self, agents_actions, concurrent: bool = False) -> MultiAgentActions:
        """
        Given each agent's actions, take a combination of them
        """

        # If it's a concurrent-based game, return as it is
        if concurrent:
            # Take combinations of agents actions
            actions_list = [a.actions for a in self.agents]
            action_products = list(itertools.product(*actions_list))
            return action_products

        # For each column, store each agent's actions
        for i in range(self.n_agent):
            # Create an empty array
            n_action = len(agents_actions[i])
            multi_actions = np.empty((n_action, self.n_agent), dtype=object)
            multi_actions.fill(None)

            # Only fill the current agent's column
            multi_actions[:, i] = agents_actions[i]
            self.agents[i].multiactions = multi_actions.tolist()

    def step(self, action: Union[MultiAgentAction, IntEnum]) -> StepData:
        """
        Multi-Agent MiniGrid can take a list of actions or an integer.
        """

        self.step_count += 1

        reward = 0
        done = False

        if isinstance(action, IntEnum):
            action = self.multiactions[action]

        # Concat rewards and dones
        for agent, a in zip(self.agents, action):
            # pass grid by reference so that each agent can
            # manipulate the grid
            reward_, done_ = agent.step(a, self.grid)

            if self.only_reward_system:
                if agent == self.sys_agent:
                    reward += reward_
            else:
                reward += reward_

            done = bool(done + done_)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}

    def render(self, **kwargs):
        """
        Render the whole-grid human view
        """
        raise NotImplementedError('')

    def put_agent(self, agent, sys_agent: bool = False):
        if self.agents is None:
            self.agents = [agent]
        else:
            self.agents.append(agent)

        if sys_agent and self.sys_agent is None:
            self.sys_agent = agent

    def place_agent(self, sys_agent: bool = False):
        """
        Set the agent's starting point at an empty position in the grid
        """

        if self.agents is None:
            agent = Agent(name='agent1')
        else:
            n_agent = len(self.agents)
            agent = Agent(name=f'agent{n_agent+1}')

        if self.agents is None:
            self.agents = [agent]
        else:
            self.agents.append(agent)

        if sys_agent and self.sys_agent is None:
            self.sys_agent = agent

    def gen_obs(self):
        return {'image': np.random.rand(2, 2)}

    @property
    def n_agent(self):
        if self.agents is None:
            return 0
        return len(self.agents)


class ModifyMultiAgentNumActionsWrapper(gym.core.Wrapper):
    def __init__(self, env: MultiAgentMiniGridEnv,
                 num_steps: Dict[str, List[int]]):
        """
        Modify agents' actions. Currently, it can only change the number of steps.

        :arg env:               Multi-Agent Environment
        :arg num_steps:         A list of No. of steps each agent can take
        """
        # Wrap the given env and register it as self.env
        # "The original" gym env can be accessed via self.env.unwrapped
        super().__init__(env)

        self.agents = self.unwrapped.agents
        self._add_agent_multistep_actions(num_steps)
        # A list of multi-step multi-actions
        # e.g. [[north, north], [south]]
        agents_actions = [a.multistep_actions for a in self.agents]
        self._add_agent_multistep_multiactions(agents_actions, env.concurrent)

        # A list of all possible multi-actions
        # e.g.) Sys agent takes 2 steps (north and south)
        #       multi_action = [[north, south], [None], [None]]
        self.multiactions = np.concatenate([a.multiactions for a in self.agents], axis=0).tolist()

        action_strings = {'_'.join([''.join(str(a)) for a in action]) \
            for action in self.multiactions}
        action_dict = {s: i for i, s in enumerate(action_strings)}

        self.actions = Enum('MultiAgentActions', action_dict)
        # self.action_space = gym.spaces.Discrete(len(self.multiactions))

        # Concatenated MultiActions as an environment
        self.ACTION_STR_TO_ENUM = {}
        for multiaction in self.multiactions:

            action_strs = []
            for agent, actions in zip(self.agents, multiaction):

                agent_action_strs = []
                for action in actions:
                    a_str = agent.ACTION_ENUM_TO_STR[action] if action is not None else 'None'
                    agent_action_strs.append(a_str)
                agent_action_str = '_'.join(agent_action_strs)

                action_strs.append(agent_action_str)

            # Connect each agent's action with __ to make it one string
            action_str = '__'.join(action_strs)

            self.ACTION_STR_TO_ENUM[action_str] = tuple(map(tuple, multiaction))

        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

    def _add_agent_multistep_actions(self, num_steps: Dict[str, List]):
        """
        :arg num_steps:        A list of allowed_steps for 'sys' & 'env'
        """
        for i in range(self.n_agent):
            # first agent is the system's agent
            if self.agents[i] == self.sys_agent:
                n_steps = num_steps['sys']
            else:
                n_steps = num_steps['env']

            atomic_actions = self.agents[i].actions
            self.agents[i].multistep_actions = []

            for n_step in n_steps:
                n_copy_of_atomic_actions = list(atomic_actions for i in range(n_step))
                n_step_actions = list(itertools.product(*n_copy_of_atomic_actions))
                n_step_actions = list(map(list, n_step_actions))
                self.agents[i].multistep_actions += n_step_actions

    def _add_agent_multistep_multiactions(self,
        agents_actions, concurrent: bool = False) -> MultiAgentActions:
        """
        Given each agent's actions, take a combination of them
        """

        # If it's a concurrent-based game, return as it is
        if concurrent:
            # Take combinations of agents actions
            actions_list = [a.actions for a in self.agents]
            action_products = list(itertools.product(*actions_list))
            return action_products

        # For each column, store each agent's actions
        for i in range(self.n_agent):
            # Create an empty array
            n_action = len(agents_actions[i])
            multi_actions = np.empty((n_action, self.n_agent), dtype=object)
            multi_actions.fill([None])

            # Only fill the current agent's column
            multi_actions[:, i] = agents_actions[i]
            self.agents[i].multiactions = multi_actions

    def step(self, action: Union[MultiStepMultiAgentAction, IntEnum]) -> StepData:
        """
        Multi-Agent MiniGrid can take a list of actions or an integer.
        """

        self.step_count += 1

        reward = 0
        done = False

        if isinstance(action, IntEnum):
            action = self.multiactions[action]

        n_step = max(len(a) for a in action)
        for i in range(n_step):
            one_step_multiaction = [a[i] if len(a)>i else None for a in action]

            # Concat rewards and dones
            for agent, a in zip(self.unwrapped.agents, one_step_multiaction):
                # pass grid by reference so that each agent can
                # manipulate the grid
                reward_, done_ = agent.step(a, self.unwrapped.grid)

                if hasattr(self.unwrapped, 'only_reward_system') and \
                    self.unwrapped.only_reward_system:
                    if agent == self.unwrapped.sys_agent:
                        reward += reward_
                else:
                    reward += reward_

                done = bool(done + done_)

        if self.step_count >= self.max_steps:
            done = True

        obs = self.gen_obs()

        return obs, reward, done, {}


class ModifyActionsWrapper(gym.core.Wrapper):
    """
    This class allows you to modify the action space and behavior of the agent

    :param      env:           The gym environment to wrap
    :param      actions_type:  The actions type string
                               {'static', 'simple_static', 'diag_static',
                                'default'}
                               'static':
                               use a directional agent only capable of going
                               forward and turning
                               'simple_static':
                               use a non-directional agent which can only move
                               in cardinal directions in the grid
                               'default':
                               use an agent which has the default MinigridEnv
                               actions, suitable for dynamic environments.
    """

    # Enumeration of possible actions
    # as this is a static environment, we will only allow for movement actions
    # For a simple environment, we only allow the agent to move:
    # North, South, East, or West
    class SimpleStaticActions(IntEnum):
        # move in this direction on the grid
        north = 0
        south = 1
        east = 2
        west = 3

    SIMPLE_ACTION_TO_DIR_IDX = {SimpleStaticActions.north: 3,
                                SimpleStaticActions.south: 1,
                                SimpleStaticActions.east: 0,
                                SimpleStaticActions.west: 2}

    # Enumeration of possible actions
    # as this is a static environment, we will only allow for movement actions
    # For a simple environment, we only allow the agent to move:
    # Northeast, Northwest, Southeast, or Southwest
    class DiagStaticActions(IntEnum):
        # move in this direction on the grid
        northeast = 0
        northwest = 1
        southeast = 2
        southwest = 3

    DIAG_ACTION_TO_POS_DELTA = {
        DiagStaticActions.northeast: (1, -1),
        DiagStaticActions.northwest: (-1, -1),
        DiagStaticActions.southeast: (1, 1),
        DiagStaticActions.southwest: (-1, 1)}

    # Enumeration of possible actions
    # as this is a static environment, we will only allow for movement actions
    class StaticActions(IntEnum):
        # Turn left, turn right, move forward
        left = 0
        right = 1
        forward = 2

    def __init__(self, env: EnvType, actions_type: str = 'static'):

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)

        self._allowed_actions_types = set(['static', 'simple_static',
                                           'diag_static', 'default'])
        if actions_type not in self._allowed_actions_types:
            msg = f'actions_type ({actions_type}) must be one of: ' + \
                  f'{actions_type}'
            raise ValueError(msg)

        # Need to change the Action enumeration in the base environment.
        # This also changes the "step" behavior, so we also change that out
        # to match the new set of actions
        self._actions_type = actions_type

        if actions_type == 'static':
            actions = ModifyActionsWrapper.StaticActions
            step_function = self._step_default
        elif actions_type == 'simple_static':
            actions = ModifyActionsWrapper.SimpleStaticActions
            step_function = self._step_simple_static
        elif actions_type == 'diag_static':
            actions = ModifyActionsWrapper.DiagStaticActions
            step_function = self._step_diag_static
        elif actions_type == 'default':
            actions = MiniGridEnv.Actions
            step_function = self._step_default

        self.unwrapped.actions = actions
        self._step_function = step_function

        # Actions are discrete integer values
        num_actions = len(self.unwrapped.actions)
        self.unwrapped.action_space = gym.spaces.Discrete(num_actions)

        # building some more constant DICTS dynamically from the env data
        self.ACTION_STR_TO_ENUM = {self._get_action_str(action): action
                                   for action in self.unwrapped.actions}
        self.ACTION_ENUM_TO_STR = dict(zip(self.ACTION_STR_TO_ENUM.values(),
                                           self.ACTION_STR_TO_ENUM.keys()))

    def step(self, action: IntEnum) -> StepData:

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        base_env.step_count += 1

        done, reward = self._step_function(action)

        if base_env.step_count >= base_env.max_steps:
            done = True

        obs = base_env.gen_obs()

        return obs, reward, done, {}

    def _step_diag_static(self, action: IntEnum) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        start_pos = base_env.agent_pos

        # a diagonal action is really just two simple actions :)
        pos_delta = ModifyActionsWrapper.DIAG_ACTION_TO_POS_DELTA[action]

        # Get the contents of the new cell of the agent
        new_pos = tuple(np.add(start_pos, pos_delta))
        new_cell = base_env.grid.get(*new_pos)

        if new_cell is None or new_cell.can_overlap():
            base_env.agent_pos = new_pos
        if new_cell is not None and new_cell.type == 'goal':
            done = True
            reward = base_env._reward()
        if new_cell is not None and new_cell.type == 'lava':
            done = True

        return done, reward

    def _step_simple_static(self, action: IntEnum) -> Tuple[Done, Reward]:

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        reward = 0
        done = False

        # save the original direction so we can reset it after moving
        old_dir = base_env.agent_dir
        new_dir = ModifyActionsWrapper.SIMPLE_ACTION_TO_DIR_IDX[action]
        base_env.agent_dir = new_dir

        # Get the contents of the cell in front of the agent
        fwd_pos = base_env.front_pos
        fwd_cell = base_env.grid.get(*fwd_pos)

        if fwd_cell is None or fwd_cell.can_overlap():
            base_env.agent_pos = fwd_pos
        if fwd_cell is not None and fwd_cell.type == 'goal':
            done = True
            reward = base_env._reward()
        if fwd_cell is not None and fwd_cell.type == 'lava':
            done = True

        # reset the direction of the agent, as it really cannot change
        # direction
        base_env.agent_dir = old_dir

        return done, reward

    def _step_default(self, action: IntEnum) -> Tuple[Done, Reward]:

        reward = 0
        done = False

        # all of these changes must affect the base environment to be seen
        # across all other wrappers
        base_env = self.unwrapped

        # Get the position in front of the agent
        fwd_pos = base_env.front_pos

        # Get the contents of the cell in front of the agent
        fwd_cell = base_env.grid.get(*fwd_pos)
        # Rotate left
        if action == base_env.actions.left:
            base_env.agent_dir -= 1
            if base_env.agent_dir < 0:
                base_env.agent_dir += 4

        # Rotate right
        elif action == base_env.actions.right:
            base_env.agent_dir = (base_env.agent_dir + 1) % 4

        # Move forward
        elif action == base_env.actions.forward:
            if fwd_cell is None or fwd_cell.can_overlap():
                base_env.agent_pos = fwd_pos
            if fwd_cell is not None and fwd_cell.type == 'goal':
                done = True
                reward = base_env._reward()
            if fwd_cell is not None and fwd_cell.type == 'lava':
                done = True

        # Pick up an object
        elif action == base_env.actions.pickup:
            if fwd_cell and fwd_cell.can_pickup():
                if base_env.carrying is None:
                    base_env.carrying = fwd_cell
                    base_env.carrying.cur_pos = np.array([-1, -1])
                    base_env.grid.set(*fwd_pos, None)

        # Drop an object
        elif action == base_env.actions.drop:
            if not fwd_cell and base_env.carrying:
                base_env.grid.set(*fwd_pos, base_env.carrying)
                base_env.carrying.cur_pos = fwd_pos
                base_env.carrying = None

        # Toggle/activate an object
        elif action == base_env.actions.toggle:
            if fwd_cell:
                fwd_cell.toggle(base_env, fwd_pos)

        # Done action (not used by default)
        elif action == base_env.actions.done:
            pass

        else:
            assert False, "unknown action"

        return done, reward

    def _get_action_str(self, action_enum: ActionsEnum) -> str:
        """
        Gets a string representation of the action enum constant

        :param      action_enum:  The action enum constant to convert

        :returns:   The action enum's string representation
        """

        return self.unwrapped.actions._member_names_[action_enum]


class StaticMinigridTSWrapper(gym.core.Wrapper):
    """
    Wrapper to define an environment that can be represented as a transition
    system.

    This means that the environment must be STATIC -> no keys or doors opening
    as this would require a reactive synthesis formulation.

        :param      env:                   The gym environment to wrap and
                                           compute transitions on
        :param      seeds:                 The random seeds given to the
                                           Minigrid environment, so when the
                                           environment is reset(), it remains
                                           the same.
        :param      actions_type:          The actions type string
                                           {'static', 'simple_static',
                                           'diag_static', 'default'}
                                           'static': use a directional agent
                                           only capable of going forward and
                                           turning
                                           'simple_static': use a
                                           non-directional agent which can only
                                           move in cardinal directions in the
                                           grid
                                           'diag_static': use a
                                           non-directional agent which can only
                                           move in intercardinal directions
                                           (diagonally) in the grid
                                           'default': use an agent which
                                           has the default MinigridEnv actions,
                                           suitable for dynamic environments.
        :param      monitor_log_location:  The location to save gym env
                                           monitor logs & videos
    """

    env: EnvType
    IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))
    DIR_TO_STRING = bidict({0: 'right', 1: 'down', 2: 'left', 3: 'up'})

    def __init__(
        self,
        env: EnvType,
        seeds: List[int] = [0],
        actions_type: str = 'static',
        monitor_log_location: str = GYM_MONITOR_LOG_DIR_NAME
    ) -> 'StaticMinigridTSWrapper':

        self.monitor_log_location = monitor_log_location
        self._force_monitor = False
        self._resume_monitor = True
        self._uid_monitor = None
        self._mode = None

        self._allowed_actions_types = set(['static', 'simple_static',
                                           'diag_static', 'default'])
        if actions_type not in self._allowed_actions_types:
            msg = f'actions_type ({actions_type}) must be one of: ' + \
                  f'{self._allowed_actions_types}'
            raise ValueError(msg)

        if actions_type == 'simple_static' or actions_type == 'diag_static':
            env.directionless_agent = True
        elif actions_type == 'static' or actions_type == 'default':
            env.directionless_agent = False

        env = ViewSizeWrapper(env, agent_view_size=3)
        env = ModifyActionsWrapper(env, actions_type)
        env = FullyObsWrapper(ReseedWrapper(env, seeds=seeds))
        env = wrappers.Monitor(env, self.monitor_log_location,
                               video_callable=False,
                               force=self._force_monitor,
                               resume=self._resume_monitor,
                               mode=self._mode)

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)
        self.actions = self.unwrapped.actions

        # We only compute state observations label maps once here, as the
        # environment MUST BE STATIC in this instance
        obs = self.state_only_obs_reset()
        self.agent_start_pos, self.agent_start_dir = self._get_agent_props()

        (self.obs_str_idxs_map,
         self.cell_obs_map,
         self.cell_to_obs) = self._get_observation_maps(self.agent_start_pos,
                                                        obs)

        self.reset()

    def render_notebook(self, filename: str = None, dpi: int = 300) -> None:
        """
        Wrapper for the env.render() that works in notebooks
        """

        plt.imshow(self.env.render(mode='rgb_image', tile_size=64),
                   interpolation='bilinear', highlight=False)
        plt.axis('off')
        if filename:
            plt.savefig(filename, dpi=dpi)
        plt.show()

    def reset(self, new_monitor_file: bool = False, **kwargs) -> np.ndarray:
        """
        Wrapper for the reset function that manages the monitor wrapper

        :param      new_monitor_file:  whether to create a new monitor file
        :param      kwargs:            The keywords arguments to pass on to the
                                       next wrapper's reset()

        :returns:   env observation
        """

        self.close()
        self._start_monitor(new_monitor_file)
        observation = self.env.reset(**kwargs)

        return observation

    def state_only_obs(self, obs: dict) -> EnvObs:
        """
        Extracts only the grid observation from a step() observation

        This command only works for a MiniGridEnv obj, as their obs:
            obs, reward, done, _ = MiniGridEnbv.step()
        is a dict containing the (full/partially) observable grid observation

        :param      obs:  Full observation received from MiniGridEnbv.step()

        :returns:   The grid-only observation
        """

        cell_obs = obs['image']

        return cell_obs

    def state_only_obs_reset(self) -> EnvObs:
        """
        Resets the environment, but returns the grid-only observation

        :returns:   The grid-only observation after reseting
        """

        obs = self.env.reset()

        return self.state_only_obs(obs)

    def state_only_obs_step(self, action: ActionsEnum) -> StepData:
        """
        step()s the environment, but returns only the grid observation

        This command only works for a MiniGridEnv obj, as their obs:
            obs, reward, done, _ = MiniGridEnbv.step()
        is a dict containing the (full/partially) observable grid observation

        :param      action:  The action to take

        :returns:   Normal step() return data, but with obs being only the grid
        """

        obs, reward, done, _ = self.env.step(action)

        return self.state_only_obs(obs), reward, done, {}

    def _get_agent_props(self) -> Tuple[AgentPos, AgentDir]:
        """
        Gets the agent's position and direction in the base environment
        """

        base_env = self.env.unwrapped

        return tuple(base_env.agent_pos), base_env.agent_dir

    def _set_agent_props(self, state=None) -> None:
        """
        Sets the agent's position and direction in the base environment

        :param      position:   The new agent grid position
        :param      direction:  The new agent direction
        """
        base_env = self.env.unwrapped

        if state is not None:
            position, direction = state

            if position is not None:
                base_env.agent_pos = position

            if direction is not None:
                base_env.agent_dir = direction

    def _get_env_prop(self, env_property_name: str):
        """
        Gets the base environment's property.

        :param      env_property_name:  The base environment's property name

        :returns:   The base environment's property.
        """

        base_env = self.env.unwrapped

        return getattr(base_env, env_property_name)

    def _set_env_prop(self, env_property_name: str, env_property) -> None:
        """
        Sets the base environment's property.

        :param      env_property_name:  The base environment's property name
        :param      env_property:       The new base environment property data
        """

        base_env = self.env.unwrapped
        setattr(base_env, env_property_name, env_property)

    def _obs_to_prop_str(self, obs: EnvObs,
                         col_idx: int, row_idx: int) -> str:
        """
        Converts a grid observation array into a string based on Minigrid ENUMs

        :param      obs:      The grid observation
        :param      col_idx:  The col index of the cell to get the obs. string
        :param      row_idx:  The row index of the cell to get the obs. string

        :returns:   verbose, string representation of the state observation
        """

        obj_type, obj_color, obj_state = obs[col_idx, row_idx]
        agent_pos, _ = self._get_agent_props()
        is_agent = (col_idx, row_idx) == tuple(agent_pos)

        prop_string_base = '_'.join([IDX_TO_OBJECT[obj_type],
                                     IDX_TO_COLOR[obj_color]])

        if is_agent:
            return '_'.join([prop_string_base, self.DIR_TO_STRING[obj_state]])
        else:
            return '_'.join([prop_string_base, self.IDX_TO_STATE[obj_state]])

    def _make_transition(self, action: ActionsEnum,
                         state: Tuple[AgentPos, AgentDir]) -> Tuple[AgentPos,
                                                                    AgentDir,
                                                                    Done]:
        """
        Makes a state transition in the environment, assuming the env has state

        :param      action:     The action to take
        :param      pos:        The agent's position
        :param      direction:  The agent's direction

        :returns:   the agent's new state, whether or not step() emitted done
        """
        self._set_agent_props(state)
        _, reward, done, _ = self.state_only_obs_step(action)

        return self._get_agent_props(), reward, done

    def _get_obs_str_of_start_cell(self, start_obs: EnvObs) -> str:
        """
        Gets the cell observation string for the start (reset) state

        if returns None, then the agent cannot leave the start state and the
        environment is broken lol.

        :param      start_obs:   The gridcell observation matrix at reset()

        :returns:   a full-cell observation at the agent's reset state

        :raises     ValueError:  If the agent can't do anything at reset()
        """

        self.reset()
        init_state = (self.agent_start_pos, self.agent_start_dir)

        # sigh.... Well, we know that if you can't move within 3 steps, then
        # the environment is completely unsolvable, or you start on the goal
        # state.
        for a1 in self.actions:
            (s1,
             _,
             done) = self._make_transition(a1, init_state)

            if done:
                self.reset()

            for a2 in self.actions:
                (s2,
                 _,
                 done) = self._make_transition(a2, s1)

                if done:
                    self.reset()

                for a3 in self.actions:
                    self._set_agent_props(s2)
                    obs, _, done, _ = self.state_only_obs_step(a3)
                    agent_pos3, _ = self._get_agent_props()
                    at_new_cell = agent_pos3 != init_state[0]

                    if at_new_cell:
                        obs_str = self._obs_to_prop_str(obs, *init_state[0])
                        self.reset()
                        return obs_str

                    if done:
                        self.reset()

        msg = f'No actions allow the agent to make any progress in the env.'
        raise ValueError(msg)

    def _get_observation_maps(self, start_pos: AgentPos,
                              obs: EnvObs) -> Tuple[bidict, defaultdict, dict]:
        """
        Computes mappings for grid state (cell) observations.

        A cell obs. array consists of [obj_type, obj_color, obj_state], where
        each element is an integer index in a ENUM from the Minigrid env.

            obs_str_idxs_map[cell_obs_str] = np.array(cell obs. array)
            cell_obs_map[cell_obs_str] = list_of((cell_col_idx, cell_row_idx))
            cell_to_obs[(cell_col_idx, cell_row_idx)] = cell_obs_str

        :param      start_pos:  The agent's start position
        :param      obs:        The grid observation

        :returns:   (mapping from cell obs. string -> cell obs. array
                     mapping from cell obs. string -> cell indices
                        NOTE: each key in this dict has a list of values assoc.
                     mapping from cell indices -> cell obs. string)
        """

        obs_str_idxs_map = bidict()
        cell_obs_map = defaultdict(list)
        cell_to_obs = dict()

        (num_cols, num_rows, num_cell_props) = obs.shape

        for row_idx in range(num_rows):
            for col_idx in range(num_cols):

                obs_str = self._obs_to_prop_str(obs, col_idx, row_idx)
                obj = obs[col_idx, row_idx][0]

                is_agent = IDX_TO_OBJECT[obj] == 'agent'
                is_wall = IDX_TO_OBJECT[obj] == 'wall'

                if not is_agent and not is_wall:
                    obs_str_idxs_map[obs_str] = tuple(obs[col_idx, row_idx])
                    cell_obs_map[obs_str].append((col_idx, row_idx))
                    cell_to_obs[(col_idx, row_idx)] = obs_str

        # need to add the agent's start cell observation to the environment
        start_cell_obs_str = self._get_obs_str_of_start_cell(obs)
        start_col, start_row = start_pos

        obs_str_idxs_map[start_cell_obs_str] = tuple(obs[start_col, start_row])
        cell_obs_map[start_cell_obs_str].append((start_col, start_row))
        cell_to_obs[(start_col, start_row)] = start_cell_obs_str

        return obs_str_idxs_map, cell_obs_map, cell_to_obs

    def _get_cell_str(self, obj_type_str: str, obs_str_idxs_map: bidict,
                      only_one_type_of_obj: bool = True) -> {str, List[str]}:
        """
        Gets the observation string(s) associated with each type of object

        :param      obj_type_str:          The object type string
        :param      obs_str_idxs_map:      mapping from cell obs.
                                           string -> cell obs. array
        :param      only_one_type_of_obj:  Whether or not there should only be
                                           one distinct version of this object
                                           in the environment

        :returns:   The cell observation string(s) associated with the object

        :raises     AssertionError:        obj_type_str must be in the ENUM.
        :raises     ValueError:            if there is more than one of an
                                           object when there should only be one
                                           in the env.
        """

        assert obj_type_str in OBJECT_TO_IDX.keys()

        cell_str = [obs_str for obs_str in list(obs_str_idxs_map.keys())
                    if obj_type_str in obs_str]

        if only_one_type_of_obj and len(cell_str) > 1:
            msg = f'there should be exactly one observation string ' + \
                  f'for a {obj_type_str} object. Found {cell_str} in ' + \
                  f'cell observations.'
            raise ValueError(msg)
        elif only_one_type_of_obj and len(cell_str) == 0:
            msg = f'could not find any {obj_type_str} objects.'
            warnings.warn(msg, RuntimeWarning)
            return None
        else:
            cell_str = cell_str[0]

        return cell_str

    def _get_state_str(self, state: Tuple[AgentPos, AgentDir]) -> str:
        """
        Gets the string label for the automaton state given the agent's state.

        :param      pos:        The agent's position
        :param      direction:  The agent's direction

        :returns:   The state label string.
        """

        pos, direction = state

        return ', '.join([str(pos), self.DIR_TO_STRING[direction]])

    def _get_state_from_str(self, state: str) -> Tuple[AgentPos, AgentDir]:
        """
        Gets the agent's state components from the state string representation

        :param      state:  The state label string

        :returns:   the agent's grid cell position, the agent's direction index
        """

        m = re.match(r'\(([\d]*), ([\d]*)\), ([a-z]*)', state)

        pos = (int(m.group(1)), int(m.group(2)))
        direction = self.DIR_TO_STRING.inv[m.group(3)]

        return pos, direction

    def _get_state_obs_from_state_str(self, state: str) -> CellObs:
        """
        Return the cell observation at the given cell from the cell's
        state string

        :param      state:  The state string to get the obs. array from

        :returns:   The state obs from state string.
        """

        agent_pos, _ = self._get_state_from_str(state)
        cell_obs_str = self.cell_to_obs[agent_pos]
        cell_obs_arr = self.obs_str_idxs_map[cell_obs_str]

        return cell_obs_arr

    def _get_state_obs_color(self, state: str) -> str:

        cell_obs_arr = self._get_state_obs_from_state_str(state)

        return IDX_TO_COLOR[cell_obs_arr[1]]

    def _add_node(self, nodes: dict, pos: AgentPos,
                  direction: AgentPos, obs_str: str) -> Tuple[dict, str]:
        """
        Adds a node to the dict of nodes used to initialize an automaton obj.

        :param      nodes:             dict of nodes to build the automaton out
                                       of. Must be in the format needed by
                                       networkx.add_nodes_from()
        :param      pos:               The agent's position
        :param      direction:         The agent's direction
        :param      obs_str:           The state observation string

        :returns:   (updated dict of nodes, new label for the added node)
        """

        state = self._get_state_str((pos, direction))
        color = self._get_state_obs_color(state)
        empty_cell_str = self._get_cell_str('empty', self.obs_str_idxs_map)

        if obs_str == empty_cell_str:
            color = 'gray'
        else:
            color = MINIGRID_TO_GRAPHVIZ_COLOR[color]

        goal_cell_str = self._get_cell_str('goal', self.obs_str_idxs_map)
        if goal_cell_str is not None:
            is_goal = obs_str == goal_cell_str
        else:
            is_goal = False

        if state not in nodes:
            state_data = {'trans_distribution': None,
                          'observation': obs_str,
                          'is_accepting': is_goal,
                          'color': color}
            nodes[state] = state_data

        return nodes, state

    def _add_edge(self, nodes: dict, edges: dict,
                  action: ActionsEnum,
                  edge: Minigrid_TSEdge) -> Tuple[dict, dict, str, str]:
        """
        Adds both nodes to the dict of nodes and to the dict of edges used to
        initialize an automaton obj.

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()
        :param      action:              The action taken
        :param      edge:                The edge to add

        :returns:   (updated dict of nodes, updated dict of edges)
        """

        action_str = self.ACTION_ENUM_TO_STR[action]

        (src, dest,
         src_pos, src_dir,
         dest_pos, dest_dir,
         obs_str_src, obs_str_dest) = self._get_edge_components(edge)

        nodes, state_src = self._add_node(nodes, src_pos, src_dir, obs_str_src)
        nodes, state_dest = self._add_node(nodes, dest_pos, dest_dir,
                                           obs_str_dest)

        edge_data = {'symbols': [action_str]}
        edge = {state_dest: edge_data}

        if state_src in edges:
            if state_dest in edges[state_src]:
                existing_edge_data = edges[state_src][state_dest]
                existing_edge_data['symbols'].extend(edge_data['symbols'])
                edges[state_src][state_dest] = existing_edge_data
            else:
                edges[state_src].update(edge)
        else:
            edges[state_src] = edge

        return nodes, edges, state_src, state_dest

    def _get_edge_components(self,
                             edge: Minigrid_TSEdge) -> Minigrid_Edge_Unpacked:
        """
        Parses the edge data structure and returns a tuple of unpacked data

        :param      edge:         The edge to unpack

        :returns:   All edge components. Not going to name them all bro-bro
        """

        edge = edge

        src, dest = edge
        src_pos, src_dir = src
        dest_pos, dest_dir = dest

        obs_str_src = self.cell_to_obs[src_pos]
        obs_str_dest = self.cell_to_obs[dest_pos]

        return (src, dest, src_pos, src_dir,
                dest_pos, dest_dir, obs_str_src, obs_str_dest)

    def extract_transition_system(self) -> Tuple[dict, dict]:
        """
        Extracts all data needed to build a transition system representation of
        the environment.

        :returns:   The transition system data.
        """

        self.reset()

        nodes = {}
        edges = {}

        init_state_label = self._get_state_str((self.agent_start_pos,
                                               self.agent_start_dir))

        search_queue = queue.Queue()
        search_queue.put(init_state_label)
        visited = set()
        done_states = set()

        while not search_queue.empty():

            curr_state_label = search_queue.get()
            visited.add(curr_state_label)
            src_state = self._get_state_from_str(curr_state_label)

            for action in self.actions:
                if curr_state_label not in done_states:

                    (dest_state,
                     _,
                     done) = self._make_transition(action, src_state)

                    possible_edge = (src_state, dest_state)

                    (nodes, edges,
                     _,
                     dest_state_label) = self._add_edge(nodes, edges,
                                                        action, possible_edge)

                    # don't want to add outgoing transitions from states that
                    # we know are done to the TS, as these are wasted space
                    if done:
                        done_states.add(dest_state_label)

                        # need to reset after done, to clear the 'done' state
                        self.reset()

                    if dest_state_label not in visited:
                        search_queue.put(dest_state_label)
                        visited.add(dest_state_label)

        # we have moved the agent a bunch, so we should reset it when done
        # extracting all of the data.
        self.reset()

        return self._package_data(nodes, edges)

    def _package_data(self, nodes: dict, edges: dict) -> dict:
        """
        Packages up extracted data from the environment in the format needed by
        automaton constructors

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()

        :returns:   configuration data dictionary
        """

        config_data = {}

        # can directly compute these from the graph data
        symbols = set()
        state_labels = set()
        observations = set()
        for state, edge in edges.items():
            for _, edge_data in edge.items():
                symbols.update(edge_data['symbols'])
                state_labels.add(state)

        for node in nodes.keys():
            observation = nodes[node]['observation']
            observations.add(observation)

        alphabet_size = len(symbols)
        num_states = len(state_labels)
        num_obs = len(observations)

        start_state = self._get_state_str((self.agent_start_pos,
                                          self.agent_start_dir))

        config_data['alphabet_size'] = alphabet_size
        config_data['num_states'] = num_states
        config_data['num_obs'] = num_obs
        config_data['nodes'] = nodes
        config_data['edges'] = edges
        config_data['start_state'] = start_state

        return config_data

    def _toggle_video_recording(self, record_video: {bool, None}=None) -> None:
        """
        Turns on / off the video monitoring for the underlying Minigrid env

        :param      record_video:  setting for environment monitoring.
                                   If not given, will toggle the current video
                                   recording state
        """

        if record_video is None:
            turn_off_video = self.env._video_enabled()
        else:
            turn_off_video = not record_video

        if turn_off_video:
            self.env.video_callable = disable_videos
        else:
            self.env.video_callable = lambda episode_id: True

    def _start_monitor(self, new_monitor_file: bool) -> None:
        """
        (Re)-Starts a the env's monitor wrapper

        :param      new_monitor_file:  whether to create a new Monitor file

        :returns:   basically re-runs the Monitor's __init__ function
        """

        env = self.env

        if new_monitor_file:
            env.videos = []
            env.stats_recorder = None
            env.video_recorder = None
            env.enabled = False
            env.episode_id = 0
            env._monitor_id = None

            self.env._start(self.env.directory,
                            self.env.video_callable,
                            self._force_monitor,
                            self._resume_monitor,
                            self.env.write_upon_reset,
                            self._uid_monitor,
                            self._mode)
        else:

            self.env._start(self.env.directory,
                            self.env.video_callable,
                            self._force_monitor,
                            self._resume_monitor,
                            self.env.write_upon_reset,
                            self._uid_monitor,
                            self._mode)

    def _get_video_path(self) -> str:
        """
        Gets the current video recording's full path.

        :returns:   The video path.
        """

        return self.env.video_recorder.path


class DynamicMinigrid2PGameWrapper(gym.core.Wrapper):

    """
    Convert Multi Agent Minigrid Env to a Dynamic Minigrid Two Player Game.
    Agents other than the main robot are classified as "Environment".
    """
    env: MultiAgentEnvType
    IDX_TO_STATE = dict(zip(STATE_TO_IDX.values(), STATE_TO_IDX.keys()))
    DIR_TO_STRING = bidict({0: 'right', 1: 'down', 2: 'left', 3: 'up'})

    def __init__(
        self,
        env: MultiAgentEnvType,
        seeds: List[int] = [0],
        player_steps: Dict[str, List[int]] = {'sys': [2], 'env': [1]},
        start_player: str = 'sys',
        force_monitor: bool = True,
        monitor_log_location: str = GYM_MONITOR_LOG_DIR_NAME,
    ) -> 'DynamicMinigrid2PGameWrapper':

        self.monitor_log_location = monitor_log_location
        self._force_monitor = force_monitor
        self._resume_monitor = False
        self._uid_monitor = None
        self._mode = None
        self.start_player = start_player
        self._next_player = {'sys': 'env', 'env': 'sys'}
        self.curr_player = None

        # Unwrapped Environment (Already instantiated)

        env.max_steps = math.inf    # prevents from "done=True"

        self.agent_to_player = {a.name: 'sys' if a == env.sys_agent else 'env' for a in env.agents}
        agent_steps = {self.agent_to_player[a.name]: player_steps[self.agent_to_player[a.name]] \
            for a in env.agents}

        # Wrapped Environments
        env = ModifyMultiAgentNumActionsWrapper(env, agent_steps)
        env = ViewSizeWrapper(env, agent_view_size=3)
        # env = FullyObsWrapper(ReseedWrapper(ev, seeds=seeds))
        env = Monitor(env, monitor_log_location,
                      video_callable=False,
                      force=self._force_monitor,
                      resume=self._resume_monitor,
                      uid=self._uid_monitor,
                      mode=self._mode)

        # actually creating the minigrid environment with appropriate wrappers
        super().__init__(env)
        self.reset()

        # Initialize some variables and functions to be used in this class
        # multiactions is expanded in ModifyMultiAgentNumActionsWrapper
        player_to_agent = {}
        player_to_agent['sys'] = [env.sys_agent]
        player_to_agent['env'] = [a for a in env.agents if a != env.sys_agent]
        self.player_actions = {p: np.concatenate([a.multiactions for a in agents], axis=0).tolist()
                                for p, agents in player_to_agent.items()}

    def render_notebook(self, filename: str = None, dpi: int = 300) -> None:
        """
        Wrapper for the env.render() that works in notebooks
        """

        plt.imshow(self.unwrapped.render(mode='rgb_image', tile_size=64),
                   interpolation='bilinear')
        plt.axis('off')
        if filename:
            plt.savefig(filename, dpi=dpi)
        plt.show()

    def reset(self, new_monitor_file: bool = False, **kwargs) -> np.ndarray:
        """
        Wrapper for the reset function that manages the monitor wrapper

        :param      new_monitor_file:  whether to create a new monitor file
        :param      kwargs:            The keywords arguments to pass on to the
                                       next wrapper's reset()

        :returns:   env observation
        """

        self.close()
        self._start_monitor(new_monitor_file)
        observation = self.env.reset(**kwargs)
        self.curr_player = None

        return observation

    def state_only_obs(self, obs: dict) -> EnvObs:
        """
        Extracts only the grid observation from a step() observation

        This command only works for a MiniGridEnv obj, as their obs:
            obs, reward, done, _ = MiniGridEnbv.step()
        is a dict containing the (full/partially) observable grid observation

        :param      obs:  Full observation received from MiniGridEnbv.step()

        :returns:   The grid-only observation
        """

        cell_obs = obs['image']

        return cell_obs

    def state_only_obs_reset(self) -> EnvObs:
        """
        Resets the environment, but returns the grid-only observation

        :returns:   The grid-only observation after reseting
        """

        obs = self.env.reset()

        return self.state_only_obs(obs)

    def state_only_obs_step(self, action: MultiAgentAction) -> StepData:
        """
        step()s the environment, but returns only the grid observation

        This command only works for a MiniGridEnv obj, as their obs:
            obs, reward, done, _ = MiniGridEnbv.step()
        is a dict containing the (full/partially) observable grid observation

        :param      action:  The action to take

        :returns:   Normal step() return data, but with obs being only the grid
        """

        obs, reward, done, _ = self.env.step(action)

        return self.state_only_obs(obs), reward, done, {}

    def _get_agent_props(self, agent: Agent = None) -> Tuple[AgentPos, AgentDir]:
        """
        Gets the agent's position and direction in the base environment
        """

        base_env = self.unwrapped

        if agent is None:
            agent_idx = 0
        else:
            agent_idx = self.unwrapped.agents.index(agent)

        return base_env.agents[agent_idx].state

    def _set_agent_props(self,
                         agent_idx: int,
                         agent: Agent,
                         state=None) -> None:
        """
        Sets the agent's position and direction in the base environment

        :param      position:   The new agent grid position
        :param      direction:  The new agent direction
        """

        base_env = self.unwrapped
        base_env.agents[agent_idx].set_state(state, base_env.grid)

    def get_state(self) -> MultiAgentState:
        if self.curr_player is None:
            curr_player = self.start_player
        else:
            curr_player = self.curr_player

        return curr_player, tuple((self._get_agent_props(agent) for agent in self.unwrapped.agents))

    def _set_state(self, curr_state: MultiAgentState) -> None:
        self.curr_player, curr_state = curr_state

        for i_agent, (agent, state) in enumerate(zip(self.unwrapped.agents, curr_state)):
            self._set_agent_props(i_agent, agent, state)

    def _make_transition(self, action: MultiAgentAction,
                         curr_state: MultiAgentState) -> Tuple[MultiAgentState, Done]:
        """
        Makes a state transition in the environment, assuming the env has state

        :param      action:     The action to take
        :param      pos:        The agent's position
        :param      direction:  The agent's direction

        :returns:   the agent's new state, whether or not step() emitted done
        """
        if isinstance(action, tuple):
            action = list(action)

        self._set_state(curr_state)

        _, reward, done, _ = self.state_only_obs_step(action)
        self.curr_player = self._next_player[self.curr_player]

        next_state = self.get_state()

        return next_state, reward, done

    def _add_node(self, nodes: dict, state: MultiAgentState) -> Tuple[dict, str]:
        """
        Adds a node to the dict of nodes used to initialize an automaton obj.

        :param      nodes:             dict of nodes to build the automaton out
                                       of. Must be in the format needed by
                                       networkx.add_nodes_from()
        :param      pos:               The agent's position
        :param      direction:         The agent's direction
        :param      obs_str:           The state observation string

        :returns:   (updated dict of nodes, new label for the added node)
        """

        node_name = self._get_state_str(state)

        curr_agent, state = state

        # For now, we only implement observation for the system agent
        sys_state = state[0]
        sys_state_pos = sys_state[0]

        obs = self._get_obs_at_state(state)
        obs_str = self._obs_to_prop_str(obs,
                                        sys_state=sys_state_pos)

        if state not in nodes:
            state_data = {'observation': obs_str,
                        #   'is_accepting': is_goal,
                          'player': curr_agent}
            nodes[node_name] = state_data

        return nodes, node_name

    def _add_edge(self, nodes: dict, edges: dict,
                  src_agent: str,
                  actions: Tuple[str, ...],
                  weight: float,
                  edge: Minigrid_TSEdge) -> Tuple[dict, dict, str, str]:
        """
        Adds both nodes to the dict of nodes and to the dict of edges used to
        initialize an automaton obj.

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()
        :param      action:              The action taken
        :param      edge:                The edge to add

        :returns:   (updated dict of nodes, updated dict of edges)
        """
        src_state, dest_state = edge

        nodes, state_src = self._add_node(nodes, src_state)
        nodes, state_dest = self._add_node(nodes, dest_state)

        edge_data = {'symbols': [actions], 'weight': [weight]}
        edge = {state_dest: edge_data}

        if state_src in edges:
            if state_dest in edges[state_src]:
                # n_edges = len(edges[state_src][state_dest])
                # edges[state_src][state_dest][n_edges] = edge_data
                existing_edge_data = edges[state_src][state_dest]
                existing_edge_data['symbols'] += [actions]
                existing_edge_data['weight'] += [weight]
                edges[state_src][state_dest] = existing_edge_data
            else:
                edges[state_src][state_dest] = edge_data
        else:
            edges[state_src] = edge

        return nodes, edges, state_src, state_dest

    def _get_state_str(self, state: MultiAgentState) -> Tuple:
        """Translate a state to an interpretable node name"""

        curr_agent, state = state

        state = tuple(a.state_str(s) for a, s in zip(self.agents, state))
        return (curr_agent, ) + state

    def _get_state_from_str(self, state_str: str) -> Tuple[AgentPos, AgentDir]:
        """
        Gets the agent's state components from the state string representation

        :param      state:  The state label string

        :returns:   the agent's grid cell position, the agent's direction index
        """

        state_str = list(state_str)
        curr_agent = state_str[0]
        state = tuple(a.str_to_state(s) for a, s in zip(self.agents, state_str[1:]))

        return curr_agent, state

    def _action_to_str(self, multiaction: MultiAgentAction, src_agent) -> str:
        """
        Convert MultiAgentAction (a list of ActionsEnum) to string.
        It skips if action=None

        :param      action:             A Multi-Agent Action

        :returns:                       A Multi-Agent Action String
        """
        # if src_agent == 'sys':
        #     self.unwrapped.sys_agent.ACTION_ENUM_TO_STR[action]
        # # action = multiaction[0] if src_agent == 'sys' else multiaction[1:]
        # action_strings = []
        # for agent, multistep_action in zip(self.unwrapped.agents, multiaction):
        #     action_str = []
        #     for action in multistep_action:
        #         if action is None or np.isnan(action):
        #             continue
        #         a_str = agent.ACTION_ENUM_TO_STR[action]
        #         action_str.append(a_str)
        #     action_strings.append(tuple(action_str))

        # if src_agent == 'sys':
        #     return action_strings[0]
        # else:
            # return action_strings[1:]
        pass

    def _obs_to_prop_str(self, obs: EnvObs, sys_state) -> str:
        """
        Converts a grid observation array into a string based on Minigrid ENUMs

        :param      obs:      The grid observation
        :param      col_idx:  The col index of the cell to get the obs. string
        :param      row_idx:  The row index of the cell to get the obs. string

        :returns:   verbose, string representation of the state observation
        """
        if isinstance(self.unwrapped.grid, FrankaWorldConfig):
            return obs

        col_idx = sys_state[0]
        row_idx = sys_state[1]

        cell = self.unwrapped.grid.get(col_idx, row_idx)
        if cell is None:
            return set()
        elif isinstance(cell, BaseWorldObj):
            cell = Cell([cell])

        encoded_cell = obs[col_idx, row_idx]
        n_obj, n_info = encoded_cell.shape
        # obj_type, obj_color, obj_state = obs[col_idx, row_idx]

        prop_strings = set()

        # for i_obj in range(n_obj):
        for obj in cell.objs:
            if  obj == self.unwrapped.sys_agent:
                continue

            if isinstance(obj, Agent):
                prop_str = obj.encode_as_str()
            else:
                obj_type, obj_color, obj_state = obj.encode()

                prop_string_base = '_'.join([IDX_TO_OBJECT[obj_type],
                                            IDX_TO_COLOR[obj_color]])
                prop_str = '_'.join([prop_string_base, self.IDX_TO_STATE[obj_state]])

            prop_strings.add(prop_str)

        return '__'.join(prop_strings)
        # return prop_strings

    def _get_obs_at_state(self, curr_state: MultiAgentState) -> EnvObs:
        """
        Makes a state transition in the environment, assuming the env has state

        :param      action:     The action to take
        :param      pos:        The agent's position
        :param      direction:  The agent's direction

        :returns:   the agent's new state, whether or not step() emitted done
        """
        # Keep current state
        temp_state = (self._get_agent_props(agent) for agent in self.unwrapped.agents)

        # Update the grid using the provided curr_state
        for i_agent, (agent, agent_state) in enumerate(zip(self.unwrapped.agents, curr_state)):
            self._set_agent_props(i_agent, agent, agent_state)

        # Get the observation
        # grid, _ = self.env.gen_obs_grid()
        obs = self.unwrapped.grid.encode()

        # Put the state back
        for i_agent, (agent, state) in enumerate(zip(self.unwrapped.agents, temp_state)):
            self._set_agent_props(i_agent, agent, state)

        return obs

    def extract_transition_system(self, n_step: int=None, wait: bool = True) -> Tuple[dict, dict]:
        """
        Extracts all data needed to build a two player game representation of
        the environment.

        :returns:   The transition system data.
        """
        self.reset()

        nodes = {}
        edges = {}

        src_state = self.get_state()
        src_agent, _ = src_state

        search_queue = queue.Queue()
        search_queue.put((src_agent, src_state, 0))
        visited = set()
        visited.add((src_agent, src_state))

        while not search_queue.empty():

            src_agent, src_state, src_n_step = search_queue.get()
            dest_agent = self._next_player[src_agent]

            if n_step is not None and src_n_step == n_step:
                continue

            for multiactions in self.player_actions[src_agent]:

                (dest_state,
                    reward,
                    done) = self._make_transition(multiactions, src_state)

                if done:
                    self.reset()
                    continue

                # action_strings = []
                # for agent, actions in zip(self.unwrapped.agents, multiactions):
                #     action_string = []
                #     for action in actions:
                #         if action is None or np.isnan(action):
                #             continue
                #         a_str = agent.ACTION_ENUM_TO_STR[action]
                #         action_string.append(a_str)
                #     action_strings.append(tuple(action_string))
                # action_strs = action_strings[0] if src_agent == 'sys' else action_strings[1:]
                action_str = self.ACTION_ENUM_TO_STR[tuple(map(tuple, multiactions))]

                possible_edge = (src_state, dest_state)
                
                # if the agent can not stay in the same cell then do not construct those edges
                if not wait:
                    i = 0 if src_agent == 'sys' else 1
                    if src_state[1][i] == dest_state[1][i]:
                        continue


                (nodes, edges,
                    _,
                    _) = self._add_edge(nodes, edges,
                                        src_agent,
                                        action_str,
                                        reward,
                                        possible_edge)

                if (dest_agent, dest_state) not in visited:
                    search_queue.put((dest_agent, dest_state, src_n_step+1))
                    visited.add((dest_agent, dest_state))

        # we have moved the agent a bunch, so we should reset it when done
        # extracting all of the data.
        self.reset()

        return self._package_data(nodes, edges)

    def _package_data(self, nodes: dict, edges: dict) -> dict:
        """
        Packages up extracted data from the environment in the format needed by
        automaton constructors

        :param      nodes:               dict of nodes to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_nodes_from()
        :param      edges:               dict of edges to build the automaton
                                         out of. Must be in the format needed
                                         by networkx.add_edges_from()

        :returns:   configuration data dictionary
        """

        config_data = {}

        # can directly compute these from the graph data
        actions = set()
        state_labels = set()
        observations = set()
        for src_state, edge in edges.items():
            state_labels.add(src_state)
            for dest_state, edge_data in edge.items():
                actions.update(edge_data['symbols'])

                # # Separate actions depending on the weights
                # weight_to_actions = defaultdict(lambda: [])
                # for a, w in zip(edge_data['symbols'], edge_data['weight']):
                #     weight_to_actions[w].append(a)
                # new_edge_data = {}
                # i = 0
                # for w, a in weight_to_actions.items():
                #     new_edge_data[i] = {'weight': w, 'symbols': a}
                #     i += 1
                # edges[src_state][dest_state] = new_edge_data

        for node in nodes.keys():
            observation = nodes[node]['observation']
            observations.add(observation)

        alphabet_size = len(actions)
        num_states = len(state_labels)
        num_obs = len(observations)

        self.reset()
        start_state = self.get_state()
        node_name = self._get_state_str(start_state)

        config_data['alphabet_size'] = alphabet_size
        config_data['num_states'] = num_states
        config_data['num_obs'] = num_obs
        config_data['nodes'] = nodes
        config_data['edges'] = edges
        config_data['start_state'] = node_name

        return config_data

    def _toggle_video_recording(self, record_video: {bool, None}=None) -> None:
        """
        Turns on / off the video monitoring for the underlying Minigrid env

        :param      record_video:  setting for environment monitoring.
                                   If not given, will toggle the current video
                                   recording state
        """

        if record_video is None:
            turn_off_video = self.env._video_enabled()
        else:
            turn_off_video = not record_video

        if turn_off_video:
            self.env.video_callable = disable_videos
        else:
            self.env.video_callable = lambda episode_id: True

    def _start_monitor(self, new_monitor_file: bool) -> None:
        """
        (Re)-Starts a the env's monitor wrapper

        :param      new_monitor_file:  whether to create a new Monitor file

        :returns:   basically re-runs the Monitor's __init__ function
        """

        env = self.env

        if new_monitor_file:
            env.videos = []
            env.stats_recorder = None
            env.video_recorder = None
            env.enabled = False
            env.episode_id = 0
            env._monitor_id = None

            self.env._start(self.env.directory,
                            self.env.video_callable,
                            self._force_monitor,
                            self._resume_monitor,
                            self.env.write_upon_reset,
                            self._uid_monitor,
                            self._mode)
        else:

            self.env._start(self.env.directory,
                            self.env.video_callable,
                            self._force_monitor,
                            self._resume_monitor,
                            self.env.write_upon_reset,
                            self._uid_monitor,
                            self._mode)

    def _get_video_path(self) -> str:
        """
        Gets the current video recording's full path.

        :returns:   The video path.
        """

        return self.env.video_recorder.path


class ObjType:
    def __init__(self, types):
        self._types = types

    def __eq__(self, other: str):
        if not isinstance(other, str):
            return False
        return other in self._types

    def __str__(self):
        return str(self._types)


class Cell:
    def __init__(self, objs: List[BaseWorldObj]=None):
        self.objs = objs
        self.type = ObjType([obj.type for obj in self.objs])

    def can_overlap(self):
        return all([obj.can_overlap() for obj in self.objs])

    def see_behind(self):
        return all([obj.see_behind() for obj in self.objs])

    def encode(self) -> Tuple[Tuple[int, int, int], ...]:
        """
        Return an array of n objects x 3 tuples (type, color, state)
        """
        return tuple(obj.encode() for obj in self.objs)

    def render(self, img):
        for obj in self.objs:
            obj.render(img)

    def add(self, other: Union[BaseWorldObj, 'Cell']):
        if isinstance(other, BaseWorldObj):
            self.objs += [other]
        else:
            self.objs += other.objs
        self.type = ObjType([obj.type for obj in self.objs])

    def __add__(self, other: Union[BaseWorldObj, 'Cell']):
        if other is None:
            return self

        if isinstance(other, BaseWorldObj):
            return Cell(self.objs + [other])
        else:
            return Cell(self.objs + other.objs)

    def __sub__(self, other: BaseWorldObj):
        if other in self.objs:
            return Cell([obj for obj in self.objs if obj != other])
        else:
            return self

    def __eq__(self, other: Union[BaseWorldObj, 'Cell']):
        if other is None:
            return False

        if isinstance(other, BaseWorldObj):
            return self.objs == [other]
        else:
            return self.objs == other.objs

    def __ne__(self, other: Union[BaseWorldObj, 'Cell']):
        return not self == other

    def __len__(self):
        if self.objs is None:
            return 0
        return len(self.objs)


class MultiObjGrid(Grid):
    def __init__(self, base_grid):
        super().__init__(base_grid.width, base_grid.height)
        self.base_grid = base_grid

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(
                "attempted to get missing private attribute '{}'".format(name)
            )
        return getattr(self.base_grid, name)

    def replace(self, i: int, j: int, v: Cell):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        self.grid[j * self.width + i] = v

    def set(self, i: int, j: int, v: Cell):
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        existing_v = self.get(i, j)
        if existing_v is not None:
            if isinstance(existing_v, BaseWorldObj):
                existing_v = Cell([existing_v])
            v = existing_v + v

        self.replace(i, j, v)

    def get(self, i: int, j: int) -> Cell:
        assert i >= 0 and i < self.width
        assert j >= 0 and j < self.height
        return self.grid[j * self.width + i]

    def update(self, prev_pos, new_pos, obj):
        # Delete the obj from the old cell

        # Get the old cell where the obj exists
        cell = self.get(*prev_pos)
        # Update the cell
        if isinstance(cell, Cell):
            old_cell = cell - obj
        elif cell == obj:
            old_cell = None
        else:
            old_cell = cell
        self.replace(*prev_pos, old_cell)

        # Set the agent in the new cell
        cell = self.get(*new_pos)
        if isinstance(cell, Cell):
            new_cell = cell + obj
        elif isinstance(cell, BaseWorldObj):
            new_cell = Cell([cell, obj])
        else:
            new_cell = obj
        self.replace(*new_pos, new_cell)

    def rotate_left(self):
        """
        Rotate the grid to the left (counter-clockwise)
        """

        grid = MultiObjGrid(Grid(self.height, self.width))

        for i in range(self.width):
            for j in range(self.height):
                v = self.get(i, j)
                grid.set(j, grid.height - 1 - i, v)

        return grid

    def slice(self, topX, topY, width, height):
        """
        Get a subset of the grid
        """

        grid = MultiObjGrid(Grid(width, height))

        for j in range(0, height):
            for i in range(0, width):
                x = topX + i
                y = topY + j

                if x >= 0 and x < self.width and \
                   y >= 0 and y < self.height:
                    v = self.get(x, y)
                else:
                    v = Wall()

                grid.set(i, j, v)

        return grid

    def encode(self, vis_mask=None):
        """
        Produce a compact numpy encoding of the grid
        """

        if vis_mask is None:
            vis_mask = np.ones((self.width, self.height), dtype=bool)

        shape = (self.width, self.height)
        array = np.zeros(shape, dtype=object)

        for i in range(self.width):
            for j in range(self.height):
                if vis_mask[i, j]:
                    cell = self.get(i, j)

                    if cell is None:
                        array[i, j] = np.array([[OBJECT_TO_IDX['empty'], 0, 0]])
                        # array[i, j, 0] = 0
                        # array[i, j, 0] = 0
                    elif isinstance(cell, BaseWorldObj):
                        array[i, j] = np.array([cell.encode()])
                    else:
                        array[i, j] = np.array(cell.encode())

        return array

    @property
    def curr_max_n_obj(self):
        max_n_obj = 0
        for e in self.grid:
            if e is None:
                continue
            if isinstance(e, Cell):
                n_obj = len(e)
            else:
                n_obj = 1
            if n_obj > max_n_obj:
                max_n_obj = n_obj
        return max_n_obj


class NoDirectionAgentGrid(Grid):
    """
    This class overrides the drawing of direction-less agents
    """

    tile_cache = {}

    def __init__(self, width: int, height: int):
        super().__init__(width, height)

    def render(
        self,
        tile_size,
        agent_pos=None,
        agent_dir=None,
        highlight_mask=None
    ):
        """
        Render this grid at a given scale

        NOTE: overridden here to change the tile rendering to be the class' own

        :param r: target renderer object
        :param tile_size: tile size in pixels
        """

        if highlight_mask is None:
            highlight_mask = np.zeros(shape=(self.width, self.height),
                                      dtype=bool)

        # Compute the total grid size
        width_px = self.width * tile_size
        height_px = self.height * tile_size

        img = np.zeros(shape=(height_px, width_px, 3), dtype=np.uint8)

        # Render the grid
        for j in range(0, self.height):
            for i in range(0, self.width):
                cell = self.get(i, j)

                agent_here = np.array_equal(agent_pos, (i, j))

                # CHANGED: Grid.render_tile(...) to self.render_tile(...)
                tile_img = self.render_tile(
                    cell,
                    agent_dir=agent_dir if agent_here else None,
                    highlight=highlight_mask[i, j],
                    tile_size=tile_size
                )

                ymin = j * tile_size
                ymax = (j + 1) * tile_size
                xmin = i * tile_size
                xmax = (i + 1) * tile_size
                img[ymin:ymax, xmin:xmax, :] = tile_img

        return img

    @classmethod
    def render_tile(
        cls,
        obj,
        agent_dir=None,
        highlight=False,
        tile_size=TILE_PIXELS,
        subdivs=3,
        white_background=True
    ):
        """
        Render a tile and cache the result
        """

        # Hash map lookup key for the cache
        key = (agent_dir, highlight, tile_size)
        key = obj.encode() + key if obj else key

        if key in cls.tile_cache:
            return cls.tile_cache[key]

        if white_background:
            img = np.full(shape=(tile_size * subdivs, tile_size * subdivs, 3),
                          fill_value=WHITE,
                          dtype=np.uint8)
        else:
            img = np.zeros(shape=(tile_size * subdivs, tile_size * subdivs, 3),
                           dtype=np.uint8)

        # Draw the grid lines (top and left edges)
        fill_coords(img, point_in_rect(0, 0.031, 0, 1), (100, 100, 100))
        fill_coords(img, point_in_rect(0, 1, 0, 0.031), (100, 100, 100))

        if obj is not None:
            obj.render(img)

        # Overlay the agent on top
        if agent_dir is not None:
            cir_fn = point_in_circle(cx=0.5, cy=0.5, r=0.3)
            fill_coords(img, cir_fn, (255, 0, 0))

        # Highlight the cell if needed
        if highlight:
            if white_background:
                highlight_img(img, alpha=0.5)
            else:
                highlight_img(img)

        # Downsample the image to perform supersampling/anti-aliasing
        img = downsample(img, subdivs)

        # Cache the rendered tile
        cls.tile_cache[key] = img

        return img


class LavaComparison(MiniGridEnv):
    """
    Environment to try comparing with Sheshia paper
    """

    def __init__(
        self,
        width=10,
        height=10,
        agent_start_pos=(3, 5),
        agent_start_dir=0,
        drying_off_task=False,
        path_only_through_water=False,
        second_goal_task=False,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = [(1, 1), (1, 8), (8, 8)]
        self.drying_off_task = drying_off_task
        self.directionless_agent = False
        self.path_only_through_water = path_only_through_water
        self.second_goal_task = second_goal_task

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        if self.second_goal_task:
            self.put_obj(Floor(color='purple'), 8, 1)
        else:
            if self.drying_off_task:
                self.put_obj(Floor(color='green'), 8, 1)
            else:
                self.put_obj(Water(), 8, 1)

        # top left Lava block
        self.put_obj(Lava(), 1, 3)
        self.put_obj(Lava(), 1, 4)
        self.put_obj(Lava(), 2, 3)
        self.put_obj(Lava(), 2, 4)

        # top right Lava block
        self.put_obj(Lava(), 7, 3)
        self.put_obj(Lava(), 7, 4)
        self.put_obj(Lava(), 8, 3)
        self.put_obj(Lava(), 8, 4)

        # bottom left Lava blocking goal
        self.put_obj(Lava(), 1, 7)
        self.put_obj(Lava(), 2, 7)
        self.put_obj(Lava(), 2, 8)

        # place the water
        if self.drying_off_task:
            if self.path_only_through_water:
                # new top left
                self.put_obj(Lava(), 3, 3)
                self.put_obj(Lava(), 1, 2)
                self.put_obj(Lava(), 2, 2)
                self.put_obj(Lava(), 2, 1)

                # new top right
                self.put_obj(Lava(), 6, 3)
                self.put_obj(Lava(), 7, 2)
                self.put_obj(Lava(), 8, 2)
                self.put_obj(Lava(), 7, 1)

            self.put_obj(Water(), 4, 6)
            self.put_obj(Water(), 4, 5)
            self.put_obj(Water(), 4, 4)
            self.put_obj(Water(), 4, 3)
            self.put_obj(Water(), 5, 6)
            self.put_obj(Water(), 5, 5)
            self.put_obj(Water(), 5, 4)
            self.put_obj(Water(), 5, 3)

            # bottom carpet
            self.put_obj(Carpet(), 3, 1)
            self.put_obj(Carpet(), 4, 1)
            self.put_obj(Carpet(), 5, 1)
            self.put_obj(Carpet(), 6, 1)

            # top carpet
            self.put_obj(Carpet(), 3, 8)
            self.put_obj(Carpet(), 4, 8)
            self.put_obj(Carpet(), 5, 8)
            self.put_obj(Carpet(), 6, 8)

        if self.path_only_through_water:
            # opened up bottom right Lava blocking goal
            self.put_obj(Lava(), 6, 7)
        else:
            # bottom right Lava blocking goal
            self.put_obj(Lava(), 8, 7)
            self.put_obj(Lava(), 7, 7)
            self.put_obj(Lava(), 7, 8)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class AlternateLavaComparison(MiniGridEnv):
    """
    Very different Environment to the Seshia Paper to show environmental indep.
    """

    def __init__(
        self,
        narrow=False,
        path_only_through_water=False,
        second_goal_task=False,
    ):

        self.width = 20

        if narrow:
            self.corridor_size = 1
            self.height = 9
        else:
            self.corridor_size = 2
            self.height = 13

        self.agent_start_pos = (2, self.height - 2)
        self.agent_start_dir = 0
        self.num_empty_left_side_cells = 2 * self.corridor_size
        self.path_only_through_water = path_only_through_water

        self.directionless_agent = False

        self.second_goal_task = second_goal_task

        super().__init__(
            width=self.width,
            height=self.height,
            max_steps=4 * self.width * self.height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        corridor_size = self.corridor_size
        wall = 1
        first_empty_col = wall
        first_empty_row = wall
        last_empty_col = self.width - (wall * 2)
        last_empty_row = self.height - (wall * 2)
        num_empty_left_side_cells = self.num_empty_left_side_cells

        # place the water blocks by flooding the whole area
        water_start_row = 2 * wall + corridor_size
        water_end_row = last_empty_row
        water_col_start = first_empty_col + num_empty_left_side_cells + wall
        corridor_base_len = self.width - water_col_start - wall - corridor_size

        for row in range(water_start_row, water_end_row + 1):
            if row < water_end_row - 1:
                self.grid.horz_wall(water_col_start, row,
                                    length=corridor_base_len - 1,
                                    obj_type=Water)
            else:
                self.grid.horz_wall(water_col_start, row,
                                    length=corridor_base_len - 1 - 1,
                                    obj_type=Water)

        # generate the horiz. corridor walls
        water_corridor_bottom_row = (2 * corridor_size) + wall

        middle_wall_length = corridor_base_len - corridor_size - wall
        bottom_wall_length = corridor_base_len - corridor_size - wall

        top_wall_row = water_corridor_bottom_row - corridor_size
        middle_wall_row = top_wall_row + corridor_size + wall
        bottom_wall_row = middle_wall_row + corridor_size + wall

        self.grid.horz_wall(water_col_start, top_wall_row,
                            length=corridor_base_len - 1)
        self.grid.horz_wall(water_col_start + wall, middle_wall_row,
                            length=middle_wall_length - 1)
        self.grid.horz_wall(water_col_start + corridor_size + wall,
                            bottom_wall_row,
                            length=bottom_wall_length - 1)

        # generate the vert. corridor walls
        left_vert_wall_col = water_col_start
        right_vert_wall_col = last_empty_col - corridor_size - wall
        right_vert_wall_length = bottom_wall_row - top_wall_row + 2 * wall

        self.grid.vert_wall(left_vert_wall_col, middle_wall_row)
        self.grid.vert_wall(right_vert_wall_col + wall, top_wall_row,
                            length=right_vert_wall_length - 1)

        # place the carpet square
        # carpet_col = water_col_start + corridor_base_len - 1
        # carpet_row = water_end_row
        carpet_col = water_col_start + corridor_base_len + wall
        carpet_row = middle_wall_row + 1
        self.put_obj(Carpet(), carpet_col, carpet_row)

        # place a recharge square in the bottom-right corner
        # goal_col, goal_row = (carpet_col + 2), carpet_row
        goal_col, goal_row = (carpet_col), water_end_row
        self.put_obj(Floor(color='green'), goal_col, goal_row)

        if self.second_goal_task:
            self.put_obj(Floor(color='purple'), int(goal_col/2), 1)

        # lava blocks
        lava_start_col = first_empty_col + corridor_size
        lava_end_col = left_vert_wall_col - corridor_size
        lava_length = right_vert_wall_length - wall
        lava_start_row = top_wall_row

        for col in range(lava_start_col, lava_end_col):
            self.grid.vert_wall(col, lava_start_row, length=lava_length - 1,
                                obj_type=Lava)

        # blocking the route around the maze if we want to force the agent
        # through water
        if self.path_only_through_water:
            new_lava_start_col = lava_start_col
            new_lava_end_col = left_vert_wall_col
            new_lava_start_row = first_empty_row
            new_lava_end_row = corridor_size
            new_lava_length = new_lava_end_col - new_lava_start_row + 1

            for row in range(new_lava_start_row, new_lava_end_row + 1):
                self.grid.horz_wall(new_lava_start_col, row,
                                    length=new_lava_length - 1, obj_type=Lava)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class MyDistShift(MiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=6,
        height=5,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=3,
        onewaypath=False,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_1_pos = (width - 2, 1)
        self.goal_2_pos = (width - 2, height - 2)
        if onewaypath:
            self.goal_1_pos = (width - 3, 2)

        self.strip2_row = strip2_row

        self.directionless_agent = False

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        # create an empty grid with different types of agents
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_1_pos)
        self.put_obj(Floor(color='purple'), *self.goal_2_pos)

        # Place the lava rows
        for i in range(self.width - 4):
            self.grid.set(2 + i, 1, Lava())
            self.grid.set(2 + i, self.strip2_row, Lava())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to both the green and purple squares"


class MyDistShiftOneWay(MyDistShift):
    def __init__(self):
        super().__init__(onewaypath=True)


class LavaComparison_noDryingOff(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=False)


class LavaComparison_seshia(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=True)


class LavaComparison_SeshiaTwoGoals(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=True,
                         second_goal_task=True)


class LavaComparison_SeshiaOnlyWaterPath(LavaComparison):
    def __init__(self):
        super().__init__(drying_off_task=True, path_only_through_water=True)


class AlternateLavaComparison_AllCorridorsOpen_Wide(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=False, path_only_through_water=False)


class AlternateLavaComparison_TwoGoalsAllCorridorsOpen_Wide(AlternateLavaComparison):
    def __init__(self):
        super().__init__(
            narrow=False,
            path_only_through_water=False,
            second_goal_task=True)


class AlternateLavaComparison_OnlyWaterPath_Wide(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=False, path_only_through_water=True)


class AlternateLavaComparison_TwoGoalsOnlyWaterPath_Wide(AlternateLavaComparison):
    def __init__(self):
        super().__init__(
            narrow=False,
            path_only_through_water=True,
            second_goal_task=True)


class AlternateLavaComparison_AllCorridorsOpen_Narrow(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=True, path_only_through_water=False)


class AlternateLavaComparison_TwoGoalsAllCorridorsOpen_Narrow(AlternateLavaComparison):
    def __init__(self):
        super().__init__(
            narrow=True,
            path_only_through_water=False,
            second_goal_task=True)


class AlternateLavaComparison_OnlyWaterPath_Narrow(AlternateLavaComparison):
    def __init__(self):
        super().__init__(narrow=True, path_only_through_water=True)


class AlternateLavaComparison_TwoGoalsOnlyWaterPath_Narrow(AlternateLavaComparison):
    def __init__(self):
        super().__init__(
            narrow=True,
            path_only_through_water=True,
            second_goal_task=True)


class TwoDifferentPaths(MiniGridEnv):
    """
    Customized environment with two different paths.
    (One has nothing on its way and the other has water on its way)
    """

    def __init__(
        self,
        width=6,
        height=6,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        strip2_row=3
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = (1, height-2)
        self.strip2_row = strip2_row

        self.directionless_agent = False

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):
        # create an empty grid with different types of agents
        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_pos)

        # Place the lava
        self.grid.set(2, 2, Lava())
        self.grid.set(3, 2, Lava())
        self.grid.set(3, 3, Lava())

        # Place the water
        self.grid.set(1, 3, Water())
        self.grid.set(2, 3, Water())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "Reach the green floor"


class FourGrids(MultiAgentMiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=6,
        height=3,
        agent_start_pos_list=[(1, 1), (4, 1)],
        agent_start_dir_list=[0, 0],
        agent_colors=['red', 'blue'],
        directionless_agent=True,
    ):
        self.agent_start_pos_list = agent_start_pos_list
        self.agent_start_dir_list = agent_start_dir_list
        self.agent_colors = agent_colors
        self.goal_1_pos = (width - 2, 1)

        self.directionless_agent = directionless_agent

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_1_pos)

        n_agent = len(self.agent_start_pos_list)

        # TODO: Place the agent
        for i in range(n_agent):
            p = self.agent_start_pos_list[i]
            d = self.agent_start_dir_list[i]
            c = self.agent_colors[i]
            if p is not None:
                self.put_agent(Agent(name=f'agent{i}', color=c, view_size=self.view_size), *p, d, True)
            else:
                self.place_agent()

        self.mission = "get to the green squares"


class ChasingAgent(MultiAgentMiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=6,
        height=4,
        agent_start_pos_list=[(1, 2), (4, 2)],
        agent_start_dir_list=[0, 0],
        agent_colors=['red', 'blue'],
        directionless_agent=True,
    ):
        self.agent_start_pos_list = agent_start_pos_list
        self.agent_start_dir_list = agent_start_dir_list
        self.agent_colors = agent_colors
        self.goal_1_pos = (4, 1)
        # self.goal_1_pos = (width - 2, 1)

        self.directionless_agent = directionless_agent

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_1_pos)
        self.put_obj(Wall(), *(2, 2))
        self.put_obj(Wall(), *(3, 2))

        n_agent = len(self.agent_start_pos_list)

        # TODO: Place the agent
        for i in range(n_agent):
            p = self.agent_start_pos_list[i]
            d = self.agent_start_dir_list[i]
            c = self.agent_colors[i]
            if p is not None:
                sys_agent = True if i == 0 else False
                self.put_agent(Agent(name=f'agent{i}', color=c, view_size=self.view_size),
                               *p, d, sys_agent)
            else:
                self.place_agent()

        self.mission = "get to the green squares"


class ChasingAgentInSquare4by4(MultiAgentMiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=6,
        height=6,
        agent_start_pos_list=[(1, 1), (4, 3)],
        agent_start_dir_list=[0, 0],
        agent_colors=['red', 'blue'],
    ):
        self.agent_start_pos_list = agent_start_pos_list
        self.agent_start_dir_list = agent_start_dir_list
        self.agent_colors = agent_colors
        self.goal_1_pos = (3, 4)
        # self.goal_1_pos = (width - 2, 1)

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_1_pos)
        # self.grid.wall_rect(2, 2, 1, 1)
        self.put_obj(Wall(), *(2, 2))
        self.put_obj(Wall(), *(2, 3))
        self.put_obj(Wall(), *(3, 2))
        self.put_obj(Wall(), *(3, 3))

        # self.put_obj(Wall(), *(4, 4))

        n_agent = len(self.agent_start_pos_list)

        # TODO: Place the agent
        for i in range(n_agent):
            p = self.agent_start_pos_list[i]
            d = self.agent_start_dir_list[i]
            c = self.agent_colors[i]
            if p is not None:
                sys_agent = True if i == 0 else False
                self.put_agent(Agent(name=f'agent{i}', color=c, view_size=self.view_size),
                               *p, d, sys_agent)
            else:
                self.place_agent()

        self.mission = "get to the green squares"


class ChasingAgentInSquare3by3(MultiAgentMiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=5,
        height=5,
        agent_start_pos_list=[(1, 1), (3, 2)],
        agent_start_dir_list=[0, 0],
        agent_colors=['red', 'blue'],
    ):
        self.agent_start_pos_list = agent_start_pos_list
        self.agent_start_dir_list = agent_start_dir_list
        self.agent_colors = agent_colors
        self.goal_1_pos = (2, 3)

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_1_pos)
        # self.grid.wall_rect(2, 2, 1, 1)
        self.put_obj(Wall(), *(2, 2))
        self.put_obj(Wall(), *(3, 3))

        n_agent = len(self.agent_start_pos_list)

        # TODO: Place the agent
        for i in range(n_agent):
            p = self.agent_start_pos_list[i]
            d = self.agent_start_dir_list[i]
            c = self.agent_colors[i]
            if p is not None:
                sys_agent = True if i == 0 else False
                self.put_agent(Agent(name=f'agent{i}', color=c, view_size=self.view_size),
                               *p, d, sys_agent)
            else:
                self.place_agent()

        self.mission = "get to the green squares"


class ChasingAgentIn4Square(MultiAgentMiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=7,
        height=7,
        agent_start_pos_list=[(1, 1), (5, 1)],
        agent_start_dir_list=[0, 0],
        agent_colors=['red', 'blue'],
    ):
        self.agent_start_pos_list = agent_start_pos_list
        self.agent_start_dir_list = agent_start_dir_list
        self.agent_colors = agent_colors
        self.goal_pos = (1, 5)

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_pos)
        # self.grid.wall_rect(2, 2, 1, 1)
        # self.grid.wall_rect(4, 2, 1, 1)
        # self.grid.wall_rect(2, 4, 1, 1)
        # self.grid.wall_rect(4, 4, 1, 1)

        self.put_obj(Lava(), *(2, 2))
        self.put_obj(Lava(), *(4, 2))
        self.put_obj(Lava(), *(2, 4))
        self.put_obj(Lava(), *(4, 4))

        n_agent = len(self.agent_start_pos_list)

        # TODO: Place the agent
        for i in range(n_agent):
            p = self.agent_start_pos_list[i]
            d = self.agent_start_dir_list[i]
            c = self.agent_colors[i]
            if p is not None:
                sys_agent = True if i == 0 else False
                self.put_agent(Agent(name=f'agent{i}', color=c, view_size=self.view_size),
                               *p, d, sys_agent)
            else:
                self.place_agent()

        self.mission = "get to the green squares"


class FishAndShipwreckAvoidAgent(MultiAgentMiniGridEnv):
    """
    Customized distributional shift environment.
    """

    def __init__(
        self,
        width=9,
        height=9,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        # env_agent_start_pos=[(2, 6), (5, 5)],
        env_agent_start_pos=[(2, 6), (7, 7)],
        env_agent_start_dir=[0, 0],
        goal_pos=(6, 1)
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_agent_start_pos = env_agent_start_pos
        self.env_agent_start_dir = env_agent_start_dir
        self.goal_pos = goal_pos

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.put_obj(Wall(), *(4, 1))
        self.put_obj(Wall(), *(4, 3))
        self.put_obj(Wall(), *(4, 5))
        self.put_obj(Wall(), *(4, 7))
        self.put_obj(Wall(), *(4, 4))
        self.put_obj(Wall(), *(1, 4))
        self.put_obj(Wall(), *(3, 4))
        self.put_obj(Wall(), *(5, 4))
        self.put_obj(Wall(), *(7, 4))

        self.put_obj(Wall(), *(5, 1))
        self.put_obj(Wall(), *(6, 2))
        self.put_obj(Wall(), *(6, 6))

        # Place the two goal squares in the bottom-right corner
        self.put_obj(Floor(color='green'), *self.goal_pos)

        # Place the agent
        p = self.agent_start_pos
        d = self.agent_start_dir
        if p is not None:
            self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
        else:
            self.place_agent()

        p = self.env_agent_start_pos[0]
        d = self.env_agent_start_dir[0]
        restricted_positions = [(i, j) for i, j in itertools.product(range(1, 7+1), range(1, 4+1))]
        restricted_positions += [(i, j) for i, j in itertools.product(range(4, 7+1), range(4, 7+1))]
        self.put_agent(
            ConstrainedAgent(
                name=f'EnvAgent0',
                view_size=self.view_size,
                color='purple',
                # restricted_objs=['lava', 'carpet', 'water', 'floor'],
                restricted_positions=restricted_positions
                ),
            *p,
            d,
            False)

        p = self.env_agent_start_pos[1]
        d = self.env_agent_start_dir[1]
        restricted_positions = [(i, j) for i, j in itertools.product(range(1, 7+1), range(1, 4+1))]
        restricted_positions += [(i, j) for i, j in itertools.product(range(1, 4+1), range(4, 7+1))]
        self.put_agent(
            ConstrainedAgent(
                name=f'EnvAgent1',
                view_size=self.view_size,
                color='blue',
                # restricted_objs=['lava', 'carpet', 'water', 'floor'],
                restricted_positions=restricted_positions
                ),
            *p,
            d,
            False)

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class FloodingLava(MultiAgentMiniGridEnv):
    """
    Environment to try comparing with Sheshia paper
    """

    def __init__(
        self,
        width=10,
        height=10,
        agent_start_pos=(3, 5),
        agent_start_dir=0,
        drying_off_task=True,
        path_only_through_water=False,
        second_goal_task=False,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.goal_pos = [(1, 1), (1, 8), (8, 8)]
        self.drying_off_task = drying_off_task
        self.directionless_agent = False
        self.path_only_through_water = path_only_through_water
        self.second_goal_task = second_goal_task

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        if self.second_goal_task:
            self.put_obj(Floor(color='purple'), 8, 1)
        else:
            if self.drying_off_task:
                self.put_obj(Floor(color='green'), 8, 1)
            else:
                self.put_obj(Water(), 8, 1)

        # top left Lava block
        self.put_obj(Lava(), 1, 3)
        self.put_obj(Lava(), 1, 4)
        self.put_obj(Lava(), 2, 3)
        self.put_obj(Lava(), 2, 4)

        # top right Lava block
        self.put_obj(Lava(), 7, 3)
        self.put_obj(Lava(), 7, 4)
        self.put_obj(Lava(), 8, 3)
        self.put_obj(Lava(), 8, 4)

        # bottom left Lava blocking goal
        self.put_obj(Lava(), 1, 7)
        self.put_obj(Lava(), 2, 7)
        self.put_obj(Lava(), 2, 8)

        # place the water
        if self.drying_off_task:
            if self.path_only_through_water:
                # new top left
                self.put_obj(Lava(), 3, 3)
                self.put_obj(Lava(), 1, 2)
                self.put_obj(Lava(), 2, 2)
                self.put_obj(Lava(), 2, 1)

                # new top right
                self.put_obj(Lava(), 6, 3)
                self.put_obj(Lava(), 7, 2)
                self.put_obj(Lava(), 8, 2)
                self.put_obj(Lava(), 7, 1)

            self.put_obj(Water(), 4, 6)
            self.put_obj(Water(), 4, 5)
            self.put_obj(Water(), 4, 4)
            self.put_obj(Water(), 4, 3)
            self.put_obj(Water(), 5, 6)
            self.put_obj(Water(), 5, 5)
            self.put_obj(Water(), 5, 4)
            self.put_obj(Water(), 5, 3)

            # bottom carpet
            self.put_obj(Carpet(), 3, 1)
            self.put_obj(Carpet(), 4, 1)
            self.put_obj(Carpet(), 5, 1)
            self.put_obj(Carpet(), 6, 1)

            # top carpet
            self.put_obj(Carpet(), 3, 8)
            self.put_obj(Carpet(), 4, 8)
            self.put_obj(Carpet(), 5, 8)
            self.put_obj(Carpet(), 6, 8)

        if self.path_only_through_water:
            # opened up bottom right Lava blocking goal
            self.put_obj(Lava(), 6, 7)
        else:
            # bottom right Lava blocking goal
            self.put_obj(Lava(), 8, 7)
            self.put_obj(Lava(), 7, 7)
            self.put_obj(Lava(), 7, 8)

        # Place the agent
        if self.agent_start_pos is not None:
            p = self.agent_start_pos
            d = self.agent_start_dir
            if p is not None:
                self.put_agent(Agent(name='agent', view_size=self.view_size), *p, d, True)
            else:
                self.place_agent()

        self.put_agent(Flood(name='flood1'), 3, 2, 1)

        self.put_agent(Flood(name='flood2'), 4, 2, 1)

        self.put_agent(Flood(name='flood3'), 3, 3, 1)

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class DynamicLava(MultiAgentMiniGridEnv):
    """
    Environment to try comparing with Sheshia paper
    """

    def __init__(
        self,
        width=10,
        height=10,
        agent_start_pos=(3, 5),
        agent_start_dir=0,
        env_agent_start_pos=[(7, 2)],
        env_agent_start_dir=[0],
        # env_agent_start_pos=[(2, 2)],
        # env_agent_start_dir=[0],
        # env_agent_start_pos=[(2, 2), (7, 2)],
        # env_agent_start_dir=[0, 0],
        drying_off_task=True,
        path_only_through_water=False,
        second_goal_task=False,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_agent_start_pos = env_agent_start_pos
        self.env_agent_start_dir = env_agent_start_dir

        self.goal_pos = [(1, 1), (1, 8), (8, 8)]
        self.drying_off_task = drying_off_task
        self.directionless_agent = False
        self.path_only_through_water = path_only_through_water
        self.second_goal_task = second_goal_task

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        if self.directionless_agent:
            self.grid = NoDirectionAgentGrid(width, height)
        else:
            self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        if self.second_goal_task:
            self.put_obj(Floor(color='purple'), 8, 1)
        else:
            if self.drying_off_task:
                self.put_obj(Floor(color='green'), 8, 1)
            else:
                self.put_obj(Water(), 8, 1)

        # top left Lava block
        self.put_obj(Lava(), 1, 3)
        self.put_obj(Lava(), 1, 4)
        self.put_obj(Lava(), 2, 3)
        self.put_obj(Lava(), 2, 4)

        # top right Lava block
        self.put_obj(Lava(), 7, 3)
        self.put_obj(Lava(), 7, 4)
        self.put_obj(Lava(), 8, 3)
        self.put_obj(Lava(), 8, 4)

        # bottom left Lava blocking goal
        self.put_obj(Lava(), 1, 7)
        self.put_obj(Lava(), 2, 7)
        self.put_obj(Lava(), 2, 8)

        # place the water
        if self.drying_off_task:
            if self.path_only_through_water:
                # new top left
                self.put_obj(Lava(), 3, 3)
                self.put_obj(Lava(), 1, 2)
                self.put_obj(Lava(), 2, 2)
                self.put_obj(Lava(), 2, 1)

                # new top right
                self.put_obj(Lava(), 6, 3)
                self.put_obj(Lava(), 7, 2)
                self.put_obj(Lava(), 8, 2)
                self.put_obj(Lava(), 7, 1)

            self.put_obj(Water(), 4, 6)
            self.put_obj(Water(), 4, 5)
            self.put_obj(Water(), 4, 4)
            self.put_obj(Water(), 4, 3)
            self.put_obj(Water(), 5, 6)
            self.put_obj(Water(), 5, 5)
            self.put_obj(Water(), 5, 4)
            self.put_obj(Water(), 5, 3)

            # bottom carpet
            self.put_obj(Carpet(), 3, 1)
            self.put_obj(Carpet(), 4, 1)
            self.put_obj(Carpet(), 5, 1)
            self.put_obj(Carpet(), 6, 1)

            # top carpet
            self.put_obj(Carpet(), 3, 8)
            self.put_obj(Carpet(), 4, 8)
            self.put_obj(Carpet(), 5, 8)
            self.put_obj(Carpet(), 6, 8)

        if self.path_only_through_water:
            # opened up bottom right Lava blocking goal
            self.put_obj(Lava(), 6, 7)
        else:
            # bottom right Lava blocking goal
            self.put_obj(Lava(), 8, 7)
            self.put_obj(Lava(), 7, 7)
            self.put_obj(Lava(), 7, 8)

        # Place the agent
        if self.agent_start_pos is not None:
            p = self.agent_start_pos
            d = self.agent_start_dir
            if p is not None:
                self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
            else:
                self.place_agent()

        for i in range(len(self.env_agent_start_pos)):
            p = self.env_agent_start_pos[i]
            d = self.env_agent_start_dir[i]
            restricted_positions = [(i+1, j+1) for i, j in itertools.product(range(8), range(2, 8))]
            self.put_agent(
                ConstrainedAgent(
                    name=f'EnvAgent{i+1}',
                    view_size=self.view_size,
                    color='blue',
                    restricted_objs=['lava', 'carpet', 'water', 'floor'],
                    restricted_positions=restricted_positions),
                *p,
                d,
                False)

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class ToyCorridorLava(MultiAgentMiniGridEnv):
    """
    Environment to try comparing with Sheshia paper
    """

    def __init__(
        self,
        width=10,
        height=8,
        agent_start_pos=(1, 6),
        agent_start_dir=0,
        env_agent_start_pos=[(3, 2)],
        env_agent_start_dir=[0],
        goal_pos=[(8, 6)],
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_agent_start_pos = env_agent_start_pos
        self.env_agent_start_dir = env_agent_start_dir

        self.goal_pos = goal_pos

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(3, 4, 4, 2)

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        # top left Lava block
        self.put_obj(Lava(), 2, 2)
        self.put_obj(Lava(), 4, 2)
        self.put_obj(Lava(), 6, 2)
        # self.put_obj(Lava(), 6, 3)

        self.put_obj(Water(), 3, 6)
        self.put_obj(Water(), 4, 6)
        self.put_obj(Water(), 5, 6)
        self.put_obj(Water(), 6, 6)

        # bottom carpet
        self.put_obj(Carpet(), 7, 3)
        self.put_obj(Carpet(), 8, 3)

        # Place the agent
        p = self.agent_start_pos
        d = self.agent_start_dir
        if p is not None:
            self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
        else:
            self.place_agent()

        for i in range(len(self.env_agent_start_pos)):
            p = self.env_agent_start_pos[i]
            d = self.env_agent_start_dir[i]
            restricted_positions = [(i+1, j+1) for i, j in itertools.product(range(8), range(3, 8))]
            self.put_agent(
                ConstrainedAgent(
                    name=f'EnvAgent{i+1}',
                    view_size=self.view_size,
                    color='blue',
                    restricted_objs=['lava', 'carpet', 'water', 'floor'],
                    restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class CorridorLava(MultiAgentMiniGridEnv):
    """
    """

    def __init__(
        self,
        width=11,
        height=8,
        agent_start_pos=(1, 6),
        agent_start_dir=0,
        # env_agent_start_pos=[(3, 2), (7, 2)],
        # env_agent_start_dir=[0, 0],
        env_agent_start_pos=[(4, 2)],
        env_agent_start_dir=[0],
        goal_pos=[(9, 6)],
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir
        self.env_agent_start_pos = env_agent_start_pos
        self.env_agent_start_dir = env_agent_start_dir

        self.goal_pos = goal_pos

        super().__init__(
            width=width,
            height=height,
            max_steps=4 * width * height,
            # Set this to True for maximum speed
            see_through_walls=True
        )

    def _gen_grid(self, width, height):

        self.grid = MultiObjGrid(Grid(width, height))

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        self.grid.wall_rect(3, 4, 5, 1)
        self.put_obj(Wall(), *(3, 5))
        self.put_obj(Wall(), *(7, 5))
        self.put_obj(Wall(), *(5, 6))

        # Place a goal square in the bottom-right corner
        for goal_pos in self.goal_pos:
            self.put_obj(Floor(color='green'), *goal_pos)

        self.put_obj(Water(), 3, 6)
        self.put_obj(Water(), 4, 6)
        self.put_obj(Water(), 4, 5)
        self.put_obj(Water(), 5, 5)
        self.put_obj(Water(), 6, 5)
        self.put_obj(Water(), 6, 6)
        self.put_obj(Water(), 7, 6)

        # bottom carpet
        # self.put_obj(Carpet(), 1, 2)
        # self.put_obj(Carpet(), 1, 3)
        # self.put_obj(Carpet(), 1, 4)
        # self.put_obj(Carpet(), 9, 2)
        # self.put_obj(Carpet(), 9, 3)
        # self.put_obj(Carpet(), 9, 4)
        self.put_obj(Carpet(), 9, 5)

        # Lava
        self.put_obj(Lava(), 3, 3)
        self.put_obj(Lava(), 4, 3)
        self.put_obj(Lava(), 5, 3)
        self.put_obj(Lava(), 6, 3)
        self.put_obj(Lava(), 7, 3)

        # Place the agent
        p = self.agent_start_pos
        d = self.agent_start_dir
        if p is not None:
            self.put_agent(Agent(name='SysAgent', view_size=self.view_size), *p, d, True)
        else:
            self.place_agent()

        for i in range(len(self.env_agent_start_pos)):
            p = self.env_agent_start_pos[i]
            d = self.env_agent_start_dir[i]
            # restricted_positions = [(i+1, j+1) for i, j in itertools.product(range(8), range(3, 8))]
            restricted_positions = []
            self.put_agent(
                ConstrainedAgent(
                    name=f'EnvAgent{i+1}',
                    view_size=self.view_size,
                    color='blue',
                    # restricted_objs=['lava', 'carpet', 'water', 'floor'],
                    restricted_objs=['lava', 'carpet', 'floor'],
                    restricted_positions=restricted_positions
                    ),
                *p,
                d,
                False)

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'


class BoxPacking(MultiAgentEnv):
    """
    """

    def __init__(
        self,
        locations: List[str] = ['L0', 'L1', 'L2', 'L3'],
        object_locations: Dict[str, str] = {'o0': 'L1', 'o1': 'L3'},
        target_locations: List[str] = ['L3', 'L2'],
        **kwargs):

        self.world_config_kwargs = {
            'locations': locations,
            'object_locations': object_locations,
            'target_locations': target_locations,
            **kwargs}

        super().__init__(max_steps=np.inf)

    def _gen_grid(self):

        self.grid = FrankaWorldConfig(**self.world_config_kwargs)

        self.put_agent(FrankaAgent(self.grid), sys_agent=True)
        self.put_agent(HumanAgent(self.grid))

        self.mission = 'get to a green goal squares, don"t touch lava, ' + \
                       'must dry off if you get wet'

    @staticmethod
    def locations_to_distance_mappings(locations: Dict[str, np.ndarray]) -> Dict[str, Dict[str, float]]:

        loc_prod = list(itertools.product(locations, locations))
        loc_prod = list(map(list, loc_prod))

        distance_mappings = defaultdict(lambda: {})
        for (u, v) in loc_prod:
            distance_mappings[u][v] = np.linalg.norm(
                locations[u] - locations[v])

        return dict(distance_mappings)


register(
    id='MiniGrid-LavaComparison_noDryingOff-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_noDryingOff'
)

register(
    id='MiniGrid-LavaComparison_seshia-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_seshia'
)

register(
    id='MiniGrid-LavaComparison_SeshiaTwoGoals-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_SeshiaTwoGoals'
)

register(
    id='MiniGrid-LavaComparison_SeshiaOnlyWaterPath-v0',
    entry_point='wombats.systems.minigrid:LavaComparison_SeshiaOnlyWaterPath'
)

register(
    id='MiniGrid-AlternateLavaComparison_AllCorridorsOpen_Wide-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_AllCorridorsOpen_Wide'
)

register(
    id='MiniGrid-AlternateLavaComparison_OnlyWaterPath_Wide-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_OnlyWaterPath_Wide'
)

register(
    id='MiniGrid-AlternateLavaComparison_AllCorridorsOpen_Narrow-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_AllCorridorsOpen_Narrow'
)

register(
    id='MiniGrid-AlternateLavaComparison_OnlyWaterPath_Narrow-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_OnlyWaterPath_Narrow'
)

register(
    id='MiniGrid-AlternateLavaComparison_TwoGoalsAllCorridorsOpen_Wide-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_TwoGoalsAllCorridorsOpen_Wide'
)

register(
    id='MiniGrid-AlternateLavaComparison_TwoGoalsOnlyWaterPath_Wide-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_TwoGoalsOnlyWaterPath_Wide'
)

register(
    id='MiniGrid-AlternateLavaComparison_TwoGoalsAllCorridorsOpen_Narrow-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_TwoGoalsAllCorridorsOpen_Narrow'
)

register(
    id='MiniGrid-AlternateLavaComparison_TwoGoalsOnlyWaterPath_Narrow-v0',
    entry_point='wombats.systems.minigrid:AlternateLavaComparison_TwoGoalsOnlyWaterPath_Narrow'
)

register(
    id='MiniGrid-MyDistShift-v0',
    entry_point='wombats.systems.minigrid:MyDistShift'
)

register(
    id='MiniGrid-MyDistShiftOneWay-v0',
    entry_point='wombats.systems.minigrid:MyDistShiftOneWay'
)

register(
    id='MiniGrid-TwoDifferentPaths-v0',
    entry_point='wombats.systems.minigrid:TwoDifferentPaths'
)

register(
    id='MiniGrid-FourGrids-v0',
    entry_point='wombats.systems.minigrid:FourGrids'
)

register(
    id='MiniGrid-ChasingAgent-v0',
    entry_point='wombats.systems.minigrid:ChasingAgent'
)

register(
    id='MiniGrid-ChasingAgentInSquare4by4-v0',
    entry_point='wombats.systems.minigrid:ChasingAgentInSquare4by4'
)

register(
    id='MiniGrid-ChasingAgentInSquare3by3-v0',
    entry_point='wombats.systems.minigrid:ChasingAgentInSquare3by3'
)

register(
    id='MiniGrid-ChasingAgentIn4Square-v0',
    entry_point='wombats.systems.minigrid:ChasingAgentIn4Square'
)

register(
    id='MiniGrid-DynamicLava-v0',
    entry_point='wombats.systems.minigrid:DynamicLava'
)

register(
    id='MiniGrid-FloodingLava-v0',
    entry_point='wombats.systems.minigrid:FloodingLava'
)

register(
    id='MiniGrid-ToyCorridorLava-v0',
    entry_point='wombats.systems.minigrid:ToyCorridorLava'
)

register(
    id='MiniGrid-CorridorLava-v0',
    entry_point='wombats.systems.minigrid:CorridorLava'
)

register(
    id='MiniGrid-FishAndShipwreckAvoidAgent-v0',
    entry_point='wombats.systems.minigrid:FishAndShipwreckAvoidAgent'
)

register(
    id='MiniGrid-Franka-BoxPacking-v0',
    entry_point='wombats.systems.minigrid:BoxPacking'
)
