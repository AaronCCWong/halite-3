import logging
import torch
from collections import deque

from hlt import constants, Direction


class Environment:
    def __init__(self, height, width, turns):
        self.turns = turns
        self.map_height = height
        self.map_width = width
        self.me_states = deque(maxlen=turns)
        self.other_states = deque(maxlen=turns)
        self.resources = deque(maxlen=turns)
        self._initialize_state()

    def get_observation(self):
        return torch.stack((list(self.me_states) +
                            list(self.other_states) +
                            list(self.resources)))

    def get_actions(self):
        return [Direction.North, Direction.South, Direction.East,
                Direction.West, Direction.Still, constants.DOCK]

    def update_observations(self, game_map, me):
        self.me = me
        self._update_state(game_map)

    def _initialize_state(self):
        for _ in range(self.turns):
            self.me_states.append(torch.zeros(self.map_height, self.map_width))
            self.other_states.append(torch.zeros(self.map_height, self.map_width))
            self.resources.append(torch.zeros(self.map_height, self.map_width))

    def _update_state(self, game_map):
        current_resources = torch.zeros(self.map_height, self.map_width)
        current_state_me = torch.zeros(self.map_height, self.map_width)
        current_state_other = torch.zeros(self.map_height, self.map_width)
        for row_idx, row in enumerate(game_map._cells):
            for col_idx, cell in enumerate(row):
                current_resources[row_idx][col_idx] = cell.halite_amount
                if cell.ship and self.me.has_ship(cell.ship.id):
                    current_state_me[row_idx][col_idx] = 1
                elif cell.ship:
                    current_state_other[row_idx][col_idx] = 1

        self.me_states.append(current_state_me)
        self.other_states.append(current_state_other)
        self.resources.append(current_resources)
