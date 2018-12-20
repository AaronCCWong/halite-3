import logging
import torch
from collections import deque

from hlt import constants, Direction


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Environment:
    def __init__(self, map_dim):
        self.map_dim = map_dim

        # feature maps
        self.me_states = deque(maxlen=1)
        self.other_states = deque(maxlen=1)
        self.resources = deque(maxlen=1)
        self.me_depots = torch.zeros(self.map_dim, self.map_dim).to(device)

        # initialize
        self._initialize_state()

    def get_observation(self):
        return torch.stack((list(self.me_states) +
                            list(self.other_states) +
                            [self.me_depots] +
                            list(self.resources)))

    def get_actions(self):
        return [Direction.North, Direction.South, Direction.East,
                Direction.West, Direction.Still, constants.DOCK]

    def update_observations(self, game_map, me):
        self.me = me
        self._update_state(me, game_map)

    def _initialize_state(self):
        self.me_states.append(torch.zeros(self.map_dim, self.map_dim).to(device))
        self.other_states.append(torch.zeros(self.map_dim, self.map_dim).to(device))
        self.resources.append(torch.zeros(self.map_dim, self.map_dim).to(device))

    def _update_state(self, me, game_map):
        current_resources = torch.zeros(self.map_dim, self.map_dim).to(device)
        current_state_me = torch.zeros(self.map_dim, self.map_dim).to(device)
        current_state_other = torch.zeros(self.map_dim, self.map_dim).to(device)
        current_depots_me = torch.zeros(self.map_dim, self.map_dim).to(device)
        for row_idx, row in enumerate(game_map._cells):
            for col_idx, cell in enumerate(row):
                current_resources[row_idx][col_idx] = cell.halite_amount
                if cell.ship and self.me.has_ship(cell.ship.id):
                    current_state_me[row_idx][col_idx] = 1
                elif cell.ship:
                    current_state_other[row_idx][col_idx] = 1

        current_depots_me[me.shipyard.position.x][me.shipyard.position.y] = 1
        for dropoff in me.get_dropoffs():
            current_depots_me[dropoff.position.x][dropoff.position.y] = 1

        self.me_states.append(current_state_me)
        self.other_states.append(current_state_other)
        self.resources.append(current_resources)
        self.me_depots = current_depots_me
