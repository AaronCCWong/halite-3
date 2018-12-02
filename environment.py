from hlt import Direction

class Environment:
    def __init__(self):
        self.game_over = False
        self.agent_moves = []
        self.observations = {}

    def get_observation(self):
        return (self.game_map, self.me)

    def get_actions(self):
        return [Direction.North, Direction.South, Direction.East,
                Direction.West, Direction.Still]

    def update_observations(self, game_map, me):
        self.game_map = game_map
        self.me = me
