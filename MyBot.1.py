import math, random
import logging

import hlt
from hlt import constants
from hlt.positionals import Direction, Position

from agent import Agent
from environment import Environment

""" <<<Game Begin>>> """
epsilon = 0.8

agent_map = {}
env = Environment()

# This game object contains the initial game state.
game = hlt.Game()
# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("Exemplifai-fast")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """

go_home = {}
while True:
    # This loop handles each turn of the game. The game object changes every turn, and you refresh that state by
    #   running update_frame().
    game.update_frame()
    # You extract player metadata and the updated map metadata here for convenience.
    me = game.me
    game_map = game.game_map

    env.update_observations(game_map, me)
    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []
    taken_position = {}

    for ship in me.get_ships():
        if ship.id not in agent_map:
            agent_map[ship.id] = Agent(ship, epsilon)
        agent = agent_map[ship.id]

        move = agent.step(env)
        command_queue.append(move)

    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 100 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

