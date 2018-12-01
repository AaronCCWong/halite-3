#!/usr/bin/env python3
# Python 3.6

# Import the Halite SDK, which will let you interact with the game.
import hlt

# This library contains constant values.
from hlt import constants

# This library contains direction metadata to better interface with the game.
from hlt.positionals import Direction, Position

# This library allows you to generate random numbers.
import math, random

# Logging allows you to save messages for yourself. This is required because the regular STDOUT
#   (print statements) are reserved for the engine-bot communication.
import logging

""" <<<Game Begin>>> """

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

    # A command queue holds all the commands you will run this turn. You build this list up and submit it at the
    #   end of the turn.
    command_queue = []
    taken_position = {}

    for ship in me.get_ships():
        # For each of your ships, move randomly if the ship is on a low halite location or the ship is full.
        #   Else, collect halite.
        if ship.position == me.shipyard.position:
            go_home[ship.id] = False

        if go_home[ship.id] or ship.halite_amount >= 0.9 * constants.MAX_HALITE:
            movement = game_map.naive_navigate(ship, me.shipyard.position)
            command_queue.append(ship.move(movement))
            go_home[ship.id] = True
        elif game_map[ship.position].halite_amount < 10:
            directions = [Direction.North, Direction.South, Direction.East, Direction.West]
            random.shuffle(directions)
            for direction in directions:
                next_pos = (ship.position.x + direction[0], ship.position.y + direction[1])
                if next_pos not in taken_position:
                    break
            if not next_pos:
                command_queue.append(ship.stay_still())
            else:
                movement = game_map.naive_navigate(ship, Position(next_pos[0], next_pos[1]))
                command_queue.append(ship.move(movement))
                taken_position[next_pos] = True
        else:
            command_queue.append(ship.stay_still())

    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 100 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

