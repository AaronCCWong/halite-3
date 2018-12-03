import copy, math, random
import logging
import torch

import hlt
from hlt import constants
from hlt.positionals import Direction, Position

from experiencebuffer import Experience
from train import agent, args, buffer, env, get_loss, map_height, map_width, net, optimizer, target_net

""" <<<Game Begin>>> """
game = hlt.Game()

# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("DQN")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

""" <<<Game Loop>>> """
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

    current_obs = env.get_observation()

    for ship in me.get_ships():
        new_env = copy.deepcopy(env)
        new_state = torch.zeros(new_env.map_height, new_env.map_width)
        action_idx, action, move = agent.issue_command(net, env, ship, args.epsilon)
        command_queue.append(move)

        x, y = ship.position.x, ship.position.y
        new_state[x][y] = 0
        new_state[(x + action[0]) % map_height][(y + action[1]) % map_width] = 1
        if me.shipyard.position.x == x + action[0] and me.shipyard.position.y == y + action[1]:
            reward = float(ship.halite_amount)
        else:
            reward = 0.0

        new_env.me_states.append(new_state)
        new_obs = new_env.get_observation()

        experience = Experience(current_obs, torch.tensor(action_idx), torch.tensor(reward), new_obs)
        buffer.append(experience)

    # can we get the network to decide when to spawn more ships?
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= 100 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    if len(buffer) >= args.replay_start:
        logging.info('Beginning back prop...')

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss = get_loss(batch, net, target_net, args.gamma)
        loss.backward()
        optimizer.step()

        logging.info('Saving model parameters')
        torch.save(net.state_dict(), 'model/model.pth')

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)

