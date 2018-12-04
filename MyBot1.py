import copy, json, math, random
import logging
import torch

import hlt
from hlt import constants
from hlt.positionals import Direction, Position

from experiencebuffer import Experience
from train import (agent, args, buffer, env, get_loss, map_dim,
                   net, optimizer, target_net, writer)


with open('data.json', 'r') as f:
    data = json.load(f)

""" <<<Game Begin>>> """
game = hlt.Game()

# At this point "game" variable is populated with initial map data.
# This is a good place to do computationally expensive start-up pre-processing.
# As soon as you call "ready" function below, the 2 second per turn timer will start.
game.ready("DQN")

# Now that your bot is initialized, save a message to yourself in the log file with some important information.
#   Here, you log here your id, which you can always fetch from the game object by using my_id.
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

go_home = {}
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
    frame_num = game.turn_number + data['num_turns_per_game'] * data['game_num']
    logging.info('Frame number: {}'.format(frame_num))
    for ship in me.get_ships():
        if ship.position == me.shipyard.position:
            go_home[ship.id] = False

        new_env = copy.deepcopy(env)
        new_state = torch.zeros(new_env.map_height, new_env.map_width)

        if go_home[ship.id]:
            destination = me.shipyard.position
            action_idx = 5
        else:
            current_epsilon = max(args.epsilon_end,
                                args.epsilon - frame_num * (args.epsilon - args.epsilon_end) / args.epsilon_decay)
            action_idx, action = agent.issue_command(net, env, current_epsilon)

            if action == constants.DOCK:
                destination = me.shipyard.position
                go_home[ship.id] = True
            else:
                destination = game_map.calculate_destination(ship, action)
        movement = game_map.naive_navigate(ship, destination)
        command_queue.append(ship.move(movement))

        new_state[ship.position.x][ship.position.y] = 0
        new_state[movement[0]][movement[1]] = 1
        # need to update this
        reward = float(ship.halite_amount) if me.shipyard.position == destination else 0.0

        new_env.me_states.append(new_state)
        new_obs = new_env.get_observation()

        is_done = game.turn_number == data['num_turns_per_game']
        experience = Experience(current_obs, torch.tensor(action_idx),
                                torch.tensor(reward), torch.tensor(is_done), new_obs)
        buffer.append(experience)
        writer.add_scalar('epsilon', current_epsilon, frame_num)
        writer.add_scalar('reward', reward, frame_num)

    # can we get the network to decide when to spawn more ships?
    # Don't spawn a ship if you currently have a ship at port, though - the ships will collide.
    if game.turn_number <= data['num_turns_per_game'] // 4 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    if frame_num % args.sync_target == 0:
        logging.info('Syncing network paramters...')
        target_net.load_state_dict(net.state_dict())

    if len(buffer) >= args.replay_start:
        logging.info('Beginning back prop...')

        optimizer.zero_grad()
        batch = buffer.sample(args.batch_size)
        loss = get_loss(batch, net, target_net, args.gamma)
        loss.backward()
        optimizer.step()

        writer.add_scalar('loss', loss.item(), frame_num)
        logging.info('Saving model parameters')
        torch.save(net.state_dict(), 'model/model_{}.pth'.format(data['game_num']))

    # Send your moves back to the game environment, ending this turn.
    game.end_turn(command_queue)
