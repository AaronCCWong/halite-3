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

if data['game_num'] > 0:
    model_params = torch.load('model/model_{}.pth'.format(data['best_game']))
    net.load_state_dict(model_params)
    target_net.load_state_dict(model_params)

""" <<<Game Begin>>> """
game = hlt.Game()
game.ready("DQN")
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

go_home = {}
""" <<<Game Loop>>> """
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    env.update_observations(game_map, me)

    command_queue = []

    frame_num = game.turn_number + data['num_turns_per_game'] * data['game_num']
    logging.info('Frame number: {}'.format(frame_num))

    current_epsilon = max(args.epsilon_end,
                          args.epsilon - frame_num * (args.epsilon - args.epsilon_end) / args.epsilon_decay)

    rewards = []
    for ship in me.get_ships():
        if ship.position == me.shipyard.position:
            go_home[ship.id] = False

        new_env = copy.deepcopy(env)
        new_state = torch.zeros(map_dim, map_dim)

        if go_home[ship.id] and not game_map[me.shipyard.position].is_occupied:
            destination = me.shipyard.position
            action_idx = 5
        else:
            go_home[ship.id] = False
            env.add_ship_layer(ship)
            action_idx, action = agent.issue_command(net, env, current_epsilon)

            if action == constants.DOCK:
                destination = me.shipyard.position
                go_home[ship.id] = True
            else:
                destination = game_map.calculate_destination(ship, action)
        movement = game_map.naive_navigate(ship, destination)
        command_queue.append(ship.move(movement))

        next_pos = game_map.calculate_next_position(ship, movement)

        new_state[ship.position.x][ship.position.y] = 0
        new_state[next_pos.x][next_pos.y] = 1

        # calculate reward for action
        if me.shipyard.position == next_pos:
            halite_deposit_amount = ship.halite_amount - (0.1 * game_map[ship.position].halite_amount)
            reward = -100.0 if halite_deposit_amount <= 100 else halite_deposit_amount
        elif 0.1 * game_map[ship.position].halite_amount > ship.halite_amount:
            reward = 0 - max(0.1 * game_map[ship.position].halite_amount, 100.0)
        elif game_map[ship.position].halite_amount == 0:
            reward = -100.0
        elif action == Direction.Still:
            reward = 0.25 * game_map[ship.position].halite_amount
        else:
            reward = 0 - (0.1 * game_map[ship.position].halite_amount)
        rewards.append(reward)

        new_env.me_states.append(new_state)
        new_obs = new_env.get_observation()

        # save experience into buffer
        is_done = game.turn_number == data['num_turns_per_game']
        current_obs = env.get_observation()
        experience = Experience(current_obs, torch.tensor(action_idx),
                                torch.tensor(reward), torch.tensor(is_done), new_obs)
        buffer.append(experience)

    if len(rewards) > 0:
        writer.add_scalar('epsilon', current_epsilon, frame_num)
        writer.add_scalar('reward', sum(rewards) / float(len(rewards)), frame_num)

    # can we get the network to decide when to spawn more ships?
    if game.turn_number <= data['num_turns_per_game'] // 4 and \
       me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
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

    game.end_turn(command_queue)
