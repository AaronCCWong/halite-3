import copy, json, math, random
import logging
import pickle
import torch
import torch.nn.functional as F

from hlt import constants
from hlt.networking import Game
from hlt.positionals import Direction, Position

from actor_critic.train import (agent, args, env, map_dim, net, optimizer, writer)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


with open('data/data.json', 'r') as f:
    data = json.load(f)

if data['game_num'] > 0:
    # model_params = torch.load('model/model_{}.pth'.format(data['best_game']))
    model_params = torch.load('model/model_{}.pth'.format(data['game_num'] - 1))
    net.load_state_dict(model_params)

""" <<<Game Begin>>> """
game = Game()
game.ready("Actor-Critic")
logging.info("Successfully created bot! My Player ID is {}.".format(game.my_id))

all_rewards = []
go_home = {}
losses = []
current_rewards = []
actions = []
""" <<<Game Loop>>> """
while True:
    game.update_frame()
    me = game.me
    game_map = game.game_map

    env.update_observations(game_map, me)

    command_queue = []

    # if game.turn_number == 1:
    #     command_queue.append(me.shipyard.spawn())

    frame_num = game.turn_number + data['num_turns_per_game'] * data['game_num']
    logging.info('Frame number: {}'.format(frame_num))

    for ship in me.get_ships():
        if ship.position == me.shipyard.position:
            go_home[ship.id] = False

        new_env = copy.deepcopy(env)
        new_state = torch.zeros(map_dim, map_dim).to(device)

        if go_home[ship.id] and not game_map[me.shipyard.position].is_occupied:
            destination = me.shipyard.position
            action_idx = 5
        else:
            go_home[ship.id] = False
            action_idx, action, rollout = agent.issue_command(net, env)
            actions.append(rollout)

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
        # if me.shipyard.position == next_pos:
        #     halite_deposit_amount = ship.halite_amount - (0.1 * game_map[ship.position].halite_amount)
        #     reward = -10.0 if halite_deposit_amount <= 100 else halite_deposit_amount
        # elif 0.1 * game_map[ship.position].halite_amount > ship.halite_amount:
        #     reward = 0 - max(0.1 * game_map[ship.position].halite_amount, 100.0)
        # elif action == Direction.Still:
        #     if ship.position == me.shipyard.position:
        #         reward = -10.0
        #     else:
        #         reward = 0.25 * game_map[ship.position].halite_amount
        # elif game_map[next_pos].halite_amount > game_map[ship.position].halite_amount:
        #     reward = 10.0
        # else:
        #     reward = 0 - (0.1 * game_map[ship.position].halite_amount)

        if me.shipyard.position == next_pos:
            hal_amount = ship.halite_amount - (0.1 * game_map[ship.position].halite_amount)
            if hal_amount > 100:
                reward = 5.0
            elif hal_amount > 10:
                reward = 0.5
            else:
                reward = -1.0
        elif action == Direction.Still and game_map[next_pos].halite_amount > 0:
            reward = 1.0
        elif action == Direction.Still and game_map[next_pos].halite_amount == 0:
            reward = -1.0
        elif game_map[next_pos].halite_amount > game_map[ship.position].halite_amount:
            reward = 1.0
        else:
            reward = -1.0
        current_rewards.append(reward)

    if game.turn_number % 100 == 0 and len(actions) > 0:
        all_rewards += current_rewards
        logging.info('Beginning back prop...')

        total_reward = 0
        policy_losses = []
        value_losses = []
        rewards = []
        for r in current_rewards[::-1]:
            total_reward = r + args.gamma * total_reward
            rewards.insert(0, total_reward)
        rewards = torch.tensor(rewards).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        for (log_prob, val), r in zip(actions, rewards):
            reward = r - val.item()
            policy_losses.append(-log_prob * reward)
            value_losses.append(F.smooth_l1_loss(torch.tensor(val.item()).to(device), torch.tensor([r]).to(device)))
        optimizer.zero_grad()

        loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        rewards = []
        actions = []
        logging.info('Saving model parameters')
        torch.save(net.state_dict(), 'model/model_{}.pth'.format(data['game_num']))

    with open('data/total_reward.json', 'w') as f:
        json.dump(sum(all_rewards), f)

    if len(losses) > 0:
        with open('data/avg_loss.json', 'w') as f:
            json.dump(sum(losses) / float(len(losses)), f)

    if game.turn_number <= 100 and me.halite_amount >= constants.SHIP_COST and not game_map[me.shipyard].is_occupied:
        command_queue.append(me.shipyard.spawn())

    game.end_turn(command_queue)
