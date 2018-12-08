import argparse, json, subprocess
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam

from agent import Agent
from dqn import DQN
from environment import Environment
from experience_buffer import ExperienceBuffer


parser = argparse.ArgumentParser()
parser.add_argument('--replay-size', type=int, default=1000,
                    help='number of replays to buffer (default: 1000)')
parser.add_argument('--replay-start', type=int, default=150,
                    help='number of replays to buffer before backprop (default: 150)')
parser.add_argument('--batch-size', type=int, default=100,
                    help='batch size to use for training (default: 100)')
parser.add_argument('--epsilon', type=float, default=0.9,
                    help='epsilon greedy probability (default: 0.9)')
parser.add_argument('--epsilon-end', type=float, default=0.1,
                    help='epsilon greedy probability (default: 0.1)')
parser.add_argument('--epsilon-decay', type=int, default=10**3,
                    help='amount to decay epsilon by each turn (default: 100)')
parser.add_argument('--games', type=int, default=10,
                    help='number of games to play (default: 10)')
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.90,
                    help='learning rate (default: 0.90)')
parser.add_argument('--sync-target', type=int, default=1000,
                    help='number of frames before syncing target net (default: 1000)')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# map size, leaving fixed for now
map_dim = 32
# arbitrarily chosen constants
num_actions = 6
turns = 2

writer = SummaryWriter()

net = DQN(turns, num_actions)
net.to(device)
target_net = DQN(turns, num_actions)
target_net.to(device)

optimizer = Adam(net.parameters(), lr=args.lr)

buffer = ExperienceBuffer(args.replay_size)
agent = Agent()
env = Environment(map_dim, turns)


def train(args):
    data = {
        'game_num': 0,
        'num_turns_per_game': 300 + 25 * map_dim / 8
    }

    for game_num in range(args.games):
        print('Starting game ({})'.format(game_num))
        data['game_num'] = game_num
        with open('data/data.json', 'w') as f:
            json.dump(data, f)

        command = "./halite --replay-directory ./replays/ --width 32 --height 32 --no-timeout --results-as-json".split()
        command.append("python3 MyBot.py")
        command.append("python3 MyBot1.py")

        print(f"Command used:\n{command}")
        results = subprocess.check_output(command)
        results = json.loads(results)
        print(f"Player stats from match:\n{results['stats']}")

        if 'score' not in data or ('score' in data and results['stats']['1']['score'] > data['score']):
            data['score'] = results['stats']['1']['score']
            data['best_game'] = game_num
            print('Saved game ({}) as the best game so far'.format(game_num))

        reward = json.load(open('data/total_reward.json', 'r'))
        writer.add_scalar('episode reward', reward, game_num)
    writer.close()


def get_loss(batch, net, target_net, gamma=0.9):
    "We will calculate the loss here"
    states, actions, rewards, dones, new_states = batch

    state_action_vs = net(states).gather(1, actions.unsqueeze(-1)).squeeze(-1)

    next_states_vs = target_net(new_states).max(1)[0]
    next_states_vs[dones] = 0.0
    next_states_vs = next_states_vs.detach()

    expected_values = next_states_vs * gamma + rewards
    return F.mse_loss(state_action_vs, expected_values)


if __name__ == "__main__":
    train(args)
