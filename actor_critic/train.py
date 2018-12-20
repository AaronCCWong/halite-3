import argparse, json, subprocess, os
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.optim import Adam

from .actor_critic import ActorCritic
from .agent import Agent
from .environment import Environment


parser = argparse.ArgumentParser()
parser.add_argument('--batch-size', type=int, default=64,
                    help='batch size to use for training (default: 64)')
parser.add_argument('--games', type=int, default=10,
                    help='number of games to play (default: 10)')
parser.add_argument('--lr', type=float, default=4e-4,
                    help='learning rate (default: 1e-4)')
parser.add_argument('--gamma', type=float, default=0.90,
                    help='learning rate (default: 0.90)')
args = parser.parse_args()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# map size
map_dim = 10
# arbitrarily chosen constants
num_actions = 6

writer = SummaryWriter()

net = ActorCritic(num_actions)
net.to(device)
optimizer = Adam(net.parameters(), lr=args.lr)

agent = Agent()
env = Environment(map_dim)


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

        command = "./halite --replay-directory ./replays/ --width 10 --height 10 --no-timeout --results-as-json".split()
        command.append("python3 MyBot.py")
        command.append("python3 MyBot2.py")

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
        if os.path.isfile('data/avg_loss.json'):
            avg_loss = json.load(open('data/avg_loss.json', 'r'))
            writer.add_scalar('avg loss', avg_loss, game_num)
    writer.close()


if __name__ == "__main__":
    train(args)
