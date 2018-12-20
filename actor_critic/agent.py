from torch.distributions import Categorical

from actor_critic.rollout import Rollout


class Agent():
    def issue_command(self, net, env):
        action_probs, critic_val = net(env.get_observation())
        dist = Categorical(action_probs)
        action_idx = dist.sample()

        rollout = Rollout(dist.log_prob(action_idx), critic_val)
        actions = env.get_actions()
        return action_idx.item(), actions[action_idx.item()], rollout
