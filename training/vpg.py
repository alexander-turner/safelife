import logging
import numpy as np

from torch.distributions.categorical import Categorical
import torch
import torch.optim as optim

from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .base_algo import BaseAlgo
from .utils import named_output, round_up


logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.idx = 0
        self.buffer = np.zeros(capacity, dtype=object)

    def push(self, *data):
        self.buffer[self.idx % self.capacity] = data
        self.idx += 1

    def sample(self, batch_size):
        sub_buffer = self.buffer[:self.idx]
        data = get_rng().choice(sub_buffer, batch_size, replace=False)
        return zip(*data)

    def __len__(self):
        return min(self.idx, self.capacity)


class MultistepReplayBuffer(object):
    def __init__(self, capacity, num_env, n_step, gamma):
        self.capacity = capacity
        self.idx = 0
        self.states = np.zeros(capacity, dtype=object)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.done = np.zeros(capacity, dtype=bool)
        self.num_env = num_env
        self.n_step = n_step
        self.gamma = gamma
        self.tail_length = n_step * num_env

    def push(self, state, action, reward, done):
        idx = self.idx % self.capacity
        self.idx += 1
        self.states[idx] = state
        self.actions[idx] = action
        self.done[idx] = done
        self.rewards[idx] = reward

        # Now discount the reward and add to prior rewards
        n = np.arange(1, self.n_step)
        idx_prior = idx - n * self.num_env
        prior_done = np.cumsum(self.done[idx_prior]) > 0
        gamma = self.gamma**(n) * ~prior_done
        self.rewards[idx_prior] += gamma * reward
        self.done[idx_prior] = prior_done | done

    @named_output("state action reward next_state done")
    def sample(self, batch_size):
        assert self.idx >= batch_size + self.tail_length

        idx = self.idx % self.capacity
        i1 = idx - 1 - get_rng().choice(len(self), batch_size, replace=False)
        i0 = i1 - self.tail_length

        return (
            list(self.states[i0]),  # don't want dtype=object in output
            self.actions[i0],
            self.rewards[i0],
            list(self.states[i1]),  # states n steps later
            self.done[i0],  # whether or not the episode ended before n steps
        )

    def __len__(self):
        return max(min(self.idx, self.capacity) - self.tail_length, 0)


class PolicyGradient(BaseAlgo):
    data_logger = None

    num_steps = 0

    gamma = 0.97
    T = 100  # trajectory length
    num_traj = 50

    training_batch_size = 96
    optimize_interval = 32
    learning_rate = 3e-4

    report_interval = 256
    test_interval = 100000

    compute_device = torch.device('cuda' if USE_CUDA else 'cpu')

    training_envs = None
    testing_envs = None

    checkpoint_attribs = (
        'policy_net', 'target_model', 'optimizer',
        'data_logger.cumulative_stats',
    )

    def __init__(self, policy_net, value_net, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.policy_net = policy_net.to(self.compute_device)
        self.value_net = value_net.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.policy_net.parameters(), lr=self.learning_rate)

        self.load_checkpoint()

    def get_policy(self, state):
        '''
        Get distribution over actions.
        '''
        logits = self.policy_net(state)
        return Categorical(logits=logits)

    def get_action(self, state):
        '''
        Sample an action for the given state. 
        '''
        return self.get_policy(state).sample().obs()

    @named_output('states actions rewards done')
    def take_one_step(self, envs):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in envs
        ]

        actions = map(self.get_action, states) 
        rewards = []
        dones = []

        for env, state, action in zip(envs, states, actions):
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
            env.last_state = next_state

            rewards.append(reward)
            dones.append(done)

        return states, actions, rewards, dones

    def optimize(self, s, a, w report=False):
        state = torch.tensor(s, device=self.compute_device, dtype=torch.float32)
        action = torch.tensor(a, device=self.compute_device, dtype=torch.int64)
        weight = torch.tensor(w, device=self.compute_device, dtype=torch.float32)
        done = torch.tensor(done, device=self.compute_device, dtype=torch.float32)

        #discount = self.gamma * (1 - done) # TODO ensure gamma used properly

        loss = self.compute_loss(state, action, weight) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if report and self.data_logger is not None:
            data = {
                "loss": loss.item(),
                "return": np.mean([weight[0] for weight in weights])
            }
            logger.info(
                "n=%i: loss=%0.3g, return=%0.3g", self.num_steps,
                data['loss'], data['return'])
            self.data_logger.log_scalars(data, self.num_steps, 'vpg')

    def reward_to_go(self, trajectories):
        '''
        Calculate reward-to-gos to reduce gradient variance.

        @param: trajectories. List of lists of (state, action, reward) tuples.
        '''
        rtgs = []
        for trajectory in trajectories:
            rtgs.append(list)
            so_far = 0
            total = sum([reward for (_, _, reward) in trajectory])
            for _, _, reward in trajectory:
                rtgs[-1].append(total - so_far)
                so_far += reward
        return torch.as_tensor(rtgs)
        
    def get_trajectories(self):
        trajectories = []
        for i in range(self.num_traj):
            trajectories.append(list)
            for t in range(self.T):  # TODO make sure trajectories end after done
                trajectories[-1].append(self.take_one_step(self.training_envs))
        return trajectories  # TODO take_one_step returns one per training env

    def compute_loss(self, state, actions, weights):
        logp = self.get_policy(state).log_prob(actions)
        return -(logp * weights).mean()

    '''
    def get_val_gradient(self, trajectories):
        errors = [[(self.value_net(state) - rtg)**2 for state, _, rtg in trajectory] for trajectory in trajectories]
    '''

    def train(self, k):
        needs_report = True
        
        for i in range(k):  # Do k gradient updates
            trajectories = self.get_trajectories()  # list of lists of (s,a,r) tuples TODO just sep lists?
            weights = self.reward_to_go(trajectories)  
            states = torch.as_tensor([state for state, _, _ in trajectory] for trajectory in trajectories])
            actions = torch.as_tensor([action for _, action, _ in trajectory] for trajectory in trajectories])
            
            # 8: Fit val fn on traj

            next_report = round_up(num_steps, self.report_interval)
            next_test = round_up(num_steps, self.test_interval)

            self.num_steps += T * num_traj

            if num_steps >= next_report:
                needs_report = True

            if num_steps >= next_opt:
                self.optimize(needs_report)
                needs_report = False

            self.save_checkpoint_if_needed()

            if self.testing_envs and num_steps >= next_test:
                self.run_episodes(self.testing_envs)  # TODO: Any param change for test-time?
