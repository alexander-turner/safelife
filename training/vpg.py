import logging
import numpy as np

import torch
import torch.optim as optim

from scipy.special import softmax
from safelife.helper_utils import load_kwargs
from safelife.random import get_rng

from .base_algo import BaseAlgo
from .utils import named_output, round_up

logger = logging.getLogger(__name__)
USE_CUDA = torch.cuda.is_available()


class VPG(BaseAlgo):
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
        'model', 'optimizer',
        'data_logger.cumulative_stats',
    )

    def __init__(self, model, **kwargs):
        load_kwargs(self, kwargs)
        assert self.training_envs is not None

        self.model = model.to(self.compute_device)
        self.optimizer = optim.Adam(
            self.model.parameters(), lr=self.learning_rate)

        self.load_checkpoint()
    
    def get_trajectories(self):
        trajectories = []
        for i in range(self.num_traj):
            for _ in range(len(self.training_envs)):
                trajectories.append([])
            for t in range(self.T):  # TODO make sure trajectories end after done
                new = self.take_one_step(self.training_envs)
                for env_n, step in enumerate(zip(*new)): # Unpack steps from different training environments
                    trajectories[-env_n-1].append(step)
        return trajectories 

    @named_output('states actions rewards done')
    def take_one_step(self, envs):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in envs
        ]
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        values, policies = self.model(tensor_states)
        actions = [] 
        rewards = []
        dones = []

        for policy, env in zip(policies, envs):
            policy = softmax(policy.detach().cpu().numpy())
            action = get_rng().choice(len(policy), p=policy)
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
            env.last_state = next_state
            actions.append(action)
            rewards.append(reward)
            dones.append(done)

        return states, actions, rewards, dones

    def optimize(self, states, actions, weights, report=False):  # TODO move batches in here
        done = torch.tensor(done, device=self.compute_device, dtype=torch.float32)

        #discount = self.gamma * (1 - done) # TODO ensure gamma used properly

        loss = self.compute_loss(states, actions, weights) 

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

    def compute_loss(self, state, actions, weights):
        logp = self.get_policy(state).log_prob(actions)
        return -(logp * weights).mean()

    def train(self, max_steps):
        needs_report = True

        while self.num_steps < max_steps:  
            trajectories = self.get_trajectories()  # list of lists of (s,a,r) tuples TODO just sep lists?
            weights = self.reward_to_go(trajectories)  
            states = torch.as_tensor([[state for state, _, _ in trajectory] for trajectory in trajectories])
            actions = torch.as_tensor([[action for _, action, _ in trajectory] for trajectory in trajectories])
            # TODO torch tensor slower for py lists
            # 8: Fit val fn on traj
            
            next_report = round_up(num_steps, self.report_interval)
            next_test = round_up(num_steps, self.test_interval)

            self.num_steps += T * num_traj

            if num_steps >= next_report:
                needs_report = True

            self.optimize(states, actions, weights, needs_report)
            needs_report = False

            self.save_checkpoint_if_needed()

            if self.testing_envs and num_steps >= next_test:
                self.run_episodes(self.testing_envs)  # TODO: Any param change for test-time?
