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
    
    gamma = 0.97
    T = 100  # trajectory length
    num_traj = 1 

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
        '''
        Gets self.num_traj trajectories of length self.T, for each training environment.

        @returns: trajectories. List of dictionaries containing step information.
        '''
        # TODO refactor into one dict containing all relevant info
        trajectories = []
        for i in range(self.num_traj):
            for _ in range(len(self.training_envs)):
                trajectories.append([])
            for t in range(self.T):  # TODO make sure trajectories end after done
                new = self.take_one_step(self.training_envs)
                for env_n, step in enumerate(new): # Append steps from different training environments
                    trajectories[-env_n-1].append(step)

        # Compute reward-to-go to reduce variance
        for i in range(len(trajectories)):
            so_far = 0
            total = sum([step['reward'] for step in trajectories[i]])
            for j in range(len(trajectories[i])):
                # Convert reward of current trajectory to rewards-to-go, and update reward so far
                trajectories[i][j]['reward'], so_far = total - so_far, so_far + trajectories[i][j]['reward']

        return [t for trajectory in trajectories for t in trajectory]  # flatten trajectory list 

    @named_output('entries')
    def take_one_step(self, envs):
        states = [
            e.last_state if hasattr(e, 'last_state') else e.reset()
            for e in envs
        ]
        entries = []
        tensor_states = torch.tensor(states, device=self.compute_device, dtype=torch.float32)
        values, policies = self.model(tensor_states)

        for i, (policy, env) in enumerate(zip(policies, envs)):
            policy = softmax(policy.detach().cpu().numpy())
            action = get_rng().choice(len(policy), p=policy)
            next_state, reward, done, info = env.step(action)
            if done:
                next_state = env.reset()
            env.last_state = next_state

            entries.append({'action': action, 'action_prob': policy[action],
                            'reward': reward, 'state': states[i], 'done': done}) 

        return entries 

    def optimize(self, states, actions, rewards_to_go, report=False):  
        #discount = self.gamma * (1 - done) # TODO ensure gamma used properly
        values, policies = self.model(states)
        action_probs = torch.gather(policies, 1, torch.unsqueeze(actions,1))
        loss = self.compute_loss(action_probs, rewards_to_go) 

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if report and self.data_logger is not None:
            data = {
                "loss": loss.item(),
                "return": rewards_to_go.mean().item(),
            }
            logger.info(
                "n=%i: loss=%0.3g, return=%0.3g", self.num_steps,
                data['loss'], data['return'])
            self.data_logger.log_scalars(data, self.num_steps, 'vpg')

    def compute_loss(self, action_probs, rewards):
        return (torch.log(action_probs) * rewards).mean()

    def train(self, max_steps):
        self.num_steps = 0
        needs_report = True

        while self.num_steps < max_steps:  
            trajectories = self.get_trajectories()  
            rewards = torch.as_tensor([entry['reward'] for entry in trajectories], dtype=torch.float32)
            states = torch.as_tensor([entry['state'] for entry in trajectories], dtype=torch.float32)
            actions = torch.as_tensor([entry['action'] for entry in trajectories], dtype=torch.long)
            # 8: Fit val fn on traj
            
            next_report = round_up(self.num_steps, self.report_interval)
            next_test = round_up(self.num_steps, self.test_interval)

            self.num_steps += self.T * self.num_traj

            if self.num_steps >= next_report:
                needs_report = True

            self.optimize(states, actions, rewards, needs_report)
            needs_report = False

            self.save_checkpoint_if_needed()

            if self.testing_envs and self.num_steps >= next_test:
                self.run_episodes(self.testing_envs)  # TODO: Any param change for test-time?
