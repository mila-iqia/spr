# -*- coding: utf-8 -*-
from __future__ import division
import os
import numpy as np
import torch
import wandb
from torch import optim

from src.dqn import DQN


class Agent():
    def __init__(self, args, action_space):
        self.action_space = action_space
        self.atoms = args.atoms
        self.Vmin = args.V_min
        self.Vmax = args.V_max
        self.support = torch.linspace(args.V_min, args.V_max, self.atoms).to(device=args.device)  # Support (range) of z
        self.delta_z = (args.V_max - args.V_min) / (self.atoms - 1)
        self.batch_size = args.batch_size
        self.n = args.multi_step
        self.discount = args.discount
        self.target_update = args.target_update
        self.steps = 0

        self.online_net = DQN(args, self.action_space).to(device=args.device)
        if args.model:  # Load pretrained model if provided
            if os.path.isfile(args.model):
                self.online_net.load_state_dict(torch.load(args.model,
                                                           map_location='cpu'))  # Always load tensors onto CPU by default, will shift to GPU if necessary
                print("Loading pretrained model: " + args.model)
            else:  # Raise error if incorrect model path provided
                raise FileNotFoundError(args.model)

        self.online_net.train()

        self.target_net = DQN(args, self.action_space).to(device=args.device)
        self.update_target_net()
        self.target_net.train()
        for param in self.target_net.parameters():
            param.requires_grad = False

        self.optimiser = optim.Adam(self.online_net.parameters(), lr=args.learning_rate, eps=args.adam_eps)

        self.q_losses = []
        self.weighted_q_losses = []
        self.args = args

    def log(self, env_steps):
        q_loss = np.mean(self.q_losses)
        weighted_q_loss = np.mean(self.weighted_q_losses)

        print("Q-Loss: {}, Weighted Q-Loss: {}, Env-steps: {}".format(q_loss,
                                                                      weighted_q_loss,
                                                                      env_steps))
        wandb.log({'Q-Loss': q_loss,
                   'Weighted Q-Loss': weighted_q_loss,
                   'Env-steps': env_steps})

        self.q_losses = []
        self.weighted_q_losses = []

    # Resets noisy weights in all linear layers (of online net only)
    def reset_noise(self):
        self.online_net.reset_noise()

    # Acts based on single state (no batch)
    def act(self, state, batch=False):
        with torch.no_grad():
            if batch:
                return (self.online_net(state) * self.support).sum(2).argmax(1)
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).argmax(1).item()

    # Acts with an ε-greedy policy (used for evaluation only)
    def act_e_greedy(self, state, epsilon=0.001):  # High ε can reduce evaluation scores drastically
        return np.random.randint(0, self.action_space) if np.random.random() < epsilon else self.act(state)

    def act_with_planner(self, state, planner, length=2, shots=100, epsilon=0.001):
        """
        A form of model-based Q bootstrapping inspired by the paper
        Bootstrapping the Expressivity with Model-Based Planning
        https://arxiv.org/pdf/1910.05927.pdf
        :param state: An individual state.
        :param planner: Forward model with predict() method returning state, reward.
        :param length: Number of steps to plan for.
        :param shots: Number of shots per initial action.
        :param epsilon: Epsilon for epsilon-greedy (0=greedy).
        :return: Action selected by bootstrapping.
        """
        if length == 0:
            # Just act, no planning is done.
            return self.act_e_greedy(state, epsilon)

        # If we roll to take a random action, just directly return a random action.
        if np.random.random() < epsilon:
            return np.random.randint(0, self.action_space)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        actions = torch.arange(self.action_space, device=state.device)
        actions = actions[:, None].expand(-1, shots).reshape(-1)
        all_runs = torch.zeros(self.action_space*shots, device=state.device)
        current_state = state.expand(self.action_space*shots, -1)
        continuation_probs = torch.ones_like(all_runs)
        for i in range(length):
            pred_state, reward, nonterminal = planner.predict(current_state, actions, mean_rew=self.args.mean_rew)
            current_state = current_state[:, current_state.shape[-1]//4:]
            current_state = torch.cat([current_state, pred_state], -1)
            all_runs += reward*continuation_probs*self.discount**i
            continuation_probs = continuation_probs*nonterminal

        final_value = (self.online_net(current_state)*self.support).sum(2)
        final_value = torch.max(final_value, -1)[0]
        all_runs = all_runs + self.discount**length * final_value * continuation_probs

        all_runs = all_runs.view(self.action_space, shots)
        best_action = torch.argmax(torch.max(all_runs, -1, keepdim=True)[0]).item()

        return best_action

    def learn(self, mem):
        # Sample transitions
        self.reset_noise()
        idxs, states, actions, returns, next_states, nonterminals, weights = mem.sample(self.batch_size)
        loss = self.update(states, actions, returns, next_states, nonterminals, weights)
        mem.update_priorities(idxs, loss.detach().cpu().numpy())  # Update priorities of sampled transitions

    def update(self,
               states,
               actions,
               returns,
               next_states,
               nonterminals,
               weights,
               step=True,
               n=None,
               target_next_states=None):
        # Calculate current state probabilities (online network noise already sampled)
        log_ps = self.online_net(states, log=True)  # Log probabilities log p(s_t, ·; θonline)
        log_ps_a = log_ps[range(self.batch_size), actions]  # log p(s_t, a_t; θonline)

        if target_next_states is None:
            target_next_states = next_states

        if n is None:
            n = self.n

        with torch.no_grad():
            # Calculate nth next state probabilities
            pns = self.online_net(next_states)  # Probabilities p(s_t+n, ·; θonline)
            dns = self.support.expand_as(pns) * pns  # Distribution d_t+n = (z, p(s_t+n, ·; θonline))
            argmax_indices_ns = dns.sum(2).argmax(
                1)  # Perform argmax action selection using online network: argmax_a[(z, p(s_t+n, a; θonline))]
            self.target_net.reset_noise()  # Sample new target net noise
            pns = self.target_net(target_next_states)  # Probabilities p(s_t+n, ·; θtarget)
            pns_a = pns[range(
                self.batch_size), argmax_indices_ns]  # Double-Q probabilities p(s_t+n, argmax_a[(z, p(s_t+n, a; θonline))]; θtarget)

            # Compute Tz (Bellman operator T applied to z)
            Tz = returns.unsqueeze(1) + nonterminals * (self.discount ** n) * self.support.unsqueeze(
                0)  # Tz = R^n + (γ^n)z (accounting for terminal states)
            Tz = Tz.clamp(min=self.Vmin, max=self.Vmax)  # Clamp between supported values
            # Compute L2 projection of Tz onto fixed support z
            b = (Tz - self.Vmin) / self.delta_z  # b = (Tz - Vmin) / Δz
            l, u = b.floor().to(torch.int64), b.ceil().to(torch.int64)
            # Fix disappearing probability mass when l = b = u (b is int)
            l[(u > 0) * (l == u)] -= 1
            u[(l < (self.atoms - 1)) * (l == u)] += 1

            # Distribute probability of Tz
            m = states.new_zeros(self.batch_size, self.atoms)
            offset = torch.linspace(0, ((self.batch_size - 1) * self.atoms), self.batch_size).unsqueeze(1).expand(
                self.batch_size, self.atoms).to(actions)
            m.view(-1).index_add_(0, (l + offset).view(-1),
                                  (pns_a * (u.float() - b)).view(-1))  # m_l = m_l + p(s_t+n, a*)(u - b)
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                  (pns_a * (b - l.float())).view(-1))  # m_u = m_u + p(s_t+n, a*)(b - l)

        loss = -torch.sum(m * log_ps_a, 1)  # Cross-entropy loss (minimises DKL(m||p(s_t, a_t)))
        if step:
            self.online_net.zero_grad()
            (weights * loss).mean().backward()  # Backpropagate importance-weighted minibatch loss
            self.optimiser.step()
            self.q_losses.append(loss.mean().detach().item())
            self.weighted_q_losses.append((weights * loss).mean().detach().item())

        self.steps += 1
        self.maybe_update_target_net()
        return loss

    def maybe_update_target_net(self):
        if self.steps > self.target_update:
            self.update_target_net()

    def update_target_net(self):
        self.steps = 0
        self.target_net.load_state_dict(self.online_net.state_dict())

    # Save model parameters on current device (don't move model between devices)
    def save(self, path):
        torch.save(self.online_net.state_dict(), os.path.join(path, 'model.pth'))

    # Evaluates Q-value based on single state (no batch)
    def evaluate_q(self, state):
        with torch.no_grad():
            return (self.online_net(state.unsqueeze(0)) * self.support).sum(2).max(1)[0].item()

    def train(self):
        self.online_net.train()

    def eval(self):
        self.online_net.eval()
