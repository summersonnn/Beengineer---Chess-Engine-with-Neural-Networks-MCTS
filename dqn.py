import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np 
import random
import math
from collections import namedtuple
import minichess


class DQN(nn.Module):
	def __init__(self,):
		super(DQN, self).__init__()

		self.fc1 = nn.Linear(in_features=2, out_features=12)
		self.fc2 = nn.Linear(in_features=12, out_features=16)
		self.out = nn.Linear(in_features=16, out_features=4)

	def forward(self, t):
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = self.out(t)
		return t

Experience = namedtuple('Experience', ('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory():
	def __init__(self, capacity):
		self.capacity = capacity
		self.memory = []
		self.push_count = 0

	def push(self, experience):
		if len(self.memory) < self.capacity:
			self.memory.append(experience)
		else:
			self.memory[self.push_count % self.capacity] = experience
		self.push_count += 1

	def sample(self, batch_size):
		return random.sample(self.memory, batch_size)

	def can_provide_sample(self, batch_size):
		return len(self.memory) >= batch_size

#ExplorationRate ayarlaması
class EpsilonGreedyStrategy():
	def __init__(self, start, end, decay):
		self.start = start
		self.end = end
		self.decay = decay

	#Exploration rate reduces as steps increase
	def get_exploration_rate(self, current_step):
		return self.end + (self.start - self.end) * math.exp(-1. * current_step * self.decay)

class Agent():
	def __init__(self, strategy, device):
		self.current_step = 0
		self.strategy = strategy #EpsilonGreedyStrategy object
		self.device = device

	def select_action(self, state, available_actions, policy_net):
		rate = self.strategy.get_exploration_rate(self.current_step)
		self.current_step += 1

		#Explore
		if rate > random.random():
			action = random.choice(available_actions) 
			return torch.tensor([action]).to(self.device)

		#Exploit
		else:
			with torch.no_grad():
				tensor_from_net = policy_net(state).to(self.device)  #.argmax()  #exploit
				while (True):
					max_index = tensor_from_net.argmax()
					#If illegal move is given as output by the model, punish that action and make it select an action again.
					if max_index.item() not in available_actions:
						tensor_from_net[max_index] = torch.tensor(-100)
					else:
						break
				return max_index.unsqueeze_(0)
						







