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

		self.fc1 = nn.Linear(in_features=109, out_features=132)
		self.fc2 = nn.Linear(in_features=132, out_features=192)
		self.fc3 = nn.Linear(in_features=192, out_features=216)
		self.out = nn.Linear(in_features=216, out_features=282)

	def forward(self, t):
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = F.relu(self.fc3(t))
		t = self.out(t)
		return t

	'''def __init__(self,):
		super(DQN, self).__init__()

		self.fc1 = nn.Linear(in_features=768, out_features=384)
		self.fc2 = nn.Linear(in_features=384, out_features=192)
		self.fc3 = nn.Linear(in_features=192, out_features=96)
		self.fc4 = nn.Linear(in_features=96, out_features=48)
		self.fc5 = nn.Linear(in_features=48, out_features=24)
		self.out = nn.Linear(in_features=24, out_features=8)
		#In real chess, there will be 3876 possible moves, so probably 3876 neurons at the final layer

	def forward(self, t):
		t = F.relu(self.fc1(t))
		t = F.relu(self.fc2(t))
		t = F.relu(self.fc3(t))
		t = F.relu(self.fc4(t))
		t = F.relu(self.fc5(t))
		t = self.out(t)
		return t'''

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

	def select_action(self, state, available_actions, episode, policy_net, isTest=False):
		if len(available_actions) == 0:
			raise ValueError("Error")

		#In Training, get the exploration ratio
		if not isTest:
			self.current_step = episode
			rate = self.strategy.get_exploration_rate(self.current_step)
			

		#Explore
		if not isTest and rate > random.random():
			action = random.choice(available_actions) 
			return torch.tensor([action]).to(self.device)

		#Exploit
		else:
			with torch.no_grad():
				tensor_from_net = policy_net(state).to(self.device)
				while (True):
					max_index = tensor_from_net.argmax()	#Index of max item is obtained
					#If illegal move is given as output by the model, punish that action and make it select an action again.
					if max_index.item() not in available_actions:
						tensor_from_net[max_index] = torch.add(tensor_from_net[max_index], -100)
					else:
						break
				return max_index.unsqueeze_(0)

	def tell_me_exploration_rate(self):	#debug function to observe exploration rate during training process
		num = self.strategy.get_exploration_rate(self.current_step)
		return "{:.3f}".format(num)

		

						







