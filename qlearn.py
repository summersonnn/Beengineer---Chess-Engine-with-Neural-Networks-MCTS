import torch as T 
import torch.nn as nn
import torch.nn.functional as functional
import torch.optim as optim
import numpy as np 
import random
import minichess

env = minichess.MiniChess()
action_space_size = 4
state_space_size = 64

q_table = np.zeros((state_space_size, action_space_size))

num_episodes = 10000
max_steps_per_episode = 100000

learning_rate = 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001

rewards_all_episodes = []

#Q-Learning
for episode in range(num_episodes):
	state = env.reset()

	done = False
	rewards_current_episode = 0

	for step in range(max_steps_per_episode):

		exploration_rate_threshold = random.uniform(0 ,1)
		#Exploit
		if exploration_rate_threshold > exploration_rate:
			action = np.argmax(q_table[state,:])
			while ( not env.checkIfMoveable(action) ):
				action = env.randommove()
		#Explore
		else:
			action = env.randommove()

		new_state, reward, done = env.step(action)

		#Update Q-table
		q_table[state, action] = q_table[state, action] * (1- learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state,:]))

		state = new_state
		rewards_current_episode += reward


		if done == True:
			break

	#Exploration rate decay
	exploration_rate = min_exploration_rate + (max_exploration_rate - min_exploration_rate) * np.exp(-exploration_decay_rate*episode)

	rewards_all_episodes.append(rewards_current_episode)

#Calculate and print the average reward per thousand episodes
rewards_per_thousand_episodes = np.split(np.array(rewards_all_episodes), num_episodes/1000)
count = 1000
print("*****Average reward per thousand episodes*****\n")
for r in rewards_per_thousand_episodes:
	print(count, ": ", str(sum(r/1000)))
	count += 1000

#Print updates Q-table
print("\n\n********Q-table********\n")
print(q_table)































'''class DeepQNetwork(nn.Module):
	def __init__(self, lr, input_dims, fc1_dims, fc2_dims, n_actions):
		super(DeepQNetwork, self).__init__()
		self.input_dims = input_dims
		self.fc1_dims = fc1_dims
		self.fc2_dims = fc2_dims
		self.n_actions = n_actions
		self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
		self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
		self.fc3 = nn.Linear(self.fc2_dims, self.n_actions)
		self.optimizer = optim.Adam(self.parameters(), lr=lr)
		self.loss = nn.MSELoss()
		self.to(self.device)

	def forward(self, observation):
		state = T.Tensor(observation).to(self.device)
		x = F.relu(self.fc1(state))
		x = F.relu(self.fc2(x))
		actions = self.fc3(x)

		return actions'''











