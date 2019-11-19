import torch
import torch.optim as optim
import dqn
import minichess
import qvalues

batch_size = 256
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 10
memory_size = 100000
lr = 0.001
num_episodes = 100
max_steps_per_episode = 100000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = minichess.MiniChess(device)
strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = dqn.Agent(strategy, device)
memory = dqn.ReplayMemory(memory_size)

policy_net = dqn.DQN().to(device)
target_net = dqn.DQN().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

steps_per_episode =[]
for episode in range(num_episodes):
	em.reset()
	state = em.get_state()

	for step in range(max_steps_per_episode):
		available_actions = em.calculate_available_actions()
		action = agent.select_action(state, available_actions, policy_net)
		reward = em.take_action(action)
		next_state = em.get_state()
		'''print("State:" + str(state))
		print("Action:" + str(action))
		print("Next State:" + str(next_state))
		print("Reward:" + str(reward) + "\n\n")'''
		memory.push(dqn.Experience(state, action, next_state, reward))
		state = next_state

		if memory.can_provide_sample(batch_size):
			experiences = memory.sample(batch_size)
			states, actions, rewards, next_states = qvalues.extract_tensors(experiences)

			current_q_values = qvalues.QValues.get_current(policy_net, states, actions)
			next_q_values = qvalues.Qvalues.get_next(target_net, next_states)
			target_q_values = (next_q_values * gamma) + rewards

			loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()

		if em.done:
			steps_per_episode.append(step)
			break

	if episode % target_update == 0:
		target_net.load_state_dict(policy_net.state_dict())





