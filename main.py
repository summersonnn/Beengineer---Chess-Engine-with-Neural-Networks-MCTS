import torch
import torch.optim as optim
import torch.nn.functional as F
import dqn
import minichess
import qvalues

batch_size = 8 #256 idi
gamma = 0.999
eps_start = 1
eps_end = 0.01
eps_decay = 0.001
target_update = 20
memory_size = 100000
lr = 0.001
num_episodes = 50
max_steps_per_episode = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = minichess.MiniChess(device)
strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
agent = dqn.Agent(strategy, device)
memory = dqn.ReplayMemory(memory_size)

policy_net = dqn.DQN().to(device)
#target_net = dqn.DQN().to(device)
#target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

steps_per_episode =[]
for episode in range(num_episodes):
	print("----------NEW EPISODE----------")
	em.reset()
	state = em.get_state()

	for step in range(max_steps_per_episode):
		#print(str(step) +  ". Step starts")
		available_actions = em.calculate_available_actions()
		action = agent.select_action(state, available_actions, policy_net)
		reward, terminal = em.take_action(action)
		next_state = em.get_state()
		memory.push(dqn.Experience(state, action, next_state, reward, terminal))

		if memory.can_provide_sample(batch_size):
			experiences = memory.sample(batch_size)
			states, actions, rewards, next_states = qvalues.extract_tensors(experiences)

			current_q_values = qvalues.QValues.get_current(policy_net, states, actions)
			# get output for the next state
			next_states = policy_net(states)

			# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
			target_q_values = torch.cat(tuple(rewards[i].unsqueeze(0) if experiences[i][4]
							else rewards[i].unsqueeze(0) + gamma * torch.max(next_states[i]).unsqueeze(0)
							for i in range(len(experiences))))
			
			optimizer.zero_grad()
			# returns a new Tensor, detached from the current graph, the result will never require gradient
			target_q_values = target_q_values.detach()
			loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
			
			loss.backward()
			optimizer.step()


			state = next_state

		if terminal:
			steps_per_episode.append(step)
			print("Terminal! : " + str(step))
			print(state)
			break

	
	print("Episode:" + str(episode) + " -Weights are updated!")
	torch.save(policy_net, "pretrained_model/MiniChess trained model" + str(episode) + ".pth")	

	print("Steps per episode: " + str(steps_per_episode)+ '\n')
	#steps_per_episode = []
print("Evren dışı!")




