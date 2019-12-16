import torch
import torch.optim as optim
import torch.nn.functional as F
import dqn
import minichess
import qvalues

batch_size = 8 #make this smth like 256 when ready
gamma = 1 #set this to 0.999 or near if you want stochasticity. 1 assumes same action always result in same rewards -> future rewards are NOT discounted
eps_start = 1	#maximum (start) exploration rate
eps_end = 0.01	#minimum exploration rate
eps_decay = 0.001 #higher decay means exploration is LESS probable, exploitation is MORE LIKELY
#target_update = 20	#use this if you're using two networks, one named target network
memory_size = 100000 #memory size to hold each state,action,next_state, reward, terminal tuple
lr = 0.001 #how much to change the model in response to the estimated error each time the model weights are updated
num_episodes = 51
max_steps_per_episode = 301

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
em = minichess.MiniChess(device)	#setting up the environment
strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay) 
agent = dqn.Agent(strategy, device)
memory = dqn.ReplayMemory(memory_size)

policy_net = dqn.DQN().to(device)
#target_net = dqn.DQN().to(device)
#target_net.load_state_dict(policy_net.state_dict())
#target_net.eval()
optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)

steps_per_episode =[]	#counts how many steps played in each episode
for episode in range(num_episodes):
	print(episode)
	em.reset()	#reset the environment to start all over again
	state = em.get_state()	#get the first state from the environment as a tensor 

	for step in range(max_steps_per_episode):
		available_actions = em.calculate_available_actions()	#Deciding the possible actions. Illegal actions are not taken into account
		action = agent.select_action(state, available_actions, policy_net)	#returns an action in tensor format
		reward, terminal = em.take_action(action)	#returns reward and terminal state info in tensor format
		next_state = em.get_state()	#get the new state 
		memory.push(dqn.Experience(state, action, next_state, reward, terminal))	#push to replay memory

		#Returns true if length of the memory is greater than or equal to batch_size
		if memory.can_provide_sample(batch_size):
			experiences = memory.sample(batch_size)	#sample experiences from memory
			states, actions, rewards, next_states = qvalues.extract_tensors(experiences)	#extract them

			current_q_values = qvalues.QValues.get_current(policy_net, states, actions)	#get the current q values to calculate loss afterwards
			# get output for the next state
			next_states = policy_net(states)

			# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q)
			target_q_values = torch.cat(tuple(rewards[i].unsqueeze(0) if experiences[i][4]
							else rewards[i].unsqueeze(0) + gamma * torch.max(next_states[i]).unsqueeze(0)
							for i in range(len(experiences))))
			
			optimizer.zero_grad()	#clear the old gradients. we only focus on this batch. pytorch accumulates gradients in default.
			# returns a new Tensor, detached from the current graph, the result will never require gradient
			target_q_values = target_q_values.detach()
			loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
			
			loss.backward()
			optimizer.step()	#take a step based on the gradients


			state = next_state #go to next state which we calculated earlier

		#If we're in a terminal state, we never step in the terminal state. We end the episode instead.
		#Record the step number.
		if terminal:
			steps_per_episode.append(step)
			#print("Terminal! : " + str(step))
			#print(state)
			break

	if( episode % 10 == 0):
		print("Episode:" + str(episode) + " -Weights are updated!")
		torch.save({ 'episode': episode,
            		'model_state_dict': policy_net.state_dict(),
            		'optimizer_state_dict': optimizer.state_dict(),
            		'loss': loss, }, "pretrained_model/MiniChess trained model" + str(episode) + ".tar"
					)
		print("Steps per episode: " + str(steps_per_episode)+ '\n')	

print("Evren dışı!")




