import os
import signal
import sys
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import dqn
import minichess
import unrelatedmethods
import fileoperations


PATH_TO_DIRECTORY = "pretrained_model/"
batch_size = 4 #make this smth like 256 when ready
gamma = 1 #set this to 0.999 or near if you want stochasticity. 1 assumes same action always result in same rewards -> future rewards are NOT discounted
eps_start = 1	#maximum (start) exploration rate
eps_end = 0.01	#minimum exploration rate
eps_decay = 0.001 #higher decay means exploration is LESS probable, exploitation is MORE LIKELY
target_update = 20	#how often does target network get updated? (in terms of episode number) This will also be used in creating model files
memory_size = 100000 #memory size to hold each state,action,next_state, reward, terminal tuple
lr = 0.001 #how much to change the model in response to the estimated error each time the model weights are updated
num_episodes = 501
max_steps_per_episode = 301

def train(policy_net, target_net):
	em = minichess.MiniChess(device)	#setting up the environment
	strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay) 
	agent = dqn.Agent(strategy, device)
	memory = dqn.ReplayMemory(memory_size)

	optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
	past_episodes = 0	#how many episode is played before? this variable may be changed in the upcoming if block.
	last_trained_model = fileoperations.find_last_edited_file(PATH_TO_DIRECTORY)	#Returns the last_trained model from multiple models.

	if last_trained_model is not None:
		print("***Last trained model: " + last_trained_model)
		checkpoint = torch.load(last_trained_model, map_location=device)
		policy_net.load_state_dict(checkpoint['model_state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
		past_episodes = checkpoint['episode']
		loss = checkpoint['loss']
		
	#Weights and biases in the target net is same as in policy net
	target_net.load_state_dict(policy_net.state_dict())

	steps_per_episode =[]	#counts how many steps played in each episode
	losses = []
	flag = True
	for episode in range(past_episodes + 1, num_episodes + past_episodes):
		print("Episode number: " + str(episode))
		em.reset()	#reset the environment to start all over again
		state = em.get_state()	#get the first state from the environment as a tensor 

		for step in range(1, max_steps_per_episode):
			available_actions = em.calculate_available_actions()	#Deciding the possible actions. Illegal actions are not taken into account
			action = agent.select_action(state, available_actions, policy_net)	#returns an action in tensor format
			reward, terminal = em.take_action(action)	#returns reward and terminal state info in tensor format
			next_state = em.get_state()	#get the new state 
			memory.push(dqn.Experience(state, action, next_state, reward, terminal))	#push to replay memory

			#Returns true if length of the memory is greater than or equal to batch_size
			if memory.can_provide_sample(batch_size):
				if ( flag ):
					print("----------BATCHING-------")
					flag = False
				experiences = memory.sample(batch_size)	#sample experiences from memory
				states, actions, rewards, next_states = unrelatedmethods.extract_tensors(experiences)	#extract them

				#get the current q values to calculate loss afterwards
				current_q_values = policy_net(states).gather(dim=-1, index=actions.unsqueeze(-1))
				# get output(q-values) for the next state. WARNING! Some terminal states may have been passed to target_net. But final states
				# don't have any q-values since there can be no action to take from terminal states. In the upcoming lines, we spot terminal states
				# and don't take their garbage q-values. Instead, we take only immediate rewards. Is this the best approach?
				next_q_values = target_net(next_states)

				# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q). Garbage q-values for terminal states are not used.
				target_q_values = torch.cat(tuple(rewards[i].unsqueeze(0) if experiences[i][4]
								else rewards[i].unsqueeze(0) + gamma * torch.max(next_states[i]).unsqueeze(0)
								for i in range(len(experiences))))
				
				#clear the old gradients. we only focus on this batch. pytorch accumulates gradients in default.
				optimizer.zero_grad()	
				# returns a new Tensor, detached from the current graph, the result will never require gradient
				target_q_values = target_q_values.detach()
				loss = F.mse_loss(current_q_values, target_q_values.unsqueeze(1))
				losses.append(loss)
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

		#Update target network with weights and biases in the policy network
		#Also create new model files
		if( episode % target_update == 0):
			target_net.load_state_dict(policy_net.state_dict())
			torch.save({ 'episode': episode,
	            		'model_state_dict': policy_net.state_dict(),
	            		'optimizer_state_dict': optimizer.state_dict(),
	            		'loss': loss, }, PATH_TO_DIRECTORY + "MiniChess-trained-model" + str(episode) + ".tar"
						)
			print("Episode:" + str(episode) + " -Weights are updated!")

		print("Steps per episode: " + str(steps_per_episode)+ '\n')
	plt.plot(losses)
	plt.ylabel('Losses')
	plt.show()
	return None

if __name__ == '__main__':

	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#Initialize policy and target net. Target net will work in eval mode and will not update the weights (on its own)
	policy_net = dqn.DQN().to(device)
	target_net = dqn.DQN().to(device)
	target_net.eval()

	train(policy_net, target_net)
	#sys.exit() or raise SystemExit doesn't work for some reason. That's the only way I could end the process.
	os.kill(os.getpid(), signal.SIGTERM)





