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
import mcts


PATH_TO_DIRECTORY = "pretrained_model/"
batch_size = 256 
gamma = 1 #set this to 0.999 or near if you want stochasticity. 1 assumes same action always result in same rewards -> future rewards are NOT discounted
eps_start = 1	#maximum (start) exploration rate
eps_end = 0.01	#minimum exploration rate
eps_decay = 0.001 #higher decay means faster reduction of exploration rate
target_update = 5	#how often does target network get updated? (in terms of episode number) This will also be used in creating model files
memory_size = 100000 #memory size to hold each state,action,next_state, reward, terminal tuple
lr = 0.001 #how much to change the model in response to the estimated error each time the model weights are updated
num_episodes = 1
max_steps_per_episode = 1

def train(policy_net, target_net):
	global loss
	steps_per_episode = []	#counts how many steps played in each episode
	for episode in range(past_episodes, num_episodes + past_episodes):
		print("Episode number: " + str(episode))
		em.reset()	#reset the environment to start all over again
		state = em.get_state()	#get the first state from the environment as a tensor 

		for step in range(max_steps_per_episode):
			#print("Humanistic state: " + str(em.get_humanistic_state()))
			print("-----------Training Starts-----------")
			checkedby = em.IsCheck("white")
			available_actions = em.calculate_available_actions("white", False, checkedby)	#Deciding the possible actions. Illegal actions are not taken into account
			mcts.initializeTree(em, "white", 1)
			raise ValueError('-----END OF MCTS-----')
			action = agent.select_action(state, available_actions, policy_net, False)	#returns an action in tensor format
			reward, terminal = em.take_action(action)	#returns reward and terminal state info in tensor format
			next_state = em.get_state()	#get the new state

			#increase their sizes and push to replay memory. Sizes of st and nst have been increased in order to concatenate them in extract_tensors more easily.
			state = state.unsqueeze(0)
			next_state = next_state.unsqueeze(0)
			memory.push(dqn.Experience(state, action, next_state, reward, terminal))	

			#Returns true if length of the memory is greater than or equal to batch_size
			if memory.can_provide_sample(batch_size):
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
								else rewards[i].unsqueeze(0) + gamma * torch.max(next_q_values[i]).unsqueeze(0)
								for i in range(len(experiences))))
				
				#clear the old gradients. we only focus on this batch. pytorch accumulates gradients in default.
				optimizer.zero_grad()	
				# returns a new Tensor, detached from the current graph, the result will never require gradient
				target_q_values = target_q_values.detach()
				
				loss = F.mse_loss(current_q_values, target_q_values)
				loss.backward()
				optimizer.step()	#take a step based on the gradients

			state = next_state.squeeze(0) #go to next state which we calculated earlier

			#If we're in a terminal state, we never step in the terminal state. We end the episode instead.
			#Record the step number.
			if terminal:
				steps_per_episode.append(step)
				#print("Terminal! : " + str(step))
				#print(state)
				break

		#Update target network with weights and biases in the policy network
		#Also create new model files
		if( episode % target_update == 0 ):
			target_net.load_state_dict(policy_net.state_dict())
			torch.save({ 'episode': episode,
	            		'model_state_dict': policy_net.state_dict(),
	            		'optimizer_state_dict': optimizer.state_dict(),
	            		'loss': loss,
	            		'current_step': agent.current_step }, PATH_TO_DIRECTORY + "MiniChess-trained-model" + str(episode) + ".tar"
						)
			print("Episode:" + str(episode) + " -------Weights are updated!")

		steps_per_episode.append(step)
		print("Exploration rate: " + str(agent.tell_me_exploration_rate()))
		print("Steps per episode: " + str(steps_per_episode[-10:])+ '\n')
		print("Average steps: " + str(sum(steps_per_episode) / len(steps_per_episode)))
	return None

def test(policy_net):
	state = em.get_state()	#get the first state from the environment as a tensor
	steps = 0
	while True:
		available_actions = em.calculate_available_actions()	#Deciding the possible actions. Illegal actions are not taken into account
		action = agent.select_action(state, available_actions, policy_net, True)	#returns an action in tensor format
		reward, terminal = em.take_action(action)	#returns reward and terminal state info in tensor format
		print("Humanistic state: " + str(em.get_humanistic_state()))
		steps += 1
		if terminal:
			print("End - Steps: " + str(steps))
			break;
		next_state = em.get_state()	#get the new state 
		state = next_state
	return None

if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#Initialize policy net and environment.
	policy_net = dqn.DQN().to(device)
	em = minichess.MiniChess(device)
	last_trained_model = fileoperations.find_last_edited_file(PATH_TO_DIRECTORY)	#Returns the last_trained model from multiple models.
	
	if (sys.argv[1]) == "train":
		strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay) 
		agent = dqn.Agent(strategy, device)
		memory = dqn.ReplayMemory(memory_size)
		optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
		target_net = dqn.DQN().to(device)
		past_episodes = 0	#how many episode is played before? this variable may be changed in the upcoming if block.
		
		if last_trained_model is not None:
			print("***Last trained model: " + last_trained_model)
			checkpoint = torch.load(last_trained_model, map_location=device)
			policy_net.load_state_dict(checkpoint['model_state_dict'])
			optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
			past_episodes = checkpoint['episode']
			loss = checkpoint['loss']
			agent.current_step = checkpoint['current_step']
			del checkpoint
			#If you don't want to start exploration rate from where its left, delete the previous line.

		#Weights and biases in the target net is same as in policy net. Target net will work in eval mode and will not update the weights (on its own)
		target_net.load_state_dict(policy_net.state_dict())
		target_net.eval()
		train(policy_net, target_net)

	elif (sys.argv[1]) == "test":
		agent = dqn.Agent(None, device)	#strategy is none since epsilon greedy strategy is not required in test mode. We don't explore.

		if last_trained_model is not None:
			print("***Last trained model: " + last_trained_model)
			checkpoint = torch.load(last_trained_model, map_location=device)
			policy_net.load_state_dict(checkpoint['model_state_dict'])
			del checkpoint
		else:
			print("Test cannot be done due to absence of weights file")
			os.kill(os.getpid(), signal.SIGTERM)

		#Policy net should be in eval mode to avoid gradient decent in test mode.
		policy_net.eval()
		test(policy_net)

	#sys.exit() or raise SystemExit doesn't work for some reason. That's the only way I could end the process.
	os.kill(os.getpid(), signal.SIGTERM)





