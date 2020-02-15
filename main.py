import os
import signal
import sys
from matplotlib import pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import dqn
import minichess
import unrelatedmethods as um
import fileoperations
import mcts
from copy import deepcopy
import timeit

PATH_TO_DIRECTORY = "pretrained_model/"
batch_size = 512 
gamma = 1 		#1 assumes same action always result in same rewards -> future rewards are NOT discounted
eps_start = 1	#maximum (start) exploration rate
eps_end = 0.35	#minimum exploration rate
eps_decay = 0.005 #higher decay means faster reduction of exploration rate
target_update = 5	#how often does target network get updated? (in terms of episode number) This will also be used in creating model files
memory_size = 15000 #memory size to hold each state,action,next_state, reward, terminal tuple
per_game_memory_size = 60 #Assuming  players will make 100 moves at most per game (includes both sides)
lr = 0.001 #how much to change the model in response to the estimated error each time the model weights are updated
move_time = 0.1	#Thinking time of a player

def train(policy_net, target_net):
	whiteWins = 0
	blackWins = 0
	drawByNoProgress = 0
	drawByTooLongGame = 0
	drawByStaleMate = 0
	move_count = 0
	global loss
	global em

	for episode in range(past_episodes + 1, num_episodes + past_episodes + 1):
		print("Episode number: " + str(episode))
		print("Exploration Rate: " + agent.tell_me_exploration_rate())
		terminal = False
		em.reset()	#reset the environment to start all over again
		tempMemory = dqn.ReplayMemory(per_game_memory_size) #Create tempMemory for one match
		

		#Calculating available actions for just once, to initiate sequence
		em.calculate_available_actions("white")
		
		while True:
			state = em.get_state()	#get the BitVectorBoard state from the environment as a tensor 
			
			#If the game didn't end with the last move, now it's white's turn to move
			if not terminal: 
				em, action = mcts.initializeTree(em, "white", move_time, episode, policy_net, agent, device)	#white makes his move
				next_state = em.get_state()

				#We don't know what the reward will be until the game ends. So put 0 for now.
				state = state.unsqueeze(0)
				next_state = next_state.unsqueeze(0)
				gem = deepcopy(em)
				next_state_av_acts = gem.calculate_available_actions("black")
				tempMemory.push(dqn.Experience(state, action, next_state, next_state_av_acts, 0, False))
				state = next_state

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory, True)
			
			#If the game didn't end with the last move, now it's black's turn to move
			if not terminal: 
				em, action = mcts.initializeTree(em, "black", move_time, episode, policy_net, agent, device)	#white makes his move
				next_state = em.get_state()
				gem = deepcopy(em)
				next_state_av_acts = gem.calculate_available_actions("white")
				#We don't know what the reward will be until the game ends. So put 0 for now.
				next_state = next_state.unsqueeze(0)
				tempMemory.push(dqn.Experience(state, action, next_state, next_state_av_acts, 0, False))

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory, True)
			
			#Returns true if length of the memory is greater than or equal to batch_size
			if memory.can_provide_sample(batch_size):
				experiences = memory.sample(batch_size)	#sample experiences from memory
				states, actions, rewards, next_states, next_state_av_actions = um.extract_tensors(experiences)	#extract them

				#get the current q values to calculate loss afterwards
				current_q_values = policy_net(states).gather(dim=-1, index=actions.unsqueeze(-1)).to(device)
				# get output(q-values) for the next state. WARNING! Some terminal states may have been passed to target_net. But final states
				# don't have any q-values since there can be no action to take from terminal states. In the upcoming lines, we spot terminal states
				# and don't take their garbage q-values. Instead, we take only immediate rewards. Is this the best approach?
				next_q_values = target_net(next_states).to(device)
				next_state_maxq = []

				#batch_corrector_start = timeit.default_timer()
				#Getting correct q-values from next_state. To do this, we have to compare against available actions
				for i in range(batch_size):
					q_values = next_q_values[i]
					q_values = q_values.to(device)
					available_actions = next_state_av_actions[i]
					if len(available_actions) == 0:
						next_state_maxq.append(torch.tensor(-10000, dtype=torch.float32))
						continue
					indices = torch.topk(q_values, len(q_values))[1].to(device).detach()

					for j in range(len(indices)):
						max_index = indices[j].to(device).detach()
						#If illegal move is given as output by the model, punish that action and make it select an action again.
						if max_index in available_actions:
							break
					next_state_maxq.append(q_values[max_index])
				#batch_corrector_end = timeit.default_timer()
				#print("Batch Corrector Time: " + str(batch_corrector_end - batch_corrector_start))
				
				# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q). Garbage q-values for terminal states are not used.
				target_q_values = torch.cat(tuple(rewards[i].unsqueeze(0) if experiences[i][5]
								else rewards[i].unsqueeze(0) + gamma * next_state_maxq[i].unsqueeze(0)
								for i in range(len(experiences))))
				target_q_values = target_q_values.to(device)
				
				#clear the old gradients. we only focus on this batch. pytorch accumulates gradients in default.
				optimizer.zero_grad()	
				# returns a new Tensor, detached from the current graph, the result will never require gradient
				target_q_values = target_q_values.detach()
				
				loss = F.mse_loss(current_q_values, target_q_values)
				loss.backward()
				optimizer.step()	#take a step based on the gradients


			#If we're in a terminal state, we never step in the terminal state. We end the episode instead.
			#Record the step number.'''
			if terminal:
				print(str(tempMemory.push_count) + " moves played in this match.\n")
				move_count += tempMemory.push_count
				#Editing the last memory, so that its terminal value is True
				tempMemory.memory[-1] = tempMemory.memory[-1]._replace(terminal = True)	

				#Moving full tuples to big memory, and deleting the temp memory
				memory.pushBlock(tempMemory)
				del tempMemory
				break 

		#Update target network with weights and biases in the policy network
		#Also create new model files
		if episode % target_update == 0:
			target_net.load_state_dict(policy_net.state_dict())
			torch.save({ 'episode': episode,
	            		'model_state_dict': policy_net.state_dict(),
	            		'optimizer_state_dict': optimizer.state_dict(),
	            		'loss': loss,
	            		'current_step': agent.current_step }, PATH_TO_DIRECTORY + "MiniChess-trained-model" + str(episode) + ".tar"
						)
			print("Episode:" + str(episode) + " -------Weights are updated!")

	print("White Wins: " + str(whiteWins) + "\t\t\tWin Rate: %" + str(100*whiteWins/num_episodes))	
	print("Black Wins: " + str(blackWins) + "\t\t\tWin Rate: %" + str(100*blackWins/num_episodes))	
	print("Draw By No Progress: " + str(drawByNoProgress) + "\t\tNo Progress Rate: %" + str(100*drawByNoProgress/num_episodes))
	print("Draw By Too Long Game: " + str(drawByTooLongGame) + "\tNo Progress Rate: %" + str(100*drawByTooLongGame/num_episodes))	
	print("Draw By Stalemate: " + str(drawByStaleMate) + "\t\tStalemate Rate: %" + str(100*drawByStaleMate/num_episodes))
	print("Average Move per game: " + str(move_count / num_episodes))
	return None

def test(policy_net):
	whiteWins = 0
	blackWins = 0
	drawByNoProgress = 0
	drawByTooLongGame = 0
	drawByStaleMate = 0
	move_count = 0
	global em

	for episode in range(1, num_episodes + 1):
		print("Match number: " + str(episode))
		terminal = False
		eps_move_count = 0
		em.reset()	#reset the environment to start all over again
		
		#Calculating available actions for just once, to initiate sequence
		em.calculate_available_actions("white")

		
		while True:
			state = em.get_state()	#get the BitVectorBoard state from the environment as a tensor 
			
			#If the game didn't end with the last move, now it's white's turn to move
			if not terminal:
				em, action = mcts.initializeTree(em, "white", move_time, episode, policy_net, agent, device)	#white makes his move
				eps_move_count += 1
				#em.print()
				#print("\n")
				next_state = em.get_state()

				#We don't know what the reward will be until the game ends. So put 0 for now.
				state = state.unsqueeze(0)
				next_state = next_state.unsqueeze(0)
				state = next_state

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None, True)
				
			#If the game didn't end with the last move, now it's black's turn to move
			if not terminal: 
				em, action = mcts.initializeTree(em, "black", move_time, episode, policy_net, agent, device)	#white makes his move
				eps_move_count += 1
				#em.print()
				#print("\n")
				next_state = em.get_state()
				#We don't know what the reward will be until the game ends. So put 0 for now.
				next_state = next_state.unsqueeze(0)

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	um.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None, True)
		
			if terminal:
				move_count += eps_move_count
				print(str(eps_move_count) + " moves played in this match.\n")
				break 

	print("White Wins: " + str(whiteWins) + "\t\t\tWin Rate: %" + str(100*whiteWins/num_episodes))	
	print("Black Wins: " + str(blackWins) + "\t\t\tWin Rate: %" + str(100*blackWins/num_episodes))	
	print("Draw By No Progress: " + str(drawByNoProgress) + "\t\tNo Progress Rate: %" + str(100*drawByNoProgress/num_episodes))
	print("Draw By Too Long Game: " + str(drawByTooLongGame) + "\tNo Progress Rate: %" + str(100*drawByTooLongGame/num_episodes))	
	print("Draw By Stalemate: " + str(drawByStaleMate) + "\t\tStalemate Rate: %" + str(100*drawByStaleMate/num_episodes))
	print("Average Move per game: " + str(move_count / num_episodes))
	return None


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#Initialize policy net and environment.
	policy_net = dqn.DQN().to(device)
	em = minichess.MiniChess(device)
	last_trained_model = fileoperations.find_last_edited_file(PATH_TO_DIRECTORY)	#Returns the last_trained model from multiple models.
	num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100

	if (sys.argv[1]) == "train":
		strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay) 
		agent = dqn.Agent(strategy, device)
		memory = dqn.ReplayMemory(memory_size)
		optimizer = optim.Adam(params=policy_net.parameters(), lr=lr)
		target_net = dqn.DQN().to(device)
		past_episodes = 0	#how many episode is played before? this variable may be changed in the upcoming if block.
		loss = 0
		
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








