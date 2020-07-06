import os
import signal
import sys
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn.functional as F
import dqn
import minichess
import helperfunctions as hf
import fileoperations
import mcts
from copy import deepcopy
import timeit

PATH_TO_DIRECTORY = "pretrained_model/"
PATH_TO_OLD_GENERATION_MODEL_DIR = "oldgeneration_model/"
batch_size = 256 
gamma = 1 		#1 assumes same action always result in same rewards -> future rewards are NOT discounted
eps_start = 1	#maximum (start) exploration rate
eps_end = 0.1	#minimum exploration rate
eps_decay = 0.002 #higher decay means faster reduction of exploration rate
target_update = 5	#how often does target network get updated? (in terms of episode number) This will also be used in creating model files
memory_size = 7500 #memory size to hold each state,action,next_state, reward, terminal tuple
per_game_memory_size = 30
move_time = 0.1	#Thinking time of a player

def train(White_policy_net, White_target_net, Black_policy_net, Black_target_net):
	whiteWins = 0
	blackWins = 0
	drawByNoProgress = 0
	drawByTooLongGame = 0
	drawByStaleMate = 0
	move_count = 0
	global White_loss
	global Black_loss
	global em

	for episode in range(past_episodes + 1, num_episodes + past_episodes + 1):
		print("Episode number: " + str(episode))
		print("Exploration Rate: " + agent.tell_me_exploration_rate(episode))
		terminal = False
		em.reset()	#reset the environment to start all over again
		White_tempMemory = dqn.ReplayMemory(per_game_memory_size) #Create tempMemory for one match
		Black_tempMemory = dqn.ReplayMemory(per_game_memory_size) #Create tempMemory for one match
		

		#Calculating available actions for just once, to initiate sequence
		em.calculate_available_actions("white")
		
		while True:
			state = em.get_state()	#get the BitVectorBoard state from the environment as a tensor 
			
			#If the game didn't end with the last move, now it's white's turn to move
			if not terminal: 
				em, action = mcts.initializeTree(em, "white", move_time, episode, White_policy_net, agent, device)	#white makes his move
				next_state = em.get_state()
			
				#We don't know what the reward will be until the game ends. So put 0 for now.
				state = state.unsqueeze(0)
				next_state = next_state.unsqueeze(0)
				gem = deepcopy(em)
				next_state_av_acts = gem.calculate_available_actions("black")
				White_tempMemory.push(dqn.Experience(state, action, next_state, next_state_av_acts, 0, False))
				state = next_state

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, White_tempMemory =	hf.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, White_tempMemory)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, White_tempMemory =	hf.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, White_tempMemory, True)
			
			#If the game didn't end with the last move, now it's black's turn to move
			if not terminal: 
				em, action = mcts.initializeTree(em, "black", move_time, episode, Black_policy_net, agent, device)	#white makes his move
				next_state = em.get_state()
				
				gem = deepcopy(em)
				next_state_av_acts = gem.calculate_available_actions("white")
				#We don't know what the reward will be until the game ends. So put 0 for now.
				next_state = next_state.unsqueeze(0)
				Black_tempMemory.push(dqn.Experience(state, action, next_state, next_state_av_acts, 0, False))

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, Black_tempMemory =	hf.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, Black_tempMemory)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, Black_tempMemory =	hf.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, Black_tempMemory, True)
			
			#-----------------------------------------------------FOR WHITE----------------------------------------
			#Returns true if length of the memory is greater than or equal to batch_size
			if White_memory.can_provide_sample(batch_size):
				White_experiences = White_memory.sample(batch_size)	#sample experiences from memory
				Wstates, Wactions, Wrewards, Wnext_states, Wnext_state_av_actions = hf.extract_tensors(White_experiences)	#extract them
				Wstates = Wstates.to(device)
				Wactions = Wactions.to(device)
				Wrewards = Wrewards.to(device)
				Wnext_states = Wnext_states.to(device)
				
				#get the current q values to calculate loss afterwards
				#Shape of policy_net(states): [batchsize, 282]
				#Shape of actions: [batchsize] (Tek rowluk bir tensor)
				#Shape of actions.unsq(-1): [batchsize, 1] (Bir üstteki rowdakileri her row'a birer tane olacak şekilde rowlara ayır)
				#Shape of current_q_values: [batchsize, 1] (Her rowdan en büyük q-value'yu seçtik.)
				Wcurrent_q_values = White_policy_net(Wstates).gather(dim=1, index=Wactions.unsqueeze(-1)).to(device)
				#Shape of next_q_values:	[batchsize,282]
				Wnext_q_values = White_target_net(Wnext_states).detach().to(device)
				Wnext_state_maxq = []
				
				#batch_corrector_start = timeit.default_timer()
				#Getting correct q-values from next_state. To do this, we have to compare against available actions
				for i in range(batch_size):
					q_values = Wnext_q_values[i]
					q_values = q_values.to(device)
					available_actions = Wnext_state_av_actions[i]
					if len(available_actions) == 0:
						Wnext_state_maxq.append(torch.tensor(-1000000, dtype=torch.float32))
						continue
					indices = torch.topk(q_values, len(q_values))[1].to(device).detach()

					for j in range(len(indices)):
						max_index = indices[j].to(device).detach()
						#If illegal move is given as output by the model, punish that action and make it select an action again.
						if max_index in available_actions:
							break
					Wnext_state_maxq.append(q_values[max_index])
				#batch_corrector_end = timeit.default_timer()
				#print("Batch Corrector Time: " + str(batch_corrector_end - batch_corrector_start))

				# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q). Garbage q-values for terminal states are not used.
				target_q_values = torch.cat(tuple(Wrewards[i].unsqueeze(0) if White_experiences[i][5]
								else Wrewards[i].unsqueeze(0) + gamma * Wnext_state_maxq[i].unsqueeze(0)
								for i in range(batch_size)))
				target_q_values = target_q_values.to(device)
				#Shape of target_q_values: [batchsize, 1]
				
				#clear the old gradients. we only focus on this batch. pytorch accumulates gradients in default.
				White_optimizer.zero_grad()	
				# returns a new Tensor, detached from the current graph, the result will never require gradient
				target_q_values = target_q_values.detach()
				
				White_loss = F.mse_loss(Wcurrent_q_values, target_q_values)
				White_loss.backward()
				White_optimizer.step()	#take a step based on the gradients

				
	#-----------------------------------------------------FOR BLACK----------------------------------------
			#Returns true if length of the memory is greater than or equal to batch_size
			if Black_memory.can_provide_sample(batch_size):
				Black_experiences = Black_memory.sample(batch_size)	#sample experiences from memory
				Bstates, Bactions, Brewards, Bnext_states, Bnext_state_av_actions = hf.extract_tensors(Black_experiences)	#extract them
				Bstates = Bstates.to(device)
				Bactions = Bactions.to(device)
				Brewards = Brewards.to(device)
				Bnext_states = Bnext_states.to(device)

				#get the current q values to calculate loss afterwards
				#Shape of policy_net(states): [batchsize, 282]
				#Shape of actions: [batchsize] (Tek rowluk bir tensor)
				#Shape of actions.unsq(-1): [batchsize, 1] (Bir üstteki rowdakileri her row'a birer tane olacak şekilde rowlara ayır)
				#Shape of current_q_values: [batchsize, 1] (Her rowdan en büyük q-value'yu seçtik.)
				Bcurrent_q_values = Black_policy_net(Bstates).gather(dim=1, index=Bactions.unsqueeze(-1)).to(device)
				#Shape of next_q_values:	[batchsize,282]
				Bnext_q_values = Black_target_net(Bnext_states).detach().to(device)
				Bnext_state_maxq = []

				#batch_corrector_start = timeit.default_timer()
				#Getting correct q-values from next_state. To do this, we have to compare against available actions
				for i in range(batch_size):
					q_values = Bnext_q_values[i]
					q_values = q_values.to(device)
					available_actions = Bnext_state_av_actions[i]
					if len(available_actions) == 0:
						Bnext_state_maxq.append(torch.tensor(-1000000, dtype=torch.float32))
						continue
					indices = torch.topk(q_values, len(q_values))[1].to(device).detach()

					for j in range(len(indices)):
						max_index = indices[j].to(device).detach()
						#If illegal move is given as output by the model, punish that action and make it select an action again.
						if max_index in available_actions:
							break
					Bnext_state_maxq.append(q_values[max_index])
				#batch_corrector_end = timeit.default_timer()
				#print("Batch Corrector Time: " + str(batch_corrector_end - batch_corrector_start))

				# set y_j to r_j for terminal state, otherwise to r_j + gamma*max(Q). Garbage q-values for terminal states are not used.
				target_q_values = torch.cat(tuple(Brewards[i].unsqueeze(0) if Black_experiences[i][5]
								else Brewards[i].unsqueeze(0) + gamma * Bnext_state_maxq[i].unsqueeze(0)
								for i in range(batch_size)))
				target_q_values = target_q_values.to(device)
				#Shape of target_q_values: [batchsize, 1]
				
				#clear the old gradients. we only focus on this batch. pytorch accumulates gradients in default.
				Black_optimizer.zero_grad()	
				# returns a new Tensor, detached from the current graph, the result will never require gradient
				target_q_values = target_q_values.detach()
				
				Black_loss = F.mse_loss(Bcurrent_q_values, target_q_values)
				Black_loss.backward()
				Black_optimizer.step()	#take a step based on the gradients
				#---------------------------------------------------------------------

			#If we're in a terminal state, we never step in the terminal state. We end the episode instead.
			#Record the step number.'''
			if terminal:
				print(str(White_tempMemory.push_count + Black_tempMemory.push_count) + " moves played in this match.\n")
				move_count += White_tempMemory.push_count
				move_count += White_tempMemory.push_count
				#Editing the last memory, so that its terminal value is True
				White_tempMemory.memory[-1] = White_tempMemory.memory[-1]._replace(terminal = True)
				Black_tempMemory.memory[-1] = Black_tempMemory.memory[-1]._replace(terminal = True)	

				#Moving full tuples to big memory, and deleting the temp memory
				White_memory.pushBlock(White_tempMemory)
				Black_memory.pushBlock(Black_tempMemory)
				del White_tempMemory
				del Black_tempMemory
				break 

		#Update target network with weights and biases in the policy network
		#Also create new model files
		if episode % target_update == 0:
			White_target_net.load_state_dict(White_policy_net.state_dict())
			Black_target_net.load_state_dict(Black_policy_net.state_dict())
			torch.save({ 'episode': episode,
	            		'White_model_state_dict': White_policy_net.state_dict(),
	            		'White_optimizer_state_dict': White_optimizer.state_dict(),
	            		'White_loss': White_loss,
	            		'Black_model_state_dict': Black_policy_net.state_dict(),
	            		'Black_optimizer_state_dict': Black_optimizer.state_dict(),
	            		'Black_loss': Black_loss,
						}, PATH_TO_DIRECTORY + "MiniChess-trained-model" + str(episode) + ".tar"
						)
			print("Episode:" + str(episode) + " -------Weights are updated!")

			
			

	print("White Wins: " + str(whiteWins) + "\t\t\tWin Rate: %" + str(100*whiteWins/num_episodes))	
	print("Black Wins: " + str(blackWins) + "\t\t\tWin Rate: %" + str(100*blackWins/num_episodes))	
	print("Draw By No Progress: " + str(drawByNoProgress) + "\t\tNo Progress Rate: %" + str(100*drawByNoProgress/num_episodes))
	print("Draw By Too Long Game: " + str(drawByTooLongGame) + "\tToo Long Game Rate: %" + str(100*drawByTooLongGame/num_episodes))	
	print("Draw By Stalemate: " + str(drawByStaleMate) + "\t\tStalemate Rate: %" + str(100*drawByStaleMate/num_episodes))
	print("Average Move per game: " + str(move_count / num_episodes))
	return None

def test(policy_net_white, policy_net_black, humanVsComputer=None):
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
				#Not play against computer as White
				if humanVsComputer != "W" and humanVsComputer != "w":
					em, action = mcts.initializeTree(em, "white", move_time*10, episode, policy_net_white, agent, device)	#white makes his move
				#Play against computer as White
				else:
					em.print()
					em = hf.get_user_move(em, "white")

				eps_move_count += 1
				#em.print()
				#print("\n")
				next_state = em.get_state()

				#We don't know what the reward will be until the game ends. So put 0 for now.
				state = state.unsqueeze(0)
				next_state = next_state.unsqueeze(0)
				state = next_state

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	hf.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	hf.check_game_termination(em , "black", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None, True)
				
			#If the game didn't end with the last move, now it's black's turn to move
			if not terminal: 
				#Not play against computer as Black
				if humanVsComputer != "B" and humanVsComputer != "b":
					em, action = mcts.initializeTree(em, "black", move_time*10, episode, policy_net_black, agent, device)	#white makes his move
				#Play against computer as Black
				else:
					em.print()
					em = hf.get_user_move(em, "black")

				eps_move_count += 1
				#em.print()
				#print("\n")
				next_state = em.get_state()
				#We don't know what the reward will be until the game ends. So put 0 for now.
				next_state = next_state.unsqueeze(0)

			#Check if game ends
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	hf.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None)
			#Check if game ends by no progress rule
			terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory =	hf.check_game_termination(em , "white", terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, None, True)
		
			if terminal:
				move_count += eps_move_count
				print(str(eps_move_count) + " moves played in this match.\n")
				break 

	print("White Wins: " + str(whiteWins) + "\t\t\tWin Rate: %" + str(100*whiteWins/num_episodes))	
	print("Black Wins: " + str(blackWins) + "\t\t\tWin Rate: %" + str(100*blackWins/num_episodes))	
	print("Draw By No Progress: " + str(drawByNoProgress) + "\t\tNo Progress Rate: %" + str(100*drawByNoProgress/num_episodes))
	print("Draw By Too Long Game: " + str(drawByTooLongGame) + "\tToo Long Game Rate: %" + str(100*drawByTooLongGame/num_episodes))	
	print("Draw By Stalemate: " + str(drawByStaleMate) + "\t\tStalemate Rate: %" + str(100*drawByStaleMate/num_episodes))
	print("Average Move per game: " + str(move_count / num_episodes))
	return None


if __name__ == '__main__':
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	#Initialize policy nets and environment.
	White_policy_net = dqn.DQN().to(device)
	Black_policy_net = dqn.DQN().to(device)
	em = minichess.MiniChess(device)
	last_trained_model = fileoperations.find_last_edited_file(PATH_TO_DIRECTORY)	#Returns the last_trained model from multiple models.
	num_episodes = int(sys.argv[2]) if len(sys.argv) > 2 else 100

	if (sys.argv[1]) == "train":
		strategy = dqn.EpsilonGreedyStrategy(eps_start, eps_end, eps_decay)
		agent = dqn.Agent(strategy, device)
		White_memory = dqn.ReplayMemory(memory_size)
		Black_memory = dqn.ReplayMemory(memory_size)
		White_target_net = dqn.DQN().to(device)
		Black_target_net = dqn.DQN().to(device)
		past_episodes = 0	#how many episode is played before? this variable may be changed in the upcoming if block.
		White_loss = 0
		Black_loss = 0
		lr = 0.2
		White_optimizer = optim.Adam(params=White_policy_net.parameters(), lr=lr)
		Black_optimizer = optim.Adam(params=Black_policy_net.parameters(), lr=lr)

		if last_trained_model is not None:
			print("***Last trained model: " + last_trained_model)
			checkpoint = torch.load(last_trained_model, map_location=device)
			past_episodes = checkpoint['episode']
			lr = hf.learning_rate_calculator(past_episodes) #how much to change the model in response to the estimated error each time the model weights are updated
			
			White_policy_net.load_state_dict(checkpoint['White_model_state_dict'])
			White_loss = checkpoint['White_loss']
			White_optimizer = optim.Adam(params=White_policy_net.parameters(), lr=lr)
			White_optimizer.load_state_dict(checkpoint['White_optimizer_state_dict'])
			
			Black_policy_net.load_state_dict(checkpoint['Black_model_state_dict'])
			Black_loss = checkpoint['Black_loss']
			Black_optimizer = optim.Adam(params=Black_policy_net.parameters(), lr=lr)
			Black_optimizer.load_state_dict(checkpoint['Black_optimizer_state_dict'])
			del checkpoint

		#Weights and biases in the target net is same as in policy net. Target net will work in eval mode and will not update the weights (on its own)
		White_target_net.load_state_dict(White_policy_net.state_dict())
		Black_target_net.load_state_dict(Black_policy_net.state_dict())
		White_target_net.eval()
		Black_target_net.eval()
		train(White_policy_net, White_target_net, Black_policy_net, Black_target_net)

	elif (sys.argv[1]) == "test":
		humanVsComputer = None
		choice = input("Play against another generation? (A)?\n Play against computer as White (W) as black (B)")
		choice2 = ""

		if choice == "A" or choice == "a":
			oldgeneration_model = fileoperations.find_last_edited_file(PATH_TO_OLD_GENERATION_MODEL_DIR)
			choice2 = input("Last trained model White(W) or Black(B) ?")

			if oldgeneration_model is not None:
				print("***Old Generation model: " + oldgeneration_model)
				checkpointOld = torch.load(oldgeneration_model, map_location=device)
				if choice2 == 'W' or choice2 == 'w':
					Black_policy_net.load_state_dict(checkpointOld['Black_model_state_dict'])
				else:
					White_policy_net.load_state_dict(checkpointOld['White_model_state_dict'])
				del checkpointOld
			else:
				print("Test cannot be done due to absence of OLDMODEL weights file")
				os.kill(os.getpid(), signal.SIGTERM)

		elif choice == "W" or choice == "w" or choice == "B" or choice == "b":
			humanVsComputer = choice

		agent = dqn.Agent(None, device)	#strategy is none since epsilon greedy strategy is not required in test mode. We don't explore.

		if last_trained_model is not None:
			print("***Last trained model: " + last_trained_model)
			checkpoint = torch.load(last_trained_model, map_location=device)
			#If humanVsComputer, load white and black model from last trained model.
			#Else, load white or black from last trained model according to last model color choice (choice2)
			if choice2 == 'W' or choice2 == 'w' or humanVsComputer != None:
				White_policy_net.load_state_dict(checkpoint['White_model_state_dict'])
			elif (choice2 != 'W' and choice2 != "w") or humanVsComputer != None:
				Black_policy_net.load_state_dict(checkpoint['Black_model_state_dict'])
			del checkpoint
		else:
			print("Test cannot be done due to absence of LASTMODEL weights file")
			os.kill(os.getpid(), signal.SIGTERM)

		#Policy net should be in eval mode to avoid gradient decent in test mode.
		White_policy_net.eval()
		Black_policy_net.eval()
		test(White_policy_net, Black_policy_net, humanVsComputer)

	#sys.exit() or raise SystemExit doesn't work for some reason. That's the only way I could end the process.
	os.kill(os.getpid(), signal.SIGTERM)








