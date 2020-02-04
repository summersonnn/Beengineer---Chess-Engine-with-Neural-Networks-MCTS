import numpy as np 
import torch
import dqn

def extract_tensors(experiences):
	# Convert batch of Experiences to Experience of batches
	batch = dqn.Experience(*zip(*experiences))

	state_batch = torch.cat(batch.state, 0)
	action_batch = torch.cat(batch.action)
	nextState_batch = torch.cat(batch.next_state, 0)
	reward_batch = torch.cat(batch.reward)

	return (state_batch, action_batch, reward_batch, nextState_batch)

#Place the incoming reward into the tempMemory
def place_rewards(tempMemory, reward):
	for i in range(len(tempMemory.memory)):
		tempMemory.memory[i] = tempMemory.memory[i]._replace(reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0))
	return tempMemory

def check_game_termination(em, color, terminal, whiteWins, blackWins, drawByNoProgress, drawByStaleMate, tempMemory, IsForNoProgess=False):

	if not IsForNoProgess:
		if not terminal and len(em.available_actions) == 0:
			terminal = True
			if em.checkedby == 0:
				print("Stalemate!\n")
				drawByStaleMate += 1
				if tempMemory != None:
					place_rewards(tempMemory, 0)  	#Place 0 into the reward section of namedtuples in the tempMemory if it's training
			elif color == "white":
				print("Black wins!\n")
				blackWins += 1
				if tempMemory != None:
					place_rewards(tempMemory, -1)  	#Place -1 into the reward section of namedtuples in the tempMemory if it's training
			else:
				print("White wins!\n")
				whiteWins += 1
				if tempMemory != None:
					place_rewards(tempMemory, 1)  	#Place 1 into the reward section of namedtuples in the tempMemory if it's training

	else:
		if not terminal and em.bitVectorBoard[108] > 20:
			terminal = True
			print("Draw by no progress!\n")
			drawByNoProgress += 1
			if tempMemory != None:
					place_rewards(tempMemory, 0)  	#Place 0 into the reward section of namedtuples in the tempMemory if it's training

	return terminal, whiteWins, blackWins, drawByNoProgress, drawByStaleMate, tempMemory









