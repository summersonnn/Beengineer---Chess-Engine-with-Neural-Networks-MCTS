import numpy as np 
import torch
import dqn
import mcts

def extract_tensors(experiences):
	# Convert batch of Experiences to Experience of batches
	batch = dqn.Experience(*zip(*experiences))

	state_batch = torch.cat(batch.state, 0)
	action_batch = torch.cat(batch.action)
	nextState_batch = torch.cat(batch.next_state, 0)
	reward_batch = torch.cat(batch.reward)
	next_state_av_actions = batch.next_state_av_acts
	
	return (state_batch, action_batch, reward_batch, nextState_batch, next_state_av_actions)

#Place the incoming reward into the tempMemory
def place_rewards(tempMemory, reward):
	for i in range(len(tempMemory.memory)):
		tempMemory.memory[i] = tempMemory.memory[i]._replace(reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0))
	return tempMemory

def check_game_termination(em, color, terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory, IsForNoProgess=False):

	if not IsForNoProgess and not terminal:
		if len(em.available_actions) == 0:
			terminal = True
			if em.checkedby == 0:
				print("Stalemate!")
				drawByStaleMate += 1
				if tempMemory != None:
					place_rewards(tempMemory, 0)  	#Place 0 into the reward section of namedtuples in the tempMemory if it's training
			elif color == "white":
				print("Black wins!")
				blackWins += 1
				if tempMemory != None:
					place_rewards(tempMemory, -1)  	#Place -1 into the reward section of namedtuples in the tempMemory if it's training
			else:
				print("White wins!")
				whiteWins += 1
				if tempMemory != None:
					place_rewards(tempMemory, 1)  	#Place 1 into the reward section of namedtuples in the tempMemory if it's training

	elif not terminal:
		if em.NoProgressCount >= 30:
			terminal = True
			print("Draw by no progress!")
			drawByNoProgress += 1
			
		elif em.MoveCount >= 60:
			terminal = True
			print("Draw by toooooo long game!")
			drawByTooLongGame += 1

		if tempMemory != None:
			place_rewards(tempMemory, 0)  	#Place 0 into the reward section of namedtuples in the tempMemory if it's training


	return terminal, whiteWins, blackWins, drawByNoProgress, drawByTooLongGame, drawByStaleMate, tempMemory

def learning_rate_calculator(episode):
	if episode < 100:
		return 0.1
	if episode < 1000:
		return 0.01
	else:
		return 0.001

def get_user_move(em, color):
	userInput = input("What's your move? (Type it like a2a3)")
	lastInput = "xxxxx"

	if userInput[0] == "a":
		lastInput = lastInput[0] + "0" + lastInput[2:]
	elif userInput[0] == "b":
		lastInput = lastInput[0] + "1" + lastInput[2:]
	elif userInput[0] == "c":
		lastInput = lastInput[0] + "2" + lastInput[2:]

	lastInput = str(6 - int(userInput[1])) + lastInput[1:]
	lastInput = lastInput[:2] + str(6 - int(userInput[3])) + lastInput[4:]

	if userInput[2] == "a":
		lastInput = lastInput[:3] + "0" + lastInput[4:]
	elif userInput[2] == "b":
		lastInput = lastInput[:3] + "1" + lastInput[4:]
	elif userInput[2] == "c":
		lastInput = lastInput[:3] + "2" + lastInput[4:]

	pieceNotationBeforeMove = em.board[int(lastInput[0])][int(lastInput[1])]
	#No promotion
	if pieceNotationBeforeMove[1] != "P" or (pieceNotationBeforeMove == "+P" and lastInput[0] != "1") or (pieceNotationBeforeMove == "-P" and lastInput[0] != "4"):
		lastInput = lastInput[:4] + pieceNotationBeforeMove[1]
	#Promotion
	else:
		lastInput = lastInput[:4] + "=R"

	incoming = mcts.State(em, color)
	outgoing = incoming.next_state(0, False, lastInput)
	return outgoing.BoardObject











