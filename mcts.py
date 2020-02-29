import random
import torch
import math
import hashlib
import argparse
import timeit
import minichess as mic
import normalizer
import actionsdefined as ad
from copy import copy, deepcopy


#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=2
EXPAND_NUMBER = 3

class State():
	inv_actions = {v: k for k, v in ad.actions.items()}
	node_count = 0
	nz = normalizer.MinMaxNormalizer(0,30,0,60)

	def __init__(self, BoardObject, color):
		self.leftActions = deepcopy(BoardObject.available_actions)	#Actions that are not used to spawn a child. Initialized same as avActs but will be changed when a child is spawned
		self.createdByThisAction = 0								#Tells which action resulted in this state
		self.BoardObject = BoardObject								# Minichess object
		self.numberOfMoves = len(self.BoardObject.available_actions)
		self.checkedby = BoardObject.checkedby						#How many checks that the player is facing?
		self.color = color											# "white" or "black"
		self.exclusive_board_string = ""  		

	def build_exclusive_string(self):
		self.exclusive_board_string =""
		for i in range(109):
			self.exclusive_board_string += str(self.BoardObject.bitVectorBoard[i])
			
	#If runs for EXPANDING, action does not come as parameter since it will expanded fully, in every possible way.
	#If runs for ROLLOUT, action comes as parameter from rollout function
	def next_state(self, action=0, forRollout=False, user_action=None):
		State.node_count += 1
		next = deepcopy(self)
		next.color = "white" if self.color == "black" else "black"

		#Selection action when expanding the tree.
		#Deleting the action that is already used to spawned a child. 
		#Original actions will be left in self.availableActions.
		if user_action == None:
			if not forRollout:
				action = self.leftActions[0] #e.g 145
				next.createdByThisAction = action
				del self.leftActions[0]
			#Converting action Scalar to String Format
			current_action = State.inv_actions[action]
		#If the move was played by human
		else:
			current_action = user_action

		#Get notation before move and current coorbit, empty the squre piece will be moved from, put zero to old coorbit position
		pieceNotationBeforeMove = self.BoardObject.board[int(current_action[0])][int(current_action[1])]	#e.g "+P"
		oldcoorBit = mic.coorToBitVector(int(current_action[0]), int(current_action[1]), pieceNotationBeforeMove) #e.g 30
		next.BoardObject.board[int(current_action[0])][int(current_action[1])] = "XX"
		next.BoardObject.bitVectorBoard[oldcoorBit] = 0

		#Increase No Progress count and move count, if there is progress it will be set to zero in the upcoming lines
		next.BoardObject.NoProgressCount += 1
		next.BoardObject.MoveCount += 1
		next.BoardObject.bitVectorBoard[108] = State.nz.normalizeNoProgress(next.BoardObject.NoProgressCount)
		next.BoardObject.bitVectorBoard[109] = State.nz.normalizeMoveCount(next.BoardObject.MoveCount)

		#If pawn moves, Set No Progress Count back to 0
		if pieceNotationBeforeMove[1] == "P":
			next.BoardObject.NoProgressCount = 0
			next.BoardObject.bitVectorBoard[108] = 0	

		#Arrange the notation of the piece after move in case of promotion. Also check if there is a promotion.
		pieceNotationAfterMove = pieceNotationBeforeMove[0] + current_action[-1]
		color = "white" if pieceNotationAfterMove[0] == "+" else "black"
		promoted = True if len(current_action) == 6 else False
		friendList = next.BoardObject.WhitePieceList if pieceNotationAfterMove[0] == '+' else next.BoardObject.BlackPieceList
		enemyList = next.BoardObject.WhitePieceList if pieceNotationAfterMove[0] == '-' else next.BoardObject.BlackPieceList
		
		#If capture happened, obtain the BitBoard repr. of captured piece, then remove the piece object from piece object list
		capturedPieceNotation = self.BoardObject.board[int(current_action[2])][int(current_action[3])]

		if capturedPieceNotation != "XX":
			capturedPieceBit = mic.coorToBitVector(int(current_action[2]), int(current_action[3]), capturedPieceNotation)
			next.BoardObject.bitVectorBoard[capturedPieceBit] = 0
			next.BoardObject.removeCapturedPiece(capturedPieceBit, enemyList)
			next.BoardObject.NoProgressCount = 0
			next.BoardObject.bitVectorBoard[108] = 0
			

		#Update the board, obtain the new Bitboard repr. of the piece and update the bitvectorboard accordingly
		next.BoardObject.board[int(current_action[2])][int(current_action[3])] = pieceNotationAfterMove
		newcoorBit = mic.coorToBitVector(int(current_action[2]), int(current_action[3]), pieceNotationAfterMove)
		next.BoardObject.bitVectorBoard[newcoorBit] = 1

		#Call the step function of the object, to make it renew itself (if the object is still valid, which means promotion did not happen)
		if not promoted:
			for i in friendList:
				if i.BitonBoard == oldcoorBit:
					i.step(newcoorBit, int(current_action[2]), int(current_action[3]) ) 
		#if promoted, create a new object and kill the pawn object
		else:
			#Not captured, but since promoted, pawn object must be deleted
			friendList.append(mic.Rook(color, int(current_action[2]), int(current_action[3])))	#Warning! Possible costly operation. Test it.
			next.BoardObject.removeCapturedPiece(oldcoorBit, friendList)

		next.build_exclusive_string()	#New exclusive string is constructed, ready for being hashed
		
		next.checkedby, checkDirectThreats, checkAllThreats = next.BoardObject.IsCheck(next.color)
		next.BoardObject.calculate_available_actions(next.color, False, next.checkedby, checkDirectThreats, checkAllThreats)
		next.leftActions = deepcopy(next.BoardObject.available_actions)
		next.numberOfMoves = len(next.leftActions)

		return next

	def terminal(self):
		if self.BoardObject.NoProgressCount >= 30 or self.BoardObject.MoveCount >= 60 or self.numberOfMoves == 0:
			return True
		return False

	def reward(self):
		if self.numberOfMoves == 0:
			#Stalemate
			if self.checkedby == 0:
				reward = 0
			else:
				reward = 1 if self.color == "black" else -1
		else:
			reward = 0

		return reward

	def __hash__(self):
		return int(hashlib.md5(self.exclusive_board_string.encode('utf-8')).hexdigest(),16)
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False
	def __repr__(self):
		return self.board.print()	

class Node():

	def __init__(self, state, parent=None):
		self.visits=0
		self.reward=0.0	
		self.state=state
		self.children=[]
		self.parent=parent	
	def add_child(self, child_state):
		child = Node(child_state, self)
		self.children.append(child)
	def __repr__(self):
		s="Node; children: %d; visits: %d; Cum reward: %f Avg reward: %.2f"%(len(self.children),self.visits,self.reward, (self.reward/self.visits) if self.visits !=0 else 999999)
		return s
		

	#Verilen süre içinde simülasyon ve backup yaparak node rewardlarını günceller. Süre sonunda en iyi child döner.
	def UCTSEARCH(self, root, episode, policy_net, agent, timeout):
		timeout_start = timeit.default_timer()
		while True:
			diff = timeit.default_timer() - timeout_start
			if diff >= timeout:
				break

			afterTraverse=self.TRAVERSAL(root)
			#First condition was to be able to expand the ROOT node but not other visit=0 nodes, second was to enable zero-move ROLLOUT for terminal leafs
			if (afterTraverse.visits != 0 or afterTraverse == root) and not afterTraverse.state.terminal():
				afterTraverse = self.EXPAND(afterTraverse)

			#ROLLOUT
			traversedState = afterTraverse.state
			while traversedState.terminal()==False:
				stateTensor = traversedState.BoardObject.get_state() 
				#randomIndex = random.randrange(0, traversedState.numberOfMoves)
				#action = traversedState.BoardObject.available_actions[randomIndex]
				#If strategy is not None, it's Training, if it is None, it's Testing
				if agent.strategy != None:
					action = agent.select_action(stateTensor, traversedState.color, traversedState.BoardObject.available_actions, episode, policy_net, False)
				else:
					action = agent.select_action(stateTensor, traversedState.color, traversedState.BoardObject.available_actions, episode, policy_net, True)
					
				action = action.item()
				traversedState = traversedState.next_state(action, True)
				
			reward = traversedState.reward()
			self.BACKUP(afterTraverse,reward)
		return self.BESTCHILD(root,0)

	#Sürekli best child'ı seçerek leaf node'a ulaştırır. Buradan ilerde ya expand edilecek ya rollout yapılacak.
	def TRAVERSAL(self, node):
		while len(node.children) != 0:
			node=self.BESTCHILD(node,SCALAR)
		return node
	 
	#Leaf node expand ettirir.
	def EXPAND(self, node):
		for i in range(node.state.numberOfMoves):
			new_state=node.state.next_state()
			node.add_child(new_state)
		return random.choice(node.children)


	#Score for childs are calculated according to color of current node. Because rewards were calculated according to color. 
	#If white, bigger score means better child.
	#If black, lower score means better child.
	def BESTCHILD(self, node,scalar):
		bestscore=-100000 if node.state.color == "white" else 100000
		bestchildren = None

		for c in random.sample(node.children, len(node.children)):
			if c.visits == 0:
				return c
			else:
				exploit=c.reward/c.visits
				explore=math.sqrt(math.log(node.visits)/float(c.visits))	
				score=exploit+scalar*explore

			if score >= bestscore and node.state.color == "white":
				bestchildren = c
				bestscore = score
			elif score <= bestscore and node.state.color =="black":
				bestchildren = c
				bestscore = score
		if bestchildren == None:
			print("OOPS: no best child found, probably fatal")
		return bestchildren

	'''def ROLLOUT(self, state, episode, policy_net, agent):
		while state.terminal()==False:
			stateTensor = state.BoardObject.get_state()

			#randomIndex = random.randrange(0, state.numberOfMoves)
			#action = state.BoardObject.available_actions[randomIndex]
			#If strategy is not None, it's Training, if it is None, it's Testing
			if agent.strategy != None:
				action = agent.select_action(stateTensor, state.BoardObject.available_actions, episode, policy_net, False)
			else:
				start = datetime.datetime.now()
				action = agent.select_action(stateTensor, state.BoardObject.available_actions, episode, policy_net, True)
				diff = datetime.datetime.now() - start
				#Node.rollout_time += diff.total_seconds()
				#Node.rollout_counter += 1
			
			action = action.item()
			state = state.next_state(action, True)
		return state.reward()'''

	def BACKUP(self, node, reward):
		while node!=None:
			node.visits+=1
			node.reward+=reward
			node=node.parent
		return

def initializeTree(boardobject, color, timeout, episode, policy_net, agent, device):
	root = Node(State(boardobject, color))
	root.state.build_exclusive_string()	#exclusive string for root is constructed (for hashing)

	result = root.UCTSEARCH(root, episode, policy_net, agent, timeout)
	root = result
	
	#print("Avg time lost in NN: " + str(Node.rollout_time/Node.rollout_counter))



	return root.state.BoardObject, torch.tensor([root.state.createdByThisAction]).to(device)
