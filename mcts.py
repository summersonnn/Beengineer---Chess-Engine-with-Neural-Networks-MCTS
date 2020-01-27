#!/usr/bin/env python
import random
import math
import hashlib
import argparse
import time
import minichess as mic
import actionsdefined as ad
from copy import copy, deepcopy


#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=2
EXPAND_NUMBER = 3

class State():
	
	def __init__(self, BoardObject, color):
		self.availableActions = BoardObject.available_actions		# e.g Kg3, Ra4
		self.leftActions = deepcopy(BoardObject.available_actions)	#Actions that are not used to spawn a child. Initialized same as avActs but will be changed when a child is spawned
		self.BoardObject = BoardObject								# Minichess object
		self.numberOfMoves = len(self.availableActions)
		self.numberOfLeftMoves = len(self.leftActions)				#Number of moves that left to spawn a child
		self.checkedby = 0							#How many checks that the player is facing?
		self.moveCount = 0							#How many moves played until reaching current state
		self.color = color							# "white" or "black"
		self.exclusive_board_string = ""  		

	def build_exclusive_string(self):
		self.exclusive_board_string =""
		for i in range(6):
			for j in range(3):
				self.exclusive_board_string += self.BoardObject.board[i][j]
			
	def next_state(self, forRollout=False):
		next = deepcopy(self)
		next.color = "white" if self.color == "black" else "black"
		nextActionToChoose = random.randint(0, self.numberOfLeftMoves - 1)
		nextActionNumber = self.leftActions[nextActionToChoose] #e.g 145

		#Decrementing number of left moves and deleting the action that is already used to spawned a child. 
		#Original actions will be left in self.availableActions
		if not forRollout:
			del self.leftActions[nextActionToChoose]
			self.numberOfLeftMoves -= 1

		inv_actions = {v: k for k, v in ad.actions.items()}
		current_action = inv_actions[nextActionNumber]
		del inv_actions
		
		#Get notation before move and current coorbit, empty the squre piece will be moved from, put zero to old coorbit position
		pieceNotationBeforeMove = self.BoardObject.board[int(current_action[0])][int(current_action[1])]	#e.g "+P"
		oldcoorBit = mic.coorToBitVector(int(current_action[0]), int(current_action[1]), pieceNotationBeforeMove) #e.g 30
		next.BoardObject.board[int(current_action[0])][int(current_action[1])] = "XX"
		next.BoardObject.bitVectorBoard[oldcoorBit] = 0

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
		next.moveCount += 1	#In the new node, movecount will be one more
		next.availableActions.clear()	#In the new node, we don't need the parent's available actions as they can be no longer valid actions
		next.numberOfMoves = 0

		checkedby, checkDirectThreats, checkAllThreats = next.BoardObject.IsCheck(next.color)
		next.availableActions = next.BoardObject.calculate_available_actions(next.color, False, checkedby, checkDirectThreats, checkAllThreats)
		next.leftActions = deepcopy(next.availableActions)
		next.numberOfMoves = len(next.availableActions)
		next.numberOfLeftMoves = len(next.leftActions)
		next.checkedby = checkedby
		return next

	def terminal(self):
		if self.moveCount > 50 or self.numberOfMoves == 0:
			return True
		return False

	def reward(self):
		if self.numberOfMoves == 0:
			#Stalemate
			if self.checkedby == 0:
				reward = 0
			else:
				reward = 1 if self.color == "black" else -1
		elif self.moveCount > 50:
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
		s="Node; children: %d; visits: %d; Cum reward: %f Avg reward: %.2f"%(len(self.children),self.visits,self.reward, self.reward/self.visits)
		return s
		

	#Verilen süre içinde simülasyon ve backup yaparak node rewardlarını günceller. Süre sonunda en iyi child döner.
	def UCTSEARCH(self, root, timeout):
		timeout_start = time.time()

		while time.time() < timeout_start + timeout:
			afterTraverse=self.TRAVERSAL(root)
			#First condition was to be able to expand the ROOT node but not other visit=0 nodes, second was to enable zero-move ROLLOUT for terminal leafs
			if (afterTraverse.visits != 0 or afterTraverse == root) and not afterTraverse.state.terminal():
				afterTraverse = self.EXPAND(afterTraverse)
		
			reward = self.ROLLOUT(afterTraverse.state)
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
			tried_children=[c.state for c in node.children]
			new_state=node.state.next_state()
			node.add_child(new_state)
			tried_children += [new_state]
		return node.children[-1]

	#Score for childs are calculated according to color of current node. Because rewards were calculated according to color. 
	#If white, bigger score means better child.
	#If black, lower score means better child.
	def BESTCHILD(self, node,scalar):
		bestscore=-1000 if node.state.color == "white" else 1000
		bestchildren=[]

		for c in node.children:
			if c.visits == 0:
				score = 100 if node.state.color == "white" else -100
			else:
				exploit=c.reward/c.visits
				explore=math.sqrt(math.log(node.visits)/float(c.visits))	
				score=exploit+scalar*explore
			if score==bestscore:
				bestchildren.append(c)
			if score > bestscore and node.state.color == "white":
				bestchildren=[c]
				bestscore=score
			elif score < bestscore and node.state.color =="black":
				bestchildren=[c]
				bestscore=score
		if len(bestchildren)==0:
			print("OOPS: no best child found, probably fatal")
		return random.choice(bestchildren)

	def ROLLOUT(self, state):
		while state.terminal()==False:
			state=state.next_state(True)
		return state.reward()

	def BACKUP(self, node, reward):
		while node!=None:
			node.visits+=1
			node.reward+=reward
			node=node.parent
		return

def initializeTree(boardobject, color, timeout):
	root = Node(State(boardobject, color))
	root.state.build_exclusive_string()	#exclusive string for root is constructed (for hashing)

	result = root.UCTSEARCH(root, timeout)
	
	root = result
	return root.state.BoardObject
