#!/usr/bin/env python
import random
import math
import hashlib
import argparse
import time


"""
A quick Monte Carlo Tree Search implementation.  For more details on MCTS see See http://pubs.doc.ic.ac.uk/survey-mcts-methods/survey-mcts-methods.pdf
The State is just a game where you have NUM_TURNS and at turn i you can make
a choice from [-2,2,3,-3]*i and this to to an accumulated value.  The goal is for the accumulated value to be as close to 0 as possible.
The game is not very interesting but it allows one to study MCTS which is.  Some features 
of the example by design are that moves do not commute and early mistakes are more costly.  
In particular there are two models of best child that one can use 
"""

#MCTS scalar.  Larger scalar will increase exploitation, smaller will increase exploration. 
SCALAR=2
EXPAND_NUMBER = 5


class State():
	ALPHABET = ['a', 'b', 'c', 'ç', 'd', 'e', 'f', 'g', 'ğ', 'h', 'ı', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'ö', 'p', 'r', 's', 'ş', 't', 'u', 'ü', 'v', 'y', 'z']
	num_moves=29
	def __init__(self, word="", moves=[]):
		self.moves=moves
		self.word = word
	def next_state(self):
		nextLetter = self.ALPHABET[random.randint(0, 28)]
		next=State(self.word+nextLetter, self.moves)
		return next
	def terminal(self):
		#Minichess için buradaki kontrol "action space 0'a düşmüşse (matsa)" olacak
		if random.random() > 0.8:
			return True
		return False
	def reward(self):
		lastIndex = 0
		reward = 0
		for i in range(len(self.word)):
			index = self.find_index(self.word[i])
			if index <= lastIndex :
				reward -= 1
			else:
				reward += 1
			lastIndex = index

		return reward

	def find_index(self, letter):
		for i in range(len(self.ALPHABET)):
			if(letter == self.ALPHABET[i]):
				return i

	def __hash__(self):
		return int(hashlib.md5(str(self.word).encode('utf-8')).hexdigest(),16)
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False
	def __repr__(self):
		s="Word: %s"%(self.word)
		return s
	

class Node():
	def __init__(self, state, parent=None):
		self.visits=0
		self.reward=0.0	
		self.state=state
		self.children=[]
		self.parent=parent	
	def add_child(self,child_state):
		child=Node(child_state,self)
		self.children.append(child)
	def __repr__(self):
		s="Node; children: %d; visits: %d; Cum reward: %f Avg reward: %.2f"%(len(self.children),self.visits,self.reward, self.reward/self.visits)
		return s
		

#Verilen süre içinde simülasyon ve backup yaparak node rewardlarını günceller. Süre sonunda en iyi child döner.
def UCTSEARCH(root, timeout):
	timeout_start = time.time()

	while time.time() < timeout_start + timeout:
		afterTraverse=TRAVERSAL(root)
		if (afterTraverse.visits != 0):
			afterTraverse = EXPAND(afterTraverse, EXPAND_NUMBER)
	
		reward = ROLLOUT(afterTraverse.state)
		BACKUP(afterTraverse,reward)
	return BESTCHILD(root,0)

#Sürekli best child'ı seçerek leaf node'a ulaştırır. Buradan ilerde ya expand edilecek ya rollout yapılacak.
def TRAVERSAL(node):
	while len(node.children) != 0:
		node=BESTCHILD(node,SCALAR)
	return node
 
#Leaf node expand ettirir. Oluşturulacak child sayısı parametre olarak verilir.
def EXPAND(node, expandNumber):
	for i in range(expandNumber):
		tried_children=[c.state for c in node.children]
		new_state=node.state.next_state()
		while new_state in tried_children:
			new_state=node.state.next_state()
		node.add_child(new_state)
		tried_children += [new_state]
	return node.children[-1]

def BESTCHILD(node,scalar):
	bestscore=-10000
	bestchildren=[]

	for c in node.children:
		if c.visits == 0:
			score = 9999999
		else:
			exploit=c.reward/c.visits
			explore=math.sqrt(math.log(node.visits)/float(c.visits))	
			score=exploit+scalar*explore
		if score==bestscore:
			bestchildren.append(c)
		if score>bestscore:
			bestchildren=[c]
			bestscore=score
	if len(bestchildren)==0:
		print("OOPS: no best child found, probably fatal")
	return random.choice(bestchildren)

def ROLLOUT(state):
	while state.terminal()==False:
		state=state.next_state()
	return state.reward()

def BACKUP(node,reward):
	while node!=None:
		node.visits+=1
		node.reward+=reward
		node=node.parent
	return

if __name__=="__main__":
	
	root=Node(State())
	root.visits = 1

	timeout = 2   # [seconds]
	
	for i in range(5):
		result = UCTSEARCH(root, timeout)
		print("At %d level, state: %s" %(i+1, result.state.word))
		print("At this level, all nodes looks like the following: ")
		for i,c in enumerate(root.children):
			print(i,c)
		root = result
		print("\n")