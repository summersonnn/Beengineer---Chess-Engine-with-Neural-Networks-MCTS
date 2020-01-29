from copy import copy, deepcopy
import random
import hashlib
import dqn

gboard = 	[
							["-R", "-K", "-R"],
							["-P", "-P", "-P"],
							["XX", "XX", "XX"],
							["XX", "XX", "XX"],
							["+P", "+P", "+P"],
							["+R", "+K", "+R"],
						]

class A():
	def	__init__(self, boardobject):
		self.boardobject = boardobject
		self.exclusive_board_string = "" 
	def build_exclusive_string(self):
		self.exclusive_board_string =""
		for i in range(6):
			for j in range(3):
				self.exclusive_board_string += self.boardobject.board[i][j]
	def __hash__(self):
		return int(hashlib.md5(self.exclusive_board_string.encode('utf-8')).hexdigest(),16)
	def __eq__(self,other):
		if hash(self)==hash(other):
			return True
		return False


class B():
	def	__init__(self, gboard):
		self.board = gboard
	def func(self):
		return 5 if 3<2 else 4
	def calc_ava_actions(self):
		self.x = random.randint(0,50)
		self.pfff = 20
	def print(self):
		for i in range(6):
			for j in range(3):
				print(self.board[i][j], end=" ")
			print(" ")

def check():
	print("CHECK")
	if False:
		return 10, 99
	if True:
		return 10
	return 50



tempWhiteMemory = dqn.ReplayMemory(10)


tempWhiteMemory.push(dqn.Experience(10, 10, 10, 0, False))
tempWhiteMemory.push(dqn.Experience(10, 10, 10, 0, False))
tempWhiteMemory.push(dqn.Experience(10, 10, 10, 0, False))
tempWhiteMemory.push(dqn.Experience(10, 10, 10, 0, False))


bigMemory = dqn.ReplayMemory(100)

bigMemory.memory += tempWhiteMemory.memory

for i in range(len(tempWhiteMemory.memory)):
	print(tempWhiteMemory.memory[i])

print("\n\n")

for i in range(len(bigMemory.memory)):
	print(bigMemory.memory[i])

print("\n\n")

del tempWhiteMemory

for i in range(len(bigMemory.memory)):
	print(bigMemory.memory[i])

print("\n\n")


for i in range(len(tempWhiteMemory.memory)):
	print(tempWhiteMemory.memory[i])






