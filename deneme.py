from copy import copy, deepcopy
import random
import hashlib

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

def check(a):
	a = 10



a = 5
check()
print(a)






