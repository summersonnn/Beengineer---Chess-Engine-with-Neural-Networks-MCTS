import numpy as np 
import random
import torch

#MiniChess class'ı Board'ı temsil eder. 
class MiniChess():
	def __init__(self, device):
		self.board =  	[
							["X", "X", "X", "X","X", "X", "X", "X"],
							["X", "X", "X", "X","X", "X", "X", "X"],
							["X", "X", "X", "X","X", "P", "P", "P"],
							["X", "X", "X", "X","X", "X", "X", "X"],
							["X", "X", "X", "X","X", "X", "X", "X"],
							["X", "X", "X", "X","X", "X", "X", "X"],
							["X", "X", "X", "X","X", "X", "X", "X"],
							["X", "X", "X", "X","X", "X", "X", "K"],
						]
		self.KingX = 7
		self.KingY = 7
		self.King = King()
		self.ThreatedSquares = []
		self.PossibleMoves = []
		self.device = device
		self.done = False

	#Tahtayı başlangıç pozisyonuna getirir. Sonuç olarak, state number'ı döndürür.
	def reset(self):
		self.board =  	[
						["X", "X", "X", "X","X", "X", "X", "X"],
						["X", "X", "X", "X","X", "X", "X", "X"],
						["X", "X", "X", "X","X", "P", "P", "P"],
						["X", "X", "X", "X","X", "X", "X", "X"],
						["X", "X", "X", "X","X", "X", "X", "X"],
						["X", "X", "X", "X","X", "X", "X", "X"],
						["X", "X", "X", "X","X", "X", "X", "X"],
						["X", "X", "X", "X","X", "X", "X", "K"],
					]

		self.KingX = 7
		self.KingY = 7
		self.King.reset()

		return self.KingX * 8 + self.KingY

	#Şah'a bir adım attırır ve tahtayı düzenler. Ayrıca Şah objesinin step fonksiyonunu çağırır ve onun da kendisini düzenlemesini sağlar.
	#Action = Q-table'daki sütun indeksi. (Nereye ilerleyeceğini belirtir.)
	def step(self, action):
		self.board[self.KingX][self.KingY] = "X"
		reward = 0
		done = False

		#Yukarı ise
		if action == 0:
			self.KingX = self.KingX - 1
		elif action == 1:
			self.KingX = self.KingX + 1
		elif action == 2:
			self.KingY = self.KingY + 1
		elif action == 3:
			self.KingY = self.KingY - 1

		self.board[self.KingX][self.KingY] = "K"
		self.King.step(self.KingX, self.KingY)

		if self.KingX == 0:
			reward += 100
			done = True
		else:
			reward -= 1;


		return self.KingX * 8 + self.KingY, reward, done

	#Tehdit edilen kareler hesaplanıyor
	def calculateThreatedSquares(self):
		#Burada her bir siyah alet için hesap yapılacak. Şimdilik default pozisyondaki tehdit edilen kareleri hardcoded verelim.
		ThreatedSquares = [[3,4],[3,5],[3,6],[3,7]]
		return ThreatedSquares

	#Burada her bir beyaz alet için possible moves hesaplanacak. Şimdilik tek alet olan şah için hesaplanıyor.
	def calculatePossibleMoves(self):
		self.ThreatedSquares = self.calculateThreatedSquares()
		self.PossibleMoves = self.King.possibleMoves(self.ThreatedSquares)
		return self.PossibleMoves

	#Tahtayı yazdırır.
	def print(self):
		for i in range(8):
			for j in range(8):
				print(self.board[i][j], end=" ")
			print(" ")
		print("King's Position : " + str(self.KingX) + "," + str(self.KingY))

	def calculate_available_actions(self):
		possibleMoves = self.calculatePossibleMoves()
		available_actions = []
		#newSquare = random.choice(possibleMoves)

		for square in possibleMoves:
			if square[0] == self.KingX - 1:
				available_actions.append(0)
			elif square[0] == self.KingX + 1:
				available_actions.append(1)
			elif square[1] == self.KingY + 1:
				available_actions.append(2)
			elif square[1] == self.KingY - 1: 
				available_actions.append(3)

		return available_actions

	def checkIfMoveable(self, action):
		#Yukarı ise
		if action == 0 and self.KingX - 1 >= 0:
			return True
		elif action == 1 and self.KingX + 1 <= 7:
			return True
		elif action == 2 and self.KingY + 1 <=7:
			return True
		elif action == 3 and self.KingY - 1 >= 0:
			return True

		return False

	'''#Bu koda göre sayı sürekli değişiyor. Belki fixlemek gerekebilir.
	def num_actions_available(self):
		possible_moves = self.calculatePossibleMoves
		return len(possible_moves)'''

	#Tensor coming in, tensor coming out
	def take_action(self, action):
		_, reward, self.done = self.step(action.item())
		return torch.tensor([reward], device=self.device)

	def get_state(self):
		return torch.tensor([self.KingX, self.KingY], dtype=torch.float)


#Şah class'ıdır. Objesi, MiniChess clasının içinde oluşturulur. Şah objesi, MiniChess clasının bir elemanıdır.
class King():
	def __init__(self):
		self.notation = "K"
		self.X = 7
		self.Y = 7

	def reset(self):
		self.X = 7
		self.Y = 7

	#MiniChess'teki step fonksiyonu tarafından çağrılır. Şah'ın pozisyonunu düzenler.
	def step(self, newX, newY):
		self.X = newX
		self.Y = newY

	def possibleMoves(self, ThreatedSquares):
		#2d list to keep track of possile moves of the King
		possibleMoves = []

		#Üst kontrol
		if self.X - 1 >= 0:
			possibleMoves.append([self.X - 1, self.Y]) if [self.X - 1, self.Y] not in ThreatedSquares else 1==1
		#Üst-Sağ Kontrol
		#if self.X - 1 >= 0 and self.Y + 1 <= 3:
		#	possibleMoves.append([self.X - 1, self.Y + 1]) if [self.X - 1, self.Y + 1] not in ThreatedSquares else 1==1
		#Sağ Kontrol
		if self.Y + 1 <= 7:
			possibleMoves.append([self.X, self.Y + 1]) if [self.X, self.Y + 1] not in ThreatedSquares else 1==1
		#Alt-Sağ Kontrol
		#if self.X + 1 <= 3 and self.Y + 1 <= 3:
		#	possibleMoves.append([self.X + 1, self.Y + 1]) if [self.X + 1, self.Y + 1] not in ThreatedSquares else 1==1
		#Alt Kontrol
		if self.X + 1 <= 7:
			possibleMoves.append([self.X + 1, self.Y]) if [self.X + 1, self.Y] not in ThreatedSquares else 1==1
		#Alt-Sol Kontrol
		#if self.X + 1 <= 3 and self.Y - 1 >= 0:
		#	possibleMoves.append([self.X + 1, self.Y - 1]) if [self.X + 1, self.Y - 1] not in ThreatedSquares else 1==1
		#Sol Kontrol
		if self.Y - 1 >= 0:
			possibleMoves.append([self.X, self.Y - 1]) if [self.X, self.Y - 1] not in ThreatedSquares else 1==1
		#Üst-Sol kontrol
		#if self.X - 1 >= 0 and self.Y - 1 >= 0:
		#	possibleMoves.append([self.X - 1, self.Y - 1]) if [self.X - 1, self.Y - 1] not in ThreatedSquares else 1==1

		return possibleMoves

	def print(self):
		print("King's Position : " + str(self.X) + "," + str(self.Y))

	def printPossibleMoves(self, ThreatedSquares):
		moves = self.possibleMoves(ThreatedSquares)
		print(moves)

'''if __name__ == "__main__":
	env = MiniChess()
	moves = env.calculatePossibleMoves()

	print (moves)

	env.step(0,0)'''
