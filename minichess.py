import numpy as np 
import random
import torch
import normalizer #bit vector representation'a geçtikten sonra kullanılmıyor. ilerde silinecek.

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
#normalizer = normalizer.MinMaxNormalizer(0,7)

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
		self.bitVectorBoard = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		#20
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,	#64 (dost şah)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]	#128 (rakip piyon)
		self.KingBitonBoard = 63
		self.KingX = 7
		self.KingY = 7
		self.available_actions = []
		self.King = King()
		self.ThreatedSquares = []
		self.PossibleMoves = []
		self.device = device
		self.terminal = False

	#Tahtayı başlangıç pozisyonuna getirir. Sonuç olarak, state number'ı döndürür.
	def reset(self):
		self.board =  	[
						["X", "X", "X", "X", "X", "X", "X", "X"],
						["X", "X", "X", "X", "X", "X", "X", "X"],
						["X", "X", "X", "X", "X", "P", "P", "P"],
						["X", "X", "X", "X", "X", "X", "X", "X"],
						["X", "X", "X", "X", "X", "X", "X", "X"],
						["X", "X", "X", "X", "X", "X", "X", "X"],
						["X", "X", "X", "X", "X", "X", "X", "X"],
						["X", "X", "X", "X", "X", "X", "X", "K"],
					]
		self.bitVectorBoard = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		#20
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,	#64 (dost şah)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]	#128 (rakip piyon)
		self.KingBitonBoard = 63
		self.KingX = 7
		self.KingY = 7
		self.available_actions = []
		self.terminal = False
		self.King.reset()

		return None

	#Şah'a bir adım attırır ve tahtayı düzenler. Ayrıca Şah objesinin step fonksiyonunu çağırır ve onun da kendisini düzenlemesini sağlar.
	def step(self, action):
		self.board[self.KingX][self.KingY] = "X"	#Şaha adım attırmadan önce bulunduğu yeri boşluk ile dolduruyoruz.
		self.bitVectorBoard[self.KingBitonBoard] = 0

		reward = 0
		terminal = False

		if action == UP:
			self.KingX = self.KingX - 1
			self.KingBitonBoard -= 8
		elif action == DOWN:
			self.KingX = self.KingX + 1
			self.KingBitonBoard += 8
		elif action == RIGHT:
			self.KingY = self.KingY + 1
			self.KingBitonBoard += 1
		elif action == LEFT:
			self.KingY = self.KingY - 1
			self.KingBitonBoard -= 1

		self.board[self.KingX][self.KingY] = "K"	#Şahın yeni konumuna notasyonunu ekliyoruz.
		self.bitVectorBoard[self.KingBitonBoard] = 1
		self.King.step(self.KingBitonBoard)		#Şah objesine adım attırıyoruz.

		#Şimdilik amacımız Şah'ın en üste çıkması olduğundan, X koordinatı 0 ise oyun biter.
		if 	self.KingBitonBoard < 8:	#self.KingX == 0:
			reward += 100
			terminal = True
		else:
			reward -= 5;

		return None, reward, terminal

	#Tehdit edilen kareler hesaplanıyor
	def calculateThreatedSquares(self):
		#Burada her bir karşı alet için hesap yapılacak. Şimdilik default pozisyondaki tehdit edilen kareleri hardcoded verelim.
		ThreatedSquares = [28, 29, 30, 31]		#[[3,4],[3,5],[3,6],[3,7]]
		return ThreatedSquares

	#Burada her bir alet için possible moves hesaplanacak. Şimdilik tek alet olan şah için hesaplanıyor.
	def calculatePossibleMoves(self):
		self.ThreatedSquares = self.calculateThreatedSquares()
		self.PossibleMoves, self.available_actions = self.King.possibleMoves(self.ThreatedSquares)
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
		return self.available_actions

	def checkIfMoveable(self, action):
		#Yukarı ise
		if action == UP and self.KingBitonBoard > 7:
			return True
		elif action == DOWN and self.KingBitonBoard < 56:
			return True
		elif action == RIGHT and (self.KingBitonBoard + 1) % 8 != 0:
			return True
		elif action == LEFT and self.KingBitonBoard % 8 != 0:
			return True

		return False

	'''#Bu koda göre sayı sürekli değişiyor. Belki fixlemek gerekebilir.
	def num_actions_available(self):
		possible_moves = self.calculatePossibleMoves
		return len(possible_moves)'''

	#Tensor coming in, tensor coming out
	def take_action(self, action):
		_, reward, self.terminal = self.step(action.item())
		return torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0), self.terminal

	def get_state(self):
		#normalizedState = normalizer.normalize([self.KingX, self.KingY])
		return torch.tensor(self.bitVectorBoard, dtype=torch.float32)

	def get_humanistic_state(self):
		return [self.KingX, self.KingY]


#Şah class'ıdır. Objesi, MiniChess clasının içinde oluşturulur. Şah objesi, MiniChess clasının bir elemanıdır.
class King():
	def __init__(self):
		self.notation = "K"
		self.KingBitonBoard = 63

	def reset(self):
		self.KingBitonBoard = 63

	#MiniChess'teki step fonksiyonu tarafından çağrılır. Şah'ın pozisyonunu düzenler.
	def step(self, KingBitonBoard):
		self.KingBitonBoard = KingBitonBoard

	#Legal hamlelerin koordinat ikililerinin listesini döndürür. Örn:  [[1,2], [1,3]]
	def possibleMoves(self, ThreatedSquares):
		#2d list to keep track of possile moves of the King
		possibleMoves = []
		available_actions = []

		#Üst kontrol
		if 	self.KingBitonBoard > 7 :	#Tek renk şah için çalışır durumda. Diğer şahın bit aralık değerleri farklı olacak.
			possibleMoves.append(self.KingBitonBoard - 8) or available_actions.append(UP) if (self.KingBitonBoard - 8) not in ThreatedSquares else 1==1
		#Üst-Sağ Kontrol
		#if self.X - 1 >= 0 and self.Y + 1 <= 3:
		#	possibleMoves.append([self.X - 1, self.Y + 1]) if [self.X - 1, self.Y + 1] not in ThreatedSquares else 1==1
		#Sağ Kontrol
		if 	(self.KingBitonBoard + 1) % 8 != 0:		#self.Y + 1 <= 7:
			possibleMoves.append(self.KingBitonBoard + 1) or available_actions.append(RIGHT) if (self.KingBitonBoard + 1) not in ThreatedSquares else 1==1
		#Alt-Sağ Kontrol
		#if self.X + 1 <= 3 and self.Y + 1 <= 3:
		#	possibleMoves.append([self.X + 1, self.Y + 1]) if [self.X + 1, self.Y + 1] not in ThreatedSquares else 1==1
		#Alt Kontrol
		if 	self.KingBitonBoard < 56:		#self.X + 1 <= 7:
			possibleMoves.append(self.KingBitonBoard + 8) or available_actions.append(DOWN) if (self.KingBitonBoard + 8) not in ThreatedSquares else 1==1
		#Alt-Sol Kontrol
		#if self.X + 1 <= 3 and self.Y - 1 >= 0:
		#	possibleMoves.append([self.X + 1, self.Y - 1]) if [self.X + 1, self.Y - 1] not in ThreatedSquares else 1==1
		#Sol Kontrol
		if 	self.KingBitonBoard % 8 != 0:		#self.Y - 1 >= 0:
			possibleMoves.append(self.KingBitonBoard - 1) or available_actions.append(LEFT) if (self.KingBitonBoard - 1) not in ThreatedSquares else 1==1
		#Üst-Sol kontrol
		#if self.X - 1 >= 0 and self.Y - 1 >= 0:
		#	possibleMoves.append([self.X - 1, self.Y - 1]) if [self.X - 1, self.Y - 1] not in ThreatedSquares else 1==1

		return possibleMoves, available_actions

	def print(self):
		print("King's Position : " + str(self.X) + "," + str(self.Y))
		print("In bitboard: " + str(self.KingBitonBoard))

	def printPossibleMoves(self, ThreatedSquares):
		moves = self.possibleMoves(ThreatedSquares)
		print(moves)
