import numpy as np 
import random
import torch
import actionsdefined as ad

#MiniChess class'ı Board'ı temsil eder. 
class MiniChess():
	def __init__(self, device):
		self.board = 	[
							["-R", "-K", "-R"],
							["-P", "-P", "-P"],
							["XX", "XX", "XX"],
							["XX", "XX", "XX"],
							["+P", "+P", "+P"],
							["+R", "+K", "+R"],
						]

		self.bitVectorBoard = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0				#18 (siyah şah)
								0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0				#36 (siyah piyonlar)
								1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0				#54 (siyah kaleler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0				#72 (beyaz şah)
								0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0				#90 (beyaz piyonlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1]			#108 (beyaz kaleler)

		'''self.board =  	[
								["-R", "-N", "-B", "-Q", "-K", "-B", "-N", "-R"],
								["-P", "-P", "-P", "-P", "-P", "-P", "-P", "-P"],
								["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
								["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
								["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
								["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
								["+P", "+P", "+P", "+P", "+P", "+P", "+P", "+P"],
								["+R", "+N", "+B", "+Q", "+K", "+B", "+N", "+R"],
							]

		self.bitVectorBoard = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		#20
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#64 (siyah şah)
								0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#64-128 (siyah piyonlar)
								1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#128-192 (siyah kaleler)
								0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#192-256 (siyah atlar)
								0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#256-320 (siyah filler)
								0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#320-384 (siyah vezir)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,	#384-448 (beyaz şah)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,	#448-512 (beyaz piyonlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,	#512-576 (beyaz kaleler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,	#576-640 (beyaz atlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,	#640-704 (beyaz filler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]	#704-768 (beyaz vezir)'''

		#Colour, Totalmovecount, P1-castling, P2-castling, No-progress count sonradan eklenecek
		self.BlackKing = King("black")
		self.WhiteKing = King("white")
		self.Black_A_Pawn = Pawn("black", 0)
		self.Black_B_Pawn = Pawn("black", 1)
		self.Black_C_Pawn = Pawn("black", 2)
		self.White_A_Pawn = Pawn("white", 3)
		self.White_B_Pawn = Pawn("white", 4)
		self.White_C_Pawn = Pawn("white", 5)
		self.Black_A_Rook = Rook("black", 0)
		self.Black_C_Rook = Rook("black", 1)
		self.White_A_Rook = Rook("white", 2)
		self.White_C_Rook = Rook("white", 3)

		self.available_actions = []
		self.ThreatedSquares = []
		self.device = device

	#Tahtayı başlangıç pozisyonuna getirir. Sonuç olarak, state number'ı döndürür.
	def reset(self):

		self.board = 	[
							["-R", "-K", "-R"],
							["-P", "-P", "-P"],
							["XX", "XX", "XX"],
							["XX", "XX", "XX"],
							["+P", "+P", "+P"],
							["+R", "+K", "+R"],
						]

		self.bitVectorBoard = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0				#18 (siyah şah)
								0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0				#36 (siyah piyonlar)
								1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0				#54 (siyah kaleler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0				#72 (beyaz şah)
								0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0				#90 (beyaz piyonlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1]			#108 (beyaz kaleler)

		'''self.board =  	[
							["-R", "-N", "-B", "-Q", "-K", "-B", "-N", "-R"],
							["-P", "-P", "-P", "-P", "-P", "-P", "-P", "-P"],
							["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
							["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
							["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
							["XX", "XX", "XX", "XX", "XX", "XX", "XX", "XX"],
							["+P", "+P", "+P", "+P", "+P", "+P", "+P", "+P"],
							["+R", "+N", "+B", "+Q", "+K", "+B", "+N", "+R"],
						]
		self.bitVectorBoard = [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		#20
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#64 (siyah şah)
								0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#64-128 (siyah piyonlar)
								1,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#128-192 (siyah kaleler)
								0,1,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#192-256 (siyah atlar)
								0,0,1,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#256-320 (siyah filler)
								0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,	#320-384 (siyah vezir)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,	#384-448 (beyaz şah)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,	#448-512 (beyaz piyonlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,		
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,1,	#512-576 (beyaz kaleler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,1,0,	#576-640 (beyaz atlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,1,0,0,	#640-704 (beyaz filler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]	#704-768 (beyaz vezir)'''

		self.available_actions = []
		self.BlackKing.reset()
		self.Black_A_Pawn.reset()
		self.Black_B_Pawn.reset()
		self.Black_C_Pawn.reset()
		self.Black_A_Rook.reset()
		self.Black_C_Rook.reset()
		self.WhiteKing.reset()
		self.White_A_Pawn.reset()
		self.White_B_Pawn.reset()
		self.White_C_Pawn.reset()
		self.White_A_Rook.reset()
		self.White_C_Rook.reset()
		self.ThreatedSquares = []

		return None

	#Şah'a bir adım attırır ve tahtayı düzenler. Ayrıca Şah objesinin step fonksiyonunu çağırır ve onun da kendisini düzenlemesini sağlar.
	def step(self, action):
		self.board[self.KingX][self.KingY] = "XX"	#Şaha adım attırmadan önce bulunduğu yeri boşluk ile dolduruyoruz.
		self.bitVectorBoard[self.WhiteKingBitonBoard] = 0

		reward = 0
		terminal = False

		if action == UP:
			self.WhiteKingX = self.WhiteKingX - 1
			self.WhiteKingBitonBoard -= 3
		elif action == DOWN:
			self.WhiteKingX = self.WhiteKingX + 1
			self.WhiteKingBitonBoard += 3
		elif action == RIGHT:
			self.WhiteKingY = self.WhiteKingY + 1
			self.WhiteKingBitonBoard += 1
		elif action == LEFT:
			self.WhiteKingY = self.WhiteKingY - 1
			self.WhiteKingBitonBoard -= 1
		elif action == UP_RIGHT:
			self.WhiteKingX = self.WhiteKingX - 1
			self.WhiteKingY = self.WhiteKingY + 1
			self.WhiteKingBitonBoard -= 2
		elif action == DOWN_RIGHT:
			self.WhiteKingX = self.WhiteKingX + 1
			self.WhiteKingY = self.WhiteKingY + 1
			self.WhiteKingBitonBoard += 4
		elif action == DOWN_LEFT:
			self.WhiteKingX = self.WhiteKingX + 1
			self.WhiteKingY = self.WhiteKingY - 1
			self.WhiteKingBitonBoard += 2
		elif action == UP_LEFT:
			self.WhiteKingX = self.WhiteKingX - 1
			self.WhiteKingY = self.WhiteKingY - 1
			self.WhiteKingBitonBoard -= 4

		self.board[self.WhiteKingX][self.WhiteKingY] = "K"	#Şahın yeni konumuna notasyonunu ekliyoruz.
		self.bitVectorBoard[self.WhiteKingBitonBoard] = 1
		self.WhiteKing.step(self.WhiteKingBitonBoard)		#Şah objesine adım attırıyoruz.

		#OYUN AMACINI TASARLA. ONA GORE REWARD MEKANIZMASI KUR
		#Şimdilik amacımız Şah'ın en üste çıkması olduğundan, X koordinatı 0 ise oyun biter.
		'''if 	self.WhiteKingBitonBoard < 8:	#self.WhiteKingX == 0:
			reward += 10000
			terminal = True
		else:
			reward -= 10;

		return None, reward, terminal'''

	#Tehdit edilen kareler hesaplanıyor
	def calculateThreatedSquares(self):
		#Burada her bir karşı alet için hesap yapılacak. Şimdilik default pozisyondaki tehdit edilen kareleri hardcoded verelim.
		ThreatedSquares = [16, 17, 18, 19, 20, 21, 22, 23]	
		return ThreatedSquares

	#Tahtayı yazdırır.
	def print(self):
		for i in range(6):
			for j in range(3):
				print(self.board[i][j], end=" ")
			print(" ")

	#Main'den ilk bu çağrılır
	#Sıra: C_A_A -> C_P_M -> King.P_M
	def calculate_available_actions(self, forColor):
		self.ThreatedSquares = self.calculateThreatedSquares()
		
		if forColor == "white":
			self.available_actions = self.WhiteKing.possibleActions(self.ThreatedSquares)
			self.available_actions += self.White_A_Pawn.possibleActions()
			self.available_actions += self.White_B_Pawn.possibleActions()
			self.available_actions += self.White_C_Pawn.possibleActions()
			self.available_actions += self.White_A_Rook.possibleActions()
			self.available_actions += self.White_C_Rook.possibleActions()
		elif forColor == "black":
			self.available_actions = self.BlackKing.possibleActions(self.ThreatedSquares)
			self.available_actions += self.Black_A_Pawn.possibleActions()
			self.available_actions += self.Black_B_Pawn.possibleActions()
			self.available_actions += self.Black_C_Pawn.possibleActions()
			self.available_actions += self.Black_A_Rook.possibleActions()
			self.available_actions += self.Black_C_Rook.possibleActions()
		return self.available_actions

	#Tensor coming in, tensor coming out
	def take_action(self, action):
		_, reward, terminal = self.step(action.item())
		return torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0), terminal

	def get_state(self):
		#normalizedState = normalizer.normalize([self.KingX, self.KingY])
		return torch.tensor(self.bitVectorBoard, dtype=torch.float32)

	def get_humanistic_state(self):
		return [self.WhiteKingX, self.WhiteKingY]


#Şah class'ıdır. Objesi, MiniChess clasının içinde oluşturulur. Şah objesi, MiniChess clasının bir elemanıdır.
class King():
	def __init__(self, color):
		self.color = color

		if color == "white":
			self.notation = "+K"
			self.KingBitonBoard = 16
			self.KingX = 5
			self.KingY = 1
		elif color == "black":
			self.notation = "-K"
			self.KingBitonBoard = 1
			self.KingX = 0
			self.KingY = 1

	def reset(self):
		if self.color == "white":
			self.KingBitonBoard = 16
			self.KingX = 5
			self.KingY = 1
		elif self.color == "black":
			self.KingBitonBoard = 1
			self.KingX = 0
			self.KingY = 1

	#MiniChess'teki step fonksiyonu tarafından çağrılır. Şah'ın pozisyonunu düzenler.
	def step(self, KingBitonBoard, newX, newY):
		self.KingBitonBoard = KingBitonBoard
		self.KingX = newX
		self.KingY = newY

	def possibleActions(self, ThreatedSquares):
		available_actions = []
		top = False
		bottom = False
		right = False

		#Üst kontrol
		if 	self.KingBitonBoard > 2 :	#Tek renk şah için çalışır durumda. Diğer şahın bit aralık değerleri farklı olacak.
			action_string = str(self.KingX) + str(self.KingY) + str(self.KingX - 1) + str(self.KingY) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard - 3) not in ThreatedSquares else 1==1
			top = True
		#Sağ Kontrol
		if 	(self.KingBitonBoard + 1) % 3 != 0:		#self.Y + 1 <= 7:
			action_string = str(self.KingX) + str(self.KingY) + str(self.KingX) + str(self.KingY + 1) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard + 1) not in ThreatedSquares else 1==1
			right = True
			if top:	#Sag-ust
				action_string = str(self.KingX) + str(self.KingY) + str(self.KingX - 1) + str(self.KingY + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard - 2) not in ThreatedSquares else 1==1
		#Alt Kontrol
		if 	self.KingBitonBoard < 15:		#self.X + 1 <= 7:
			action_string = str(self.KingX) + str(self.KingY) + str(self.KingX + 1) + str(self.KingY) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard + 3) not in ThreatedSquares else 1==1
			bottom = True
			if right:	#Sag-alt
				action_string = str(self.KingX) + str(self.KingY) + str(self.KingX + 1) + str(self.KingY + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard + 4) not in ThreatedSquares else 1==1
		#Sol Kontrol
		if 	self.KingBitonBoard % 3 != 0:		#self.Y - 1 >= 0:
			action_string = str(self.KingX) + str(self.KingY) + str(self.KingX) + str(self.KingY - 1) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard - 1) not in ThreatedSquares else 1==1
			if top:	#Sol-üst
				action_string = str(self.KingX) + str(self.KingY) + str(self.KingX - 1) + str(self.KingY - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard - 4) not in ThreatedSquares else 1==1
			if bottom:	#Sol-alt
				action_string = str(self.KingX) + str(self.KingY) + str(self.KingX + 1) + str(self.KingY - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.KingBitonBoard + 2) not in ThreatedSquares else 1==1

		return available_actions

class Pawn():
	def __init__(self, color, pawnid):
		self.color = color
		self.pawnid = pawnid
		self.IsActive = True

		if color == "white":
			self.notation = "+P"
			self.PawnBitonBoard = pawnid + 9
			self.PawnX = 4
			self.PawnY = pawnid - 3
		elif color == "black":
			self.notation = "-P"
			self.PawnBitonBoard = pawnid + 3
			self.PawnX = 1
			self.PawnY = pawnid

	def reset(self):
		self.IsActive = True
		#PawnBit may change but pawnid never changes. So, reset the pawnbit according to pawnid
		if self.color == "white":
			self.PawnBitonBoard = pawnid + 9
			self.PawnX = 4
			self.PawnY = self.pawnid - 3
		elif self.color == "black":
			self.PawnBitonBoard = pawnid + 3
			self.PawnX = 1
			self.PawnY = self.pawnid

	def step(self, PawnBitonBoard, newX, newY, IsCaptured=False):
		self.PawnBitonBoard = PawnBitonBoard
		self.PawnX = newX
		self.PawnY = newY
		self.IsActive = not IsCaptured

	def possibleActions(self, theboard)
	#theboard is the board member in the MiniChess object
		available_actions = []

		if self.color == "black":
			#Başlangıçta iki ileri gidebilme kontrolü
			if self.PawnX == 1 and theboard[self.PawnX + 1][self.PawnY] == "XX" and theboard[self.PawnX + 2][self.PawnY] == "XX":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 2) + str(self.PawnY) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Kaleye çıkma kontrolü
			elif self.PawnX == 4:
				if theboard[self.PawnX + 1][self.PawnY] == "XX":
					action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 1) + str(self.PawnY) + "=R"
					available_actions.append(ad.actions[action_string])
				if self.PawnY > 0 and theboard[self.PawnX + 1][self.PawnY - 1][0] == "+":
					action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 1) + str(self.PawnY - 1) + "=R"
					available_actions.append(ad.actions[action_string])
				if 	self.PawnY < 2 and	theboard[self.PawnX + 1][self.PawnY + 1][0] == "+":
					action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 1) + str(self.PawnY + 1) + "=R"
					available_actions.append(ad.actions[action_string])

			#Bir altındakinin (önü) kontrolü
			if theboard[self.PawnX + 1][self.PawnY] == "XX":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 1) + str(self.PawnY) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sol altındakinin kontrolü
			if 	self.PawnY > 0 and	theboard[self.PawnX + 1][self.PawnY - 1][0] == "+":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 1) + str(self.PawnY - 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sağ altındakinin kontrolü
			if 	self.PawnY < 2 and	theboard[self.PawnX + 1][self.PawnY + 1][0] == "+":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX + 1) + str(self.PawnY + 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			
		else:	#White
			#Başlangıçta iki ileri gidebilme kontrolü
			if self.PawnX == 4 and theboard[self.PawnX - 1][self.PawnY] == "XX" and theboard[self.PawnX - 2][self.PawnY] == "XX":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 2) + str(self.PawnY) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Kaleye çıkma kontrolü
			elif self.PawnX == 1:
				if theboard[self.PawnX - 1][self.PawnY] == "XX":
					action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 1) + str(self.PawnY) + "=R"
					available_actions.append(ad.actions[action_string])
				if self.PawnY > 0 and theboard[self.PawnX - 1][self.PawnY - 1][0] == "-":
					action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 1) + str(self.PawnY - 1) + "=R"
					available_actions.append(ad.actions[action_string])
				if 	self.PawnY < 2 and	theboard[self.PawnX - 1][self.PawnY + 1][0] == "-":
					action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 1) + str(self.PawnY + 1) + "=R"
					available_actions.append(ad.actions[action_string])

			#Bir üsttekinin (önü) kontrolü
			if theboard[self.PawnX - 1][self.PawnY] == "XX":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 1) + str(self.PawnY) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sol üsttekinin kontrolü
			if 	self.PawnY > 0 and	theboard[self.PawnX - 1][self.PawnY - 1][0] == "-":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 1) + str(self.PawnY - 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sağ üsttekinin kontrolü
			if 	self.PawnY < 2 and	theboard[self.PawnX - 1][self.PawnY + 1][0] == "-":
				action_string = str(self.PawnX) + str(self.PawnY) + str(self.PawnX - 1) + str(self.PawnY + 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])


		return available_actions

class Rook():
	def __init__(self, color, rookid):
		self.color = color
		self.rookid = rookid
		self.IsActive = True

		if color == "white":
			self.notation = "+R"
			self.RookBitonBoard = rookid*2 + 11
			self.RookX = 5
			self.RookY = rookid*2 - 4
		elif color == "black":
			self.notation = "-R"
			self.RookBitonBoard = rookid*2
			self.RookX = 0
			self.RookY = rookid*2

	def reset(self):
		self.IsActive = True

		if self.color == "white":
			self.RookBitonBoard = self.rookid*2 + 11
			self.RookX = 5
			self.RookY = self.rookid*2 - 4
		elif self.color == "black":
			self.RookBitonBoard = self.rookid*2
			self.RookX = 0
			self.RookY = self.rookid*2

	def step(self, RookBitonBoard, newX, newY, IsCaptured=False):
		self.RookBitonBoard = RookBitonBoard
		self.PawnX = newX
		self.PawnY = newY
		self.IsActive = not IsCaptured

	def possibleActions(self, theboard)
	#theboard is the board member in the MiniChess object
		available_actions = []
		
		#horizontal
		rightFlag = True
		leftFlag = True
		for i in range (1,3)
			#Right
			if rightFlag and self.RookY + i < 3:
				if theboard[self.RookX][self.RookY + i] == "XX":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX) + str(self.RookY + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.RookX][self.RookY + i][0] == "-" or self.color == "black" and theboard[self.RookX][self.RookY + i][0] == "+":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX) + str(self.RookY + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					rightFlag = False

				else:
					rightFlag = False
			#Left
			if leftFlag and self.RookY - i >= 0:
				if theboard[self.RookX][self.RookY - i] == "XX":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX) + str(self.RookY - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.RookX][self.RookY - i][0] == "-" or self.color == "black" and theboard[self.RookX][self.RookY - i][0] == "+":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX) + str(self.RookY - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					leftFlag = False

				else:
					leftFlag = False

		#vertical
		upFlag = True
		downFlag = True
		for i in range (1,6)
			#Down
			if downFlag and self.RookX + i < 6:
				if theboard[self.RookX + i][self.RookY] == "XX":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX + i) + str(self.RookY) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.RookX + i][self.RookY][0] == "-" or self.color == "black" and theboard[self.RookX + i][self.RookY][0] == "+":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX + i) + str(self.RookY) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					downFlag = False

				else:
					downFlag = False
			#Up
			if upFlag and self.RookX - i >= 0:
				if theboard[self.RookX - i][self.RookY] == "XX":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX - i) + str(self.RookY) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.RookX - i][self.RookY][0] == "-" or self.color == "black" and theboard[self.RookX - i][self.RookY][0] == "+":
					action_string = str(self.RookX) + str(self.RookY) + str(self.RookX - i) + str(self.RookY) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					upFlag = False

				else:
					upFlag = False

		return available_actions










