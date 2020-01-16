import numpy as np 
import random
import torch
import sys
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
		
		self.pieceList = [King("black"), Pawn("black", 0), Pawn("black", 1), Pawn("black", 2), Rook("black", 0), Rook("black", 1), King("white"), Pawn("white", 3), Pawn("white", 4), Pawn("white", 5), Rook("white", 2), Rook("white", 3)]
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
		for i in pieceList:
			del i
		self.pieceList = [King("black", 0, 1), Pawn("black", 1, 0), Pawn("black", 1, 1), Pawn("black", 1, 2), Rook("black", 0, 0), Rook("black", 0, 2), King("white", 5, 1), Pawn("white", 4, 0), Pawn("white", 4, 1), Pawn("white", 4, 2), Rook("white", 5, 0), Rook("white", 5, 2)]
		self.ThreatedSquares = []

		return None

	#Şah'a bir adım attırır ve tahtayı düzenler. Ayrıca Şah objesinin step fonksiyonunu çağırır ve onun da kendisini düzenlemesini sağlar.
	def step(self, action):
		inv_actions = {v: k for k, v in ad.actions.items()}
		current_action = inv_actions[action]
		#We got the action number, e.g 112
		#Now we will edit the board and pass the resulting board to MCSearchTree so that it can return us a reward

		#Get notation before move and current coorbit, empty the squre piece will be moved from, put zero to old coorbit position
		pieceNotationBeforeMove = self.board[int(current_action[0])][int(current_action[1])]	#e.g "+P"
		oldcoorBit = coorToBitVector(int(current_action[0]), int(current_action[1]), pieceNotationBeforeMove) #e.g 30
		self.board[int(current_action[0])][int(current_action[1])] = "XX"
		self.bitVectorBoard[oldcoorBit] = 0

		pieceNotationAfterMove = pieceNotationBeforeMove[0] + current_action[-1]
		if len(current_action) == 6:
			promoted = True 
		else:
			promoted = False
		
		#If capture happened, 
		capturedPieceNotation = self.board[int(current_action[2])][int(current_action[3])]
		if capturedPieceNotation != "XX":
			capturedPieceBit = coorToBitVector(int(current_action[2]), int(current_action[3]), capturedPieceNotation)
			self.bitVectorBoard[capturedPieceBit] = 0
			removeCapturedPiece(capturedPieceBit)

		self.board[int(current_action[2])][int(current_action[3])] = pieceNotationAfterMove
		newcoorBit = coorToBitVector(int(current_action[2]), int(current_action[3]), pieceNotationAfterMove)
		self.bitVectorBoard[newcoorBit] = 1

		#Call the step function of the object, to make it renew itself (if the object is still valid, which means promotion did not happen)
		if not promoted:
			for i in pieceList:
				if i.X == int(current_action[0]) and i.Y == int(current_action[1]):
					i.step(newcoorBit, int(current_action[2]), int(current_action[3]) ) 
		#if promoted, create a new object
		else:
			color = "white" if pieceNotationAfterMove[0] == "+" else "black"
			pieceList += Rook(color, int(current_action[2]), int(current_action[3]))

		reward = 0
		terminal = False

		#OYUN AMACINI TASARLA. ONA GORE REWARD MEKANIZMASI KUR
		#Şimdilik amacımız Şah'ın en üste çıkması olduğundan, X koordinatı 0 ise oyun biter.
		'''if 	self.WhiteKingBitonBoard < 8:	#self.WhiteX == 0:
			reward += 10000
			terminal = True
		else:
			reward -= 10;

		return None, reward, terminal'''

	def coorToBitVector(self, x, y, notation):
		coor = x*3 + y
		if notation == "+K":
			return coor
		elif notation == "+P":
			return coor + 18
		elif notation == "+R":
			return coor + 36
		elif notation == "-K":
			return coor + 54
		elif notation == "-P":
			return coor + 72
		elif notation == "-R":
			return coor + 90

	def removeCapturedPiece(self, BitonBoard):
		for index,i in enumerate(pieceList,0):
			if i.BitonBoard == BitonBoard:
				indexToBeDeleted = index
				break
		del pieceList[indexToBeDeleted]

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
		#normalizedState = normalizer.normalize([self.X, self.Y])
		return torch.tensor(self.bitVectorBoard, dtype=torch.float32)

	def get_humanistic_state(self):
		return [self.WhiteX, self.WhiteY]


#Şah class'ıdır. Objesi, MiniChess clasının içinde oluşturulur. Şah objesi, MiniChess clasının bir elemanıdır.
class King():
	class_counter = 0

	def __init__(self, color, x, y):
		self.color = color
		if color == "white":
			self.notation = "+K"
		elif color == "black":
			self.notation = "-K"

		self.X = x
		self.Y = y
		self.BitonBoard = coorToBitVector(x, y, self.notation)
		self.id = King.class_counter
        King.class_counter += 1

	#MiniChess'teki step fonksiyonu tarafından çağrılır. Şah'ın pozisyonunu düzenler.
	def step(self, BitonBoard, newX, newY):
		self.BitonBoard = BitonBoard
		self.X = newX
		self.Y = newY

	def possibleActions(self, ThreatedSquares):
		available_actions = []
		top = False
		bottom = False
		right = False

		#Üst kontrol
		if 	self.BitonBoard > 2 :	#Tek renk şah için çalışır durumda. Diğer şahın bit aralık değerleri farklı olacak.
			action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 3) not in ThreatedSquares else 1==1
			top = True
		#Sağ Kontrol
		if 	(self.BitonBoard + 1) % 3 != 0:		#self.Y + 1 <= 7:
			action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + 1) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 1) not in ThreatedSquares else 1==1
			right = True
			if top:	#Sag-ust
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 2) not in ThreatedSquares else 1==1
		#Alt Kontrol
		if 	self.BitonBoard < 15:		#self.X + 1 <= 7:
			action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 3) not in ThreatedSquares else 1==1
			bottom = True
			if right:	#Sag-alt
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 4) not in ThreatedSquares else 1==1
		#Sol Kontrol
		if 	self.BitonBoard % 3 != 0:		#self.Y - 1 >= 0:
			action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - 1) + str(self.notation[1])
			available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 1) not in ThreatedSquares else 1==1
			if top:	#Sol-üst
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 4) not in ThreatedSquares else 1==1
			if bottom:	#Sol-alt
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 2) not in ThreatedSquares else 1==1

		return available_actions

class Pawn():
	class_counter = 0

	def __init__(self, color, x, y):
		self.color = color
		if color == "white":
			self.notation = "+P"
		elif color == "black":
			self.notation = "-P"

		self.X = x
		self.Y = y
		self.BitonBoard = coorToBitVector(x, y, self.notation)
		self.id = Pawn.class_counter
        Pawn.class_counter += 1

	def step(self, BitonBoard, newX, newY):
		self.BitonBoard = BitonBoard
		self.X = newX
		self.Y = newY

	def possibleActions(self, theboard)
	#theboard is the board member in the MiniChess object
		available_actions = []

		if self.color == "black":
			#Başlangıçta iki ileri gidebilme kontrolü
			if self.X == 1 and theboard[self.X + 1][self.Y] == "XX" and theboard[self.X + 2][self.Y] == "XX":
				action_string = str(self.X) + str(self.Y) + str(self.X + 2) + str(self.Y) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Kaleye çıkma kontrolü
			elif self.X == 4:
				if theboard[self.X + 1][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + "=R"
					available_actions.append(ad.actions[action_string])
				if self.Y > 0 and theboard[self.X + 1][self.Y - 1][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + "=R"
					available_actions.append(ad.actions[action_string])
				if 	self.Y < 2 and	theboard[self.X + 1][self.Y + 1][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + "=R"
					available_actions.append(ad.actions[action_string])

			#Bir altındakinin (önü) kontrolü
			if theboard[self.X + 1][self.Y] == "XX":
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sol altındakinin kontrolü
			if 	self.Y > 0 and	theboard[self.X + 1][self.Y - 1][0] == "+":
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sağ altındakinin kontrolü
			if 	self.Y < 2 and	theboard[self.X + 1][self.Y + 1][0] == "+":
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			
		else:	#White
			#Başlangıçta iki ileri gidebilme kontrolü
			if self.X == 4 and theboard[self.X - 1][self.Y] == "XX" and theboard[self.X - 2][self.Y] == "XX":
				action_string = str(self.X) + str(self.Y) + str(self.X - 2) + str(self.Y) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Kaleye çıkma kontrolü
			elif self.X == 1:
				if theboard[self.X - 1][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + "=R"
					available_actions.append(ad.actions[action_string])
				if self.Y > 0 and theboard[self.X - 1][self.Y - 1][0] == "-":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + "=R"
					available_actions.append(ad.actions[action_string])
				if 	self.Y < 2 and	theboard[self.X - 1][self.Y + 1][0] == "-":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + "=R"
					available_actions.append(ad.actions[action_string])

			#Bir üsttekinin (önü) kontrolü
			if theboard[self.X - 1][self.Y] == "XX":
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sol üsttekinin kontrolü
			if 	self.Y > 0 and	theboard[self.X - 1][self.Y - 1][0] == "-":
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])
			#Sağ üsttekinin kontrolü
			if 	self.Y < 2 and	theboard[self.X - 1][self.Y + 1][0] == "-":
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + self.notation[1]
				available_actions.append(ad.actions[action_string])


		return available_actions

class Rook():
	class_counter = 0

	def __init__(self, color, x, y):
		self.color = color
		if color == "white":
			self.notation = "+R"
		elif color == "black":
			self.notation = "-R"

		self.X = x
		self.Y = y
		self.BitonBoard = coorToBitVector(x, y, self.notation)
		self.id = Rook.class_counter
        Rook.class_counter += 1

	def step(self, BitonBoard, newX, newY, IsCaptured=False):
		self.BitonBoard = BitonBoard
		self.X = newX
		self.Y = newY

	def possibleActions(self, theboard)
	#theboard is the board member in the MiniChess object
		available_actions = []
		
		#horizontal
		rightFlag = True
		leftFlag = True
		for i in range (1,3)
			#Right
			if rightFlag and self.Y + i < 3:
				if theboard[self.X][self.Y + i] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.X][self.Y + i][0] == "-" or self.color == "black" and theboard[self.X][self.Y + i][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					rightFlag = False

				else:
					rightFlag = False
			#Left
			if leftFlag and self.Y - i >= 0:
				if theboard[self.X][self.Y - i] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.X][self.Y - i][0] == "-" or self.color == "black" and theboard[self.X][self.Y - i][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					leftFlag = False

				else:
					leftFlag = False

		#vertical
		upFlag = True
		downFlag = True
		for i in range (1,6)
			#Down
			if downFlag and self.X + i < 6:
				if theboard[self.X + i][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X + i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.X + i][self.Y][0] == "-" or self.color == "black" and theboard[self.X + i][self.Y][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					downFlag = False

				else:
					downFlag = False
			#Up
			if upFlag and self.X - i >= 0:
				if theboard[self.X - i][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X - i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])

				elif self.color == "white" and theboard[self.X - i][self.Y][0] == "-" or self.color == "black" and theboard[self.X - i][self.Y][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X - i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					upFlag = False

				else:
					upFlag = False

		return available_actions










