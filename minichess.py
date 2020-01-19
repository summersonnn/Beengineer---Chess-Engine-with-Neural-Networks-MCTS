import numpy as np 
import random
import torch
import actionsdefined as ad
import mcts

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

		self.bitVectorBoard = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,					#18 (siyah şah)
								0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,				#36 (siyah piyonlar)
								1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,				#54 (siyah kaleler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,				#72 (beyaz şah)
								0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,				#90 (beyaz piyonlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1]				#108 (beyaz kaleler)

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
		
		self.WhitePieceList = [King("white", 5, 1), Pawn("white", 4, 0), Pawn("white", 4, 1), Pawn("white", 4, 2), Rook("white", 5, 0), Rook("white", 5, 2)]
		self.BlackPieceList = [King("black", 0, 1), Pawn("black", 1, 0), Pawn("black", 1, 1), Pawn("black", 1, 2), Rook("black", 0, 0), Rook("black", 0, 2)]
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

		self.bitVectorBoard = [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,					#18 (siyah şah)
								0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,				#36 (siyah piyonlar)
								1,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,				#54 (siyah kaleler)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,				#72 (beyaz şah)
								0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,0,0,0,				#90 (beyaz piyonlar)
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1]				#108 (beyaz kaleler)

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
		for i in self.WhitePieceList:
			del i
		for i in self.BlackPieceList:
			del i
		self.WhitePieceList = [King("white", 5, 1), Pawn("white", 4, 0), Pawn("white", 4, 1), Pawn("white", 4, 2), Rook("white", 5, 0), Rook("white", 5, 2)]
		self.BlackPieceList = [King("black", 0, 1), Pawn("black", 1, 0), Pawn("black", 1, 1), Pawn("black", 1, 2), Rook("black", 0, 0), Rook("black", 0, 2)]
		self.ThreatedSquaresByWhite = []
		self.ThreatedSquaresByBlack = []

		return None

	#Şah'a bir adım attırır ve tahtayı düzenler. Ayrıca Şah objesinin step fonksiyonunu çağırır ve onun da kendisini düzenlemesini sağlar.
	def step(self, action):
		print("----------------------STEP------------------")
		inv_actions = {v: k for k, v in ad.actions.items()}
		current_action = inv_actions[action]
		del inv_actions
		#We got the action string e.g "0201K"
		#Now we will edit the board and pass the resulting board to MCSearchTree so that it can return us a reward

		#Get notation before move and current coorbit, empty the squre piece will be moved from, put zero to old coorbit position
		pieceNotationBeforeMove = self.board[int(current_action[0])][int(current_action[1])]	#e.g "+P"
		oldcoorBit = coorToBitVector(int(current_action[0]), int(current_action[1]), pieceNotationBeforeMove) #e.g 30
		self.board[int(current_action[0])][int(current_action[1])] = "XX"
		self.bitVectorBoard[oldcoorBit] = 0

		pieceNotationAfterMove = pieceNotationBeforeMove[0] + current_action[-1]
		color = "white" if pieceNotationAfterMove[0] == "+" else "black"
		promoted = True if len(current_action) == 6 else False
		ListToUse = self.WhitePieceList if pieceNotationAfterMove[0] == '+' else self.BlackPieceList
		otherList = self.WhitePieceList if pieceNotationAfterMove[0] == '-' else self.BlackPieceList
		
		#If capture happened, obtain the BitBoard repr. of captured piece, then remove the piece object from piece object list
		capturedPieceNotation = self.board[int(current_action[2])][int(current_action[3])]
		if capturedPieceNotation != "XX":
			capturedPieceBit = coorToBitVector(int(current_action[2]), int(current_action[3]), capturedPieceNotation)
			self.bitVectorBoard[capturedPieceBit] = 0
			self.removeCapturedPiece(capturedPieceBit, otherList)

		#Update the board, obtain the new Bitboard repr. of the piece and update the bitvectorboard accordingly
		self.board[int(current_action[2])][int(current_action[3])] = pieceNotationAfterMove
		newcoorBit = coorToBitVector(int(current_action[2]), int(current_action[3]), pieceNotationAfterMove)
		self.bitVectorBoard[newcoorBit] = 1

		#Call the step function of the object, to make it renew itself (if the object is still valid, which means promotion did not happen)
		if not promoted:
			for i in ListToUse:
				if i.BitonBoard == oldcoorBit:
					i.step(newcoorBit, int(current_action[2]), int(current_action[3]) ) 
		#if promoted, create a new object and kill the pawn object
		else:
			ListToUse += Rook(color, int(current_action[2]), int(current_action[3]), self)	#Warning! Possible costly operation. Test it.
			#Not captured, but since promoted, pawn object must be deleted
			self.removeCapturedPiece(oldcoorBit, ListToUse)

		reward = 0
		terminal = False

		mcts.initializeTree(self, color, 5)
		#Yeni durumdaki board'ı MCTS'ye, süreyle birlikte ver. 
		#İlk etapta NN'i kullanmayıp MCTS'yi test edelim. Bunun için sürekli explore etmesini sağlayacağım.
		#Bu durumda reward ve terminal hiçbir şey ifade etmeyeceği için, değiştirmiyoruz, 0 kalsınlar.

		return reward, terminal

	def removeCapturedPiece(self, BitonBoard, incomingList):
		for index,i in enumerate(incomingList,0):
			if i.BitonBoard == BitonBoard:
				indexToBeDeleted = index
				break
		del incomingList[indexToBeDeleted]

	#Tehdit edilen kareler hesaplanıyor
	def calculateThreatedSquares(self, color):
		#Burada her bir karşı alet için hesap yapılacak. Şimdilik default pozisyondaki tehdit edilen kareleri hardcoded verelim.
		if color == "white":
			return [65,65,65]	
		elif color == "black":
			return [65,65,65]
	#Tahtayı yazdırır.
	def print(self):
		for i in range(6):
			for j in range(3):
				print(self.board[i][j], end=" ")
			print(" ")

	#Main'den ilk bu çağrılır
	#Sıra: C_A_A -> C_P_M -> King.P_M
	def calculate_available_actions(self, forColor):
		self.ThreatedSquares = self.calculateThreatedSquares(forColor)
		
		if forColor == "white":
			self.available_actions += self.WhitePieceList[0].possibleActions(self.board, self.ThreatedSquares)
			for i in self.WhitePieceList[1:]:
				self.available_actions += i.possibleActions(self.board, self.WhitePieceList[0])

		elif forColor == "black":
			self.available_actions += self.BlackPieceList[0].possibleActions(self.board, self.ThreatedSquares)
			for i in self.BlackPieceList:
				self.available_actions += i.possibleActions(self.board, self.BlackPieceList[0])

		return self.available_actions

	#Tensor coming in, tensor coming out
	def take_action(self, action):
		reward, terminal = self.step(action.item())
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

	def possibleActions(self, theboard, ThreatedSquares, IsForCalculatingThreats=False):
		available_actions = []
		threated_bits = [] #For the enemy king
		top = False
		bottom = False
		right = False
		left = False
	
		#Üst kontrol
		if 	self.X > 0:
			top = True
			threated_bits.append(coorToBitVector(self.X - 1, self.Y, "+K" if self.color == "black" else "-K")) if (self.BitonBoard - 3) not in ThreatedSquares else None
			if theboard[self.X - 1][self.Y][0] != "+" and self.color == "white" or theboard[self.X - 1][self.Y][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 3) not in ThreatedSquares else None
		#Sağ Kontrol
		if 	self.Y < 2:
			right = True
			threated_bits.append(coorToBitVector(self.X, self.Y + 1, "+K" if self.color == "black" else "-K")) if (self.BitonBoard + 1) not in ThreatedSquares else None
			if theboard[self.X][self.Y + 1][0] != "+" and self.color == "white" or theboard[self.X][self.Y + 1][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 1) not in ThreatedSquares else None
				#Sag-ust
				if theboard[self.X - 1][self.Y + 1][0] != "+" and self.color == "white" or theboard[self.X - 1][self.Y + 1][0] != "-" and self.color == "black":	
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + str(self.notation[1])
					available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 2) not in ThreatedSquares else None
		#Alt Kontrol
		if 	self.X < 5:
			bottom = True
			threated_bits.append(coorToBitVector(self.X + 1, self.Y, "+K" if self.color == "black" else "-K")) if (self.BitonBoard + 3) not in ThreatedSquares else None
			if theboard[self.X + 1][self.Y][0] != "+" and self.color == "white" or theboard[self.X + 1][self.Y][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 3) not in ThreatedSquares else None
				#Sag-alt
				if theboard[self.X + 1][self.Y + 1][0] != "+" and self.color == "white" or theboard[self.X + 1][self.Y + 1][0] != "-" and self.color == "black":	
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + str(self.notation[1])
					available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 4) not in ThreatedSquares else None
		#Sol Kontrol
		if 	self.Y > 0:
			left = True
			threated_bits.append(coorToBitVector(self.X, self.Y - 1, "+K" if self.color == "black" else "-K")) if (self.BitonBoard - 1) not in ThreatedSquares else None
			if theboard[self.X][self.Y - 1][0] != "+" and self.color == "white" or theboard[self.X][self.Y - 1][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 1) not in ThreatedSquares else None
				#Sol-üst
				if theboard[self.X - 1][self.Y - 1][0] != "+" and self.color == "white" or theboard[self.X - 1][self.Y - 1][0] != "-" and self.color == "black":	
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + str(self.notation[1])
					available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 4) not in ThreatedSquares else None
				#Sol-alt
				if theboard[self.X + 1][self.Y - 1][0] != "+" and self.color == "white" or theboard[self.X + 1][self.Y - 1][0] != "-" and self.color == "black":	
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + str(self.notation[1])
					available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 2) not in ThreatedSquares else None

		if top and right:
			threated_bits.append(coorToBitVector(self.X - 1, self.Y + 1, "+K" if self.color == "black" else "-K")) if (self.BitonBoard - 2) not in ThreatedSquares else None
		if bottom and right:
			threated_bits.append(coorToBitVector(self.X + 1, self.Y + 1, "+K" if self.color == "black" else "-K")) if (self.BitonBoard + 4) not in ThreatedSquares else None
		if bottom and left:
			threated_bits.append(coorToBitVector(self.X + 1, self.Y - 1, "+K" if self.color == "black" else "-K")) if (self.BitonBoard + 2) not in ThreatedSquares else None
		if left and top:
			threated_bits.append(coorToBitVector(self.X - 1, self.Y - 1, "+K" if self.color == "black" else "-K")) if (self.BitonBoard - 4) not in ThreatedSquares else None

		return available_actions if not IsForCalculatingThreats else threated_bits

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

	def possibleActions(self, theboard, FriendlyKing, IsForCalculatingThreats=False):
	#theboard is the board member in the MiniChess object
		available_actions = []
		threated_bits = [] #For the enemy king

		if self.color == "black":
			#Bir altındakinin (önü) kontrolü
			if theboard[self.X + 1][self.Y] == "XX" and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+1, self.Y):
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + (self.notation[1] if self.X != 4 else "=R")
				available_actions.append(ad.actions[action_string])
				#Başlangıçta iki ileri gidebilme kontrolü
				if self.X == 1 and theboard[self.X + 2][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X + 2) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
			#Sol altındakinin kontrolü
			if 	self.Y > 0 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+1, self.Y-1):
				threated_bits.append(coorToBitVector(self.X + 1, self.Y - 1, "+K"))
				if theboard[self.X + 1][self.Y - 1][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + (self.notation[1] if self.X != 4 else "=R")
					available_actions.append(ad.actions[action_string])
			#Sağ altındakinin kontrolü
			if 	self.Y < 2 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+1, self.Y+1):
				threated_bits.append(coorToBitVector(self.X + 1, self.Y + 1, "+K"))
				if theboard[self.X + 1][self.Y + 1][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + (self.notation[1] if self.X != 4 else "=R")
					available_actions.append(ad.actions[action_string])
			
		else:	#White
			#Bir üsttekinin (önü) kontrolü
			if theboard[self.X - 1][self.Y] == "XX" and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-1, self.Y):
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + (self.notation[1] if self.X != 1 else "=R")
				available_actions.append(ad.actions[action_string])
				#Başlangıçta iki ileri gidebilme kontrolü
				if self.X == 4 and theboard[self.X - 2][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X - 2) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
			#Sol üsttekinin kontrolü
			if 	self.Y > 0 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-1, self.Y-1):
				threated_bits.append(coorToBitVector(self.X - 1, self.Y - 1, "-K"))
				if theboard[self.X - 1][self.Y - 1][0] == "-":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + (self.notation[1] if self.X != 1 else "=R")
					available_actions.append(ad.actions[action_string])
			#Sağ üsttekinin kontrolü
			if 	self.Y < 2 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-1, self.Y+1):
				threated_bits.append(coorToBitVector(self.X - 1, self.Y + 1, "-K"))
				if theboard[self.X - 1][self.Y + 1][0] == "-":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + (self.notation[1] if self.X != 1 else "=R")
					available_actions.append(ad.actions[action_string])


		return available_actions if not IsForCalculatingThreats else threated_bits

	#When switched to full board, check the cases where slope is 1 or -1. There is no need to check for them since there are no bishop or queens to threat the king.
	def IsOkayForKingSafety(self, theboard, FriendlyKing, candidateX, candidateY):
		DistanceVertical = self.X - FriendlyKing.X
		DistanceHorizontal = self.Y - FriendlyKing.Y
		isXGreaterThanKing = 1 if DistanceVertical > 0 else -1
		isYGreaterThanKing = 1 if DistanceHorizontal > 0 else -1
		slope = DistanceVertical / DistanceHorizontal if DistanceHorizontal != 0 else 10

		if slope == 0:
			targetY = FriendlyKing.Y
			while True:
				targetY += isYGreaterThanKing
				#If we're out of board
				if targetY > 2 or targetY < 0:
					return True
				#If there is a piece between current piece and king, then there is no pin which means it's safe to move
				if targetY < self.Y and theboard[self.X][targetY] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if targetY <= self.Y:
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[self.X][targetY][0] == "+" and self.color == "white" or theboard[self.X][targetY][0] == "-" and self.color == "black":
					return True
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[self.X][targetY][1] == "R":
						return False
					else
						return True

		elif slope == 10:
			targetX = FriendlyKing.X
			while True:
				targetX += isXGreaterThanKing
				#If we're out of board
				if targetX > 5 or targetX < 0:
					return True
				#If there is a piece between current piece and king, then there is no pin which means it's safe to move
				if targetX < self.X and theboard[targetX][self.Y] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if targetX <= self.X:
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[targetX][self.Y][0] == "+" and self.color == "white" or theboard[targetX][self.Y][0] == "-" and self.color == "black":
					return True
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[targetX][self.Y][1] == "R":
						return False
					else
						return True
		else:
			return True

class Rook():
	class_counter = 0

	#Minichess object is not NONE if the Rook is initalized via promotion
	def __init__(self, color, x, y, MiniChessObject=None):
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
		
		#Edit the minichess object, so that bitvector is updated after promotion
		if MiniChessObject != None:
			MiniChessObject.bitVectorBoard[self.BitonBoard] = 1


	def step(self, BitonBoard, newX, newY, IsCaptured=False):
		self.BitonBoard = BitonBoard
		self.X = newX
		self.Y = newY

	def possibleActions(self, theboard, FriendlyKing, IsForCalculatingThreats=False):
	#theboard is the board member in the MiniChess object
		available_actions = []
		threated_bits = [] #For the enemy king
		
		#horizontal
		rightFlag = True
		leftFlag = True
		for i in range (1,3):
			#Right
			if rightFlag and self.Y + i < 3 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X, self.Y+i):
				if theboard[self.X][self.Y + i] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X, self.Y + i, "+K" if self.color == "black" else "-K"))

				elif self.color == "white" and theboard[self.X][self.Y + i][0] == "-" or self.color == "black" and theboard[self.X][self.Y + i][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X, self.Y + i, "+K" if self.color == "black" else "-K"))
					rightFlag = False

				else:
					threated_bits.append(coorToBitVector(self.X, self.Y + i, "+K" if self.color == "black" else "-K"))
					rightFlag = False
			#Left
			if leftFlag and self.Y - i >= 0 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X, self.Y-i):
				if theboard[self.X][self.Y - i] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X, self.Y - i, "+K" if self.color == "black" else "-K"))

				elif self.color == "white" and theboard[self.X][self.Y - i][0] == "-" or self.color == "black" and theboard[self.X][self.Y - i][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X, self.Y - i, "+K" if self.color == "black" else "-K"))
					leftFlag = False

				else:
					threated_bits.append(coorToBitVector(self.X, self.Y - i, "+K" if self.color == "black" else "-K"))
					leftFlag = False

		#vertical
		upFlag = True
		downFlag = True
		for i in range (1,6):
			#Down
			if downFlag and self.X + i < 6 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+i, self.Y):
				if theboard[self.X + i][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X + i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X + i, self.Y, "+K" if self.color == "black" else "-K"))

				elif self.color == "white" and theboard[self.X + i][self.Y][0] == "-" or self.color == "black" and theboard[self.X + i][self.Y][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X + i, self.Y, "+K" if self.color == "black" else "-K"))
					downFlag = False

				else:
					threated_bits.append(coorToBitVector(self.X + i, self.Y, "+K" if self.color == "black" else "-K"))
					downFlag = False
			#Up
			if upFlag and self.X - i >= 0 and self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-i, self.Y):
				if theboard[self.X - i][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X - i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X - i, self.Y, "+K" if self.color == "black" else "-K"))

				elif self.color == "white" and theboard[self.X - i][self.Y][0] == "-" or self.color == "black" and theboard[self.X - i][self.Y][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X - i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X - i, self.Y, "+K" if self.color == "black" else "-K"))
					upFlag = False

				else:
					threated_bits.append(coorToBitVector(self.X - i, self.Y, "+K" if self.color == "black" else "-K"))
					upFlag = False

		return available_actions if not IsForCalculatingThreats else threated_bits

	#Same code again! TODO: This can be inherited! 
	#When switched to full board, check the cases where slope is 1 or -1. There is no need to check for them since there are no bishop or queens to threat the king.
	def IsOkayForKingSafety(self, theboard, FriendlyKing, candidateX, candidateY):
		DistanceVertical = self.X - FriendlyKing.X
		DistanceHorizontal = self.Y - FriendlyKing.Y
		isXGreaterThanKing = 1 if DistanceVertical > 0 else -1
		isYGreaterThanKing = 1 if DistanceHorizontal > 0 else -1
		slope = DistanceVertical / DistanceHorizontal if DistanceHorizontal != 0 else 10

		if slope == 0:
			targetY = FriendlyKing.Y
			while True:
				targetY += isYGreaterThanKing
				#If we're out of board
				if targetY > 2 or targetY < 0:
					return True
				#If there is a piece between current piece and king, then there is no pin which means it's safe to move
				if targetY < self.Y and theboard[self.X][targetY] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if targetY <= self.Y:
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[self.X][targetY][0] == "+" and self.color == "white" or theboard[self.X][targetY][0] == "-" and self.color == "black":
					return True
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[self.X][targetY][1] == "R":
						return False
					else
						return True

		elif slope == 10:
			targetX = FriendlyKing.X
			while True:
				targetX += isXGreaterThanKing
				#If we're out of board
				if targetX > 5 or targetX < 0:
					return True
				#If there is a piece between current piece and king, then there is no pin which means it's safe to move
				if targetX < self.X and theboard[targetX][self.Y] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if targetX <= self.X:
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[targetX][self.Y][0] == "+" and self.color == "white" or theboard[targetX][self.Y][0] == "-" and self.color == "black":
					return True
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[targetX][self.Y][1] == "R":
						return False
					else
						return True
		else:
			return True

def coorToBitVector(x, y, notation):
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










