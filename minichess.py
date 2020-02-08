import random
from copy import copy, deepcopy
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
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,				#108 (beyaz kaleler)
								0]													#109 -> No Progress Count (Real Valued)

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
		self.checkedby = 0
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
								0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,				#108 (beyaz kaleler)
								0]													#109 -> No Progress Count (Real Valued)

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

		
		for i in self.WhitePieceList:
			del i
		for i in self.BlackPieceList:
			del i
		self.available_actions.clear()
		self.WhitePieceList = [King("white", 5, 1), Pawn("white", 4, 0), Pawn("white", 4, 1), Pawn("white", 4, 2), Rook("white", 5, 0), Rook("white", 5, 2)]
		self.BlackPieceList = [King("black", 0, 1), Pawn("black", 1, 0), Pawn("black", 1, 1), Pawn("black", 1, 2), Rook("black", 0, 0), Rook("black", 0, 2)]
		self.available_actions = []
		self.checkedby = 0
		return None

	#At the moment, runs for user movements. Engine moves use InitializeTree Function in mcts.py which returns best child(continuation) of given state
	def step(self, enemyMove):
		first = 6 - int(enemyMove[1])
		third = 6 - int(enemyMove[3])

		if enemyMove[0] == "a":
			second = 0
		elif enemyMove[0] == "b":
			second = 1  
		elif enemyMove[0] == "c":
			second = 2  
		else:
			raise ValueError('-----WRONG INPUT-----')
		if enemyMove[2] == "a":
			fourth = 0
		elif enemyMove[2] == "b":
			fourth = 1  
		elif enemyMove[2] == "c":
			fourth = 2 
		else:
			raise ValueError('-----WRONG INPUT-----')

		#Get notation before move and current coorbit, empty the squre piece will be moved from, put zero to old coorbit position
		pieceNotationBeforeMove = self.board[first][second]	#e.g "+P"
		oldcoorBit = coorToBitVector(first, second, pieceNotationBeforeMove) #e.g 30
		self.board[first][second] = "XX"
		self.bitVectorBoard[oldcoorBit] = 0

		#Arrange the notation of the piece after move in case of promotion. Also check if there is a promotion.
		pieceNotationAfterMove = pieceNotationBeforeMove
		promoted = False
		if pieceNotationBeforeMove == "+P" and third == 0:
			pieceNotationAfterMove = "+R"
			promoted = True
		elif pieceNotationBeforeMove == "-P" and third == 5:
			pieceNotationAfterMove = "-R"
			promoted = True

		#Increase No Progress count, if there is progress it will be set to zero in the upcoming lines
		self.bitVectorBoard[108] += 1

		#If pawn moves, Set No Progress Count back to 0
		if pieceNotationBeforeMove[1] == "P":
			self.bitVectorBoard[108] = 0

		color = "white" if pieceNotationAfterMove[0] == "+" else "black"
		friendList = self.WhitePieceList if pieceNotationAfterMove[0] == '+' else self.BlackPieceList
		enemyList = self.WhitePieceList if pieceNotationAfterMove[0] == '-' else self.BlackPieceList
		
		#If capture happened, obtain the BitBoard repr. of captured piece, then remove the piece object from piece object list
		#Set No Progress Count back to 0
		capturedPieceNotation = self.board[third][fourth]
		if capturedPieceNotation != "XX":
			capturedPieceBit = coorToBitVector(third, fourth, capturedPieceNotation)
			self.bitVectorBoard[capturedPieceBit] = 0
			self.removeCapturedPiece(capturedPieceBit, enemyList)
			self.bitVectorBoard[108] = 0

		#Update the board, obtain the new Bitboard repr. of the piece and update the bitvectorboard accordingly
		self.board[third][fourth] = pieceNotationAfterMove
		newcoorBit = coorToBitVector(third, fourth, pieceNotationAfterMove)
		self.bitVectorBoard[newcoorBit] = 1

		#Call the step function of the object, to make it renew itself (if the object is still valid, which means promotion did not happen)
		if not promoted:
			for i in friendList:
				if i.BitonBoard == oldcoorBit:
					i.step(newcoorBit, third, fourth) 
		#if promoted, create a new object and kill the pawn object
		else:
			#Not captured, but since promoted, pawn object must be deleted. Also create new Rook object.
			friendList.append(Rook(color, third, fourth))
			self.removeCapturedPiece(oldcoorBit, friendList)

	#Removes the piece from corresponding color's piece list by matching the BitVector of pieces to incoming argument
	def removeCapturedPiece(self, BitonBoard, incomingList):
		for index,i in enumerate(incomingList,0):
			if i.BitonBoard == BitonBoard:
				indexToBeDeleted = index
				break
		del incomingList[indexToBeDeleted]

	#Color is the color the player who may be in check (whom turn to move)
	#Returns the number of checks from pieces. 0 = No check  e.g 2 = Checked by 2 pieces at the same time
	#Also checks direct threats that arise from check, and all threats from enemy pieces.
	#Behindkingbit threat actually arises from DirectThreats but we add it to AllThreats to make things easier in the future.
	def IsCheck(self, color):
		checkedBy = 0
		OurKingX = self.WhitePieceList[0].X if color == "white" else self.BlackPieceList[0].X
		OurKingY = self.WhitePieceList[0].Y if color == "white" else self.BlackPieceList[0].Y

		friendlyList = self.WhitePieceList if color == "white" else self.BlackPieceList
		enemyList = self.WhitePieceList if color == "black" else self.BlackPieceList
		DirectThreatedBits = []
		behindKingBit = []

		#Check for enemy pawns. If we're at the last enemy rank, no enemy pawn can give a check
		if color == "black" and OurKingX != 5:
			if 	OurKingY > 0 and self.board[OurKingX + 1][OurKingY - 1] == "+P":
				checkedBy += 1
				DirectThreatedBits.append(coorToBitVector(OurKingX + 1, OurKingY - 1, "-K"))
			if OurKingY < 2 and self.board[OurKingX + 1][OurKingY + 1] == "+P":
				checkedBy += 1
				DirectThreatedBits.append(coorToBitVector(OurKingX + 1, OurKingY + 1, "-K"))
		if color == "white" and OurKingX != 0:
			if 	OurKingY > 0 and self.board[OurKingX - 1][OurKingY - 1] == "-P": 
				checkedBy += 1
				DirectThreatedBits.append(coorToBitVector(OurKingX - 1, OurKingY - 1, "+K"))
			if OurKingY < 2 and self.board[OurKingX - 1][OurKingY + 1] == "-P":
				checkedBy += 1
				DirectThreatedBits.append(coorToBitVector(OurKingX - 1, OurKingY + 1, "+K"))

		#Check for king rank and king file for enemy rook. Rook only returns the threated bits from Rook to King, not opposite direction
		for i in enemyList:
			if i.notation[1] == "R":
				if (i.X == OurKingX or i.Y == OurKingY) and i.DoesItThreatSquare(self.board, friendlyList[0].X, friendlyList[0].Y):
					direct, behindKingBit = i.possibleActions(self.board, enemyList[0], True, True, OurKingX, OurKingY)
					DirectThreatedBits += direct
					DirectThreatedBits.append(coorToBitVector(i.X, i.Y, "+K" if color == "white" else "-K"))

		#After obtaining Direct threated bits by enemy pieces, we check if our king is in one of them.
		for bit in DirectThreatedBits:
			if friendlyList[0].BitonBoard == bit:
				checkedBy += 1

		self.checkedby = checkedBy

		#We used threatedbits from checking squares to obtain how many checks, but we need to pass all threats in the end so we compute it.
		return checkedBy, DirectThreatedBits, self.calculateThreatedSquares(color) + behindKingBit

	#Calculates threated squares in therms of bit representation.
	#Incoming argument = White means calculate black's pieces' threats (or vice versa)
	def calculateThreatedSquares(self, color):
		if color == "white":
			return self.calculate_available_actions("black", True)	
		elif color == "black":
			return self.calculate_available_actions("white", True)

	#Print the board
	def print(self):
		for i in range(6):
			for j in range(3):
				print(self.board[i][j], end=" ")
			print(" ")

	#If IsForCalculatingThreats is True, this function calculates vectorbits of threated squares, if False, available actions.
	#Checkby is given if we're in check, and it is the number of checks at the same time. If it is not explicitly given, which means we're not in check, it comes as 0
	#checkRookThreats is the threats from enemy Rook piece that provide checks, we will look at these bits to destroy the check
	#checkAllThreats is all threats from enemy pieces
	def calculate_available_actions(self, forColor, IsForCalculatingThreats=False, checkedBy=0, checkRookThreats=None, checkAllThreats=None):
		self.available_actions.clear()
		available_actions = []
		ThreatedSquares = []

		#If this function is called in order to calculate available actions, we FIRST need to calculate threatedsquares. So we fill it!
		if not IsForCalculatingThreats and checkedBy == 0:
			ThreatedSquares = self.calculateThreatedSquares(forColor)

		#But if we are in check, threated squares are already coming as parameter, we use it.
		if checkedBy > 0:
			ThreatedSquares = checkAllThreats
		
		if forColor == "white":
			available_actions += self.WhitePieceList[0].possibleActions(self.board, ThreatedSquares, IsForCalculatingThreats)
			#If we're in check from multiple enemy pieces, we have to move the king so we don't have to calculate av.act. of other pieces
			#If we're calculating legal moves to get rid of the check, we give threatedsquares to possibleActions function of the pieces
			if checkedBy < 2:
				for i in self.WhitePieceList[1:]:
					available_actions += i.possibleActions(self.board, self.WhitePieceList[0], IsForCalculatingThreats) if checkedBy == 0 else i.possibleActions(self.board, self.WhitePieceList[0], False, False, 100, 100, checkAllThreats, checkRookThreats)

		elif forColor == "black":
			available_actions += self.BlackPieceList[0].possibleActions(self.board, ThreatedSquares, IsForCalculatingThreats)
			#If we're in check from multiple enemy pieces, we have to move the king so we don't have to calculate av.act. of other pieces
			if checkedBy < 2:
				for i in self.BlackPieceList[1:]:
					available_actions += i.possibleActions(self.board, self.BlackPieceList[0], IsForCalculatingThreats) if checkedBy == 0 else i.possibleActions(self.board, self.BlackPieceList[0], False, False, 100, 100, checkAllThreats, checkRookThreats)

		#If this function has ran for calculating available actions, we update object's available actions list
		if not IsForCalculatingThreats:
			self.available_actions += available_actions

		return available_actions

	def get_state(self):
		#normalizedState = normalizer.normalize([self.X, self.Y])
		return torch.tensor(self.bitVectorBoard, dtype=torch.float32).to(self.device)

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
			threated_bits.append( coorToBitVector(self.X - 1, self.Y, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard - 3) not in ThreatedSquares else None
			if theboard[self.X - 1][self.Y][0] != "+" and self.color == "white" or theboard[self.X - 1][self.Y][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 3) not in ThreatedSquares else None
		#Sağ Kontrol
		if 	self.Y < 2:
			right = True
			threated_bits.append( coorToBitVector(self.X, self.Y + 1, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard + 1) not in ThreatedSquares else None
			if theboard[self.X][self.Y + 1][0] != "+" and self.color == "white" or theboard[self.X][self.Y + 1][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 1) not in ThreatedSquares else None
			#Sag-ust
			if self.X > 0 and (theboard[self.X - 1][self.Y + 1][0] != "+" and self.color == "white" or theboard[self.X - 1][self.Y + 1][0] != "-" and self.color == "black"):
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 2) not in ThreatedSquares else None
		#Alt Kontrol
		if 	self.X < 5:
			bottom = True
			threated_bits.append( coorToBitVector(self.X + 1, self.Y, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard + 3) not in ThreatedSquares else None
			if theboard[self.X + 1][self.Y][0] != "+" and self.color == "white" or theboard[self.X + 1][self.Y][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 3) not in ThreatedSquares else None
			#Sag-alt
			if self.Y < 2 and (theboard[self.X + 1][self.Y + 1][0] != "+" and self.color == "white" or theboard[self.X + 1][self.Y + 1][0] != "-" and self.color == "black"):	
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 4) not in ThreatedSquares else None
		#Sol Kontrol
		if 	self.Y > 0:
			left = True
			threated_bits.append( coorToBitVector(self.X, self.Y - 1, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard - 1) not in ThreatedSquares else None
			if theboard[self.X][self.Y - 1][0] != "+" and self.color == "white" or theboard[self.X][self.Y - 1][0] != "-" and self.color == "black":
				action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 1) not in ThreatedSquares else None
			#Sol-üst	
			if self.X > 0 and (theboard[self.X - 1][self.Y - 1][0] != "+" and self.color == "white" or theboard[self.X - 1][self.Y - 1][0] != "-" and self.color == "black"):		
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard - 4) not in ThreatedSquares else None
			#Sol-alt
			if self.X < 5 and (theboard[self.X + 1][self.Y - 1][0] != "+" and self.color == "white" or theboard[self.X + 1][self.Y - 1][0] != "-" and self.color == "black"):		
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + str(self.notation[1])
				available_actions.append(ad.actions[action_string]) if (self.BitonBoard + 2) not in ThreatedSquares else None

		if top and right:
			threated_bits.append( coorToBitVector(self.X - 1, self.Y + 1, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard - 2) not in ThreatedSquares else None
		if bottom and right:
			threated_bits.append( coorToBitVector(self.X + 1, self.Y + 1, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard + 4) not in ThreatedSquares else None
		if bottom and left:
			threated_bits.append( coorToBitVector(self.X + 1, self.Y - 1, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard + 2) not in ThreatedSquares else None
		if left and top:
			threated_bits.append( coorToBitVector(self.X - 1, self.Y - 1, "+K" if self.color == "black" else "-K") ) if (self.BitonBoard - 4) not in ThreatedSquares else None

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

	#checkDirectThreats is NOT false whenever we're calculating legal moves to get rid of the check
	#EnemyKingX, EnemyKingY are reduntant here. Remove these parameters once you correct all calls to here.
	#checkDirectThreats is the threats from enemy piece that provide checks, we will look at these bits to destroy the check
	#CheckThreats is redundant
	def possibleActions(self, theboard, FriendlyKing, IsForCalculatingThreats=False, IsForCheck=False, EnemyKingX=None, EnemyKingY=None, checkThreats=None, checkDirectThreats=None):
	#theboard is the board member in the MiniChess object
		available_actions = []
		threated_bits = [] #For the enemy king
	
		if self.color == "black":
			#Bir altındakinin (önü) kontrolü
			if theboard[self.X + 1][self.Y] == "XX" and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+1, self.Y)) and (checkDirectThreats == None or coorToBitVector(self.X + 1, self.Y, "-K") in checkDirectThreats):
				action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y) + (self.notation[1] if self.X != 4 else "=R")
				available_actions.append(ad.actions[action_string])
				#Başlangıçta iki ileri gidebilme kontrolü
				if self.X == 1 and theboard[self.X + 2][self.Y] == "XX" and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X + 2, self.Y)) and (checkDirectThreats == None or coorToBitVector(self.X + 2, self.Y, "-K") in checkDirectThreats):
					action_string = str(self.X) + str(self.Y) + str(self.X + 2) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
			#Sol altındakinin kontrolü
			if 	self.Y > 0 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+1, self.Y-1)) and (checkDirectThreats == None or coorToBitVector(self.X + 1, self.Y - 1, "-K") in checkDirectThreats):
				threated_bits.append(coorToBitVector(self.X + 1, self.Y - 1, "+K"))
				if theboard[self.X + 1][self.Y - 1][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y - 1) + (self.notation[1] if self.X != 4 else "=R")
					available_actions.append(ad.actions[action_string])
			#Sağ altındakinin kontrolü
			if 	self.Y < 2 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+1, self.Y+1)) and (checkDirectThreats == None or coorToBitVector(self.X + 1, self.Y + 1, "-K") in checkDirectThreats):
				threated_bits.append(coorToBitVector(self.X + 1, self.Y + 1, "+K"))
				if theboard[self.X + 1][self.Y + 1][0] == "+":
					action_string = str(self.X) + str(self.Y) + str(self.X + 1) + str(self.Y + 1) + (self.notation[1] if self.X != 4 else "=R")
					available_actions.append(ad.actions[action_string])
			
		else:	#White
			#Bir üsttekinin (önü) kontrolü
			if theboard[self.X - 1][self.Y] == "XX" and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-1, self.Y)) and (checkDirectThreats == None or coorToBitVector(self.X - 1, self.Y, "+K") in checkDirectThreats):
				action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y) + (self.notation[1] if self.X != 1 else "=R")
				available_actions.append(ad.actions[action_string])
				#Başlangıçta iki ileri gidebilme kontrolü
				if self.X == 4 and theboard[self.X - 2][self.Y] == "XX" and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X - 2, self.Y)) and (checkDirectThreats == None or coorToBitVector(self.X - 2, self.Y, "+K") in checkDirectThreats):
					action_string = str(self.X) + str(self.Y) + str(self.X - 2) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
			#Sol üsttekinin kontrolü
			if 	self.Y > 0 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-1, self.Y-1)) and (checkDirectThreats == None or coorToBitVector(self.X - 1, self.Y - 1, "+K") in checkDirectThreats):
				threated_bits.append(coorToBitVector(self.X - 1, self.Y - 1, "-K"))
				if theboard[self.X - 1][self.Y - 1][0] == "-":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y - 1) + (self.notation[1] if self.X != 1 else "=R")
					available_actions.append(ad.actions[action_string])
			#Sağ üsttekinin kontrolü
			if 	self.Y < 2 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-1, self.Y+1)) and (checkDirectThreats == None or coorToBitVector(self.X - 1, self.Y + 1, "+K") in checkDirectThreats):
				threated_bits.append(coorToBitVector(self.X - 1, self.Y + 1, "-K"))
				if theboard[self.X - 1][self.Y + 1][0] == "-":
					action_string = str(self.X) + str(self.Y) + str(self.X - 1) + str(self.Y + 1) + (self.notation[1] if self.X != 1 else "=R")
					available_actions.append(ad.actions[action_string])
	
		if IsForCheck or IsForCalculatingThreats:
			return threated_bits
		
		return available_actions

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
				if (targetY < self.Y and isYGreaterThanKing == 1 or targetY > self.Y and isYGreaterThanKing == -1) and theboard[self.X][targetY] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if (targetY <= self.Y and isYGreaterThanKing == 1 or targetY >= self.Y and isYGreaterThanKing == -1):
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[self.X][targetY][0] == "+" and self.color == "white" or theboard[self.X][targetY][0] == "-" and self.color == "black":
					return True
				#If it is empty, pass (The square from piece to opposite direction of king)	
				elif theboard[self.X][targetY] == "XX":
					continue
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[self.X][targetY][1] == "R" and not isInBetween(self.X, targetY, FriendlyKing.X, FriendlyKing.Y, candidateX, candidateY):
						return False
					else:
						return True

		elif slope == 10:
			targetX = FriendlyKing.X
			while True:
				targetX += isXGreaterThanKing
				#If we're out of board
				if targetX > 5 or targetX < 0:
					return True
				#If there is a piece between current piece and king, then there is no pin which means it's safe to move
				if (targetX < self.X and isXGreaterThanKing == 1 or targetX > self.X and isXGreaterThanKing == -1) and theboard[targetX][self.Y] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if (targetX <= self.X and isXGreaterThanKing == 1 or targetX >= self.X and isXGreaterThanKing == -1):
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[targetX][self.Y][0] == "+" and self.color == "white" or theboard[targetX][self.Y][0] == "-" and self.color == "black":
					return True
				#If it is empty, pass (The square from piece to opposite direction of king)	
				elif theboard[targetX][self.Y] == "XX":
					continue
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[targetX][self.Y][1] == "R" and not isInBetween(targetX, self.Y, FriendlyKing.X, FriendlyKing.Y, candidateX, candidateY):
						return False
					else:
						return True
		else:
			return True

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

	#IsForCheck is used when we're checking if player A is in check or not. To do this, we check the possible Rook moves of player B.
	#EnemyKing coordinates are given if IsForCheck = True
	#checkDirectThreats is the threats from enemy piece that provide checks, we will look at these bits to destroy the check
	#checkDirectThreats is NOT false whenever we're calculating legal moves to get rid of the check
	#CheckThreats is redundant
	def possibleActions(self, theboard, FriendlyKing, IsForCalculatingThreats=False, IsForCheck=False, EnemyKingX=None, EnemyKingY=None, checkThreats=None, checkDirectThreats=None):
	#theboard is the board member in the MiniChess object
		available_actions = []
		threated_bits = [] #For the enemy king
		behindKingBit = []	#Bit behind the king

		#CheckInX = Check in Horizontal Line
		#CheckInY = Check in Vertical Line
		checkInX = True if EnemyKingX == self.X else False
		checkInY = True if EnemyKingY == self.Y else False
		
		#horizontal
		rightFlag = True
		leftFlag = True
		for i in range (1,3):
			#Right - About the condition with abs: Whenever IsForCheck is True, We ONLY want the bits from Rook to King, not opposite direction
			if rightFlag and self.Y + i < 3 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X, self.Y+i)) and (not IsForCheck or (checkInX and	abs(self.Y - EnemyKingY) >= abs(self.Y + i - EnemyKingY))) and (checkDirectThreats == None or coorToBitVector(self.X, self.Y + i, "-K" if self.color == "black" else "+K") in checkDirectThreats):
				if theboard[self.X][self.Y + i] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X, self.Y + i, "+K" if self.color == "black" else "-K"))

				elif rightFlag and (self.color == "white" and theboard[self.X][self.Y + i][0] == "-" or self.color == "black" and theboard[self.X][self.Y + i][0] == "+"):
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y + i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					rightFlag = False
					#If for check threats, don't label enemy pieces as threat except the enemy king of course (from the rook)
					if not IsForCheck or theboard[self.X][self.Y + i][1] == "K"  : threated_bits.append(coorToBitVector(self.X, self.Y + i, "+K" if self.color == "black" else "-K"))
					
					#The square behind the enemy king, will also be threatened, so we add it. (If we're calculating for checks)
					if IsForCheck and self.Y + i + 1 < 3:
						behindKingBit.append(coorToBitVector(self.X, self.Y + i + 1, "+K" if self.color == "black" else "-K"))

				else:
					#If for check threats, don't label our pieces as threat (from the rook)
					rightFlag = False
					if not IsForCheck: threated_bits.append(coorToBitVector(self.X, self.Y + i, "+K" if self.color == "black" else "-K")) 
					
			#Left - About the condition with abs: Whenever IsForCheck is True, We ONLY want the bits from Rook to King, not opposite direction
			if leftFlag and self.Y - i >= 0 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X, self.Y-i)) and (not IsForCheck or (checkInX and	abs(self.Y - EnemyKingY) >= abs(self.Y - i - EnemyKingY))) and (checkDirectThreats == None or coorToBitVector(self.X, self.Y - i, "-K" if self.color == "black" else "+K") in checkDirectThreats):
				if theboard[self.X][self.Y - i] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X, self.Y - i, "+K" if self.color == "black" else "-K"))

				elif leftFlag and (self.color == "white" and theboard[self.X][self.Y - i][0] == "-" or self.color == "black" and theboard[self.X][self.Y - i][0] == "+"):
					action_string = str(self.X) + str(self.Y) + str(self.X) + str(self.Y - i) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					leftFlag = False
					#If for check threats, don't label enemy pieces as threat except the enemy king of course (from the rook)
					if not IsForCheck or theboard[self.X][self.Y - i][1] == "K" : threated_bits.append(coorToBitVector(self.X, self.Y - i, "+K" if self.color == "black" else "-K"))
					
					#The square behind the enemy king, will also be threatened, so we add it. (If we're calculating for checks)
					if IsForCheck and self.Y - i - 1 >=0:
						behindKingBit.append(coorToBitVector(self.X, self.Y - i - 1, "+K" if self.color == "black" else "-K"))

				else:
					leftFlag = False
					#If for check threats, don't label our pieces as threat (from the rook)
					if not IsForCheck: threated_bits.append(coorToBitVector(self.X, self.Y - i, "+K" if self.color == "black" else "-K")) 
					

		#vertical
		upFlag = True
		downFlag = True
		for i in range (1,6):
			#Down - About the last condition: Whenever IsForCheck is True, We ONLY want the bits from Rook to King, not opposite direction
			if downFlag and self.X + i < 6 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X+i, self.Y)) and (not IsForCheck or (checkInY and abs(self.X - EnemyKingX) >= abs(self.X + i - EnemyKingX))) and (checkDirectThreats == None or coorToBitVector(self.X+i, self.Y, "-K" if self.color == "black" else "+K") in checkDirectThreats):
				if theboard[self.X + i][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X + i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X + i, self.Y, "+K" if self.color == "black" else "-K"))

				elif downFlag and (self.color == "white" and theboard[self.X + i][self.Y][0] == "-" or self.color == "black" and theboard[self.X + i][self.Y][0] == "+"):
					action_string = str(self.X) + str(self.Y) + str(self.X + i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					downFlag = False
					#If for check threats, don't label enemy pieces as threat except the enemy king of course (from the rook)
					if not IsForCheck or theboard[self.X + i][self.Y][1] == "K" : threated_bits.append(coorToBitVector(self.X + i, self.Y, "+K" if self.color == "black" else "-K"))
					
					#The square behind the enemy king, will also be threatened, so we add it. (If we're calculating for checks)
					if IsForCheck and self.X + i + 1 < 6:
						behindKingBit.append(coorToBitVector(self.X + i + 1, self.Y, "+K" if self.color == "black" else "-K"))

				else:
					downFlag = False
					#If for check threats, don't label our pieces as threat (from the rook)
					if not IsForCheck: threated_bits.append(coorToBitVector(self.X + i, self.Y, "+K" if self.color == "black" else "-K"))
					
			#Up - About the last condition: Whenever IsForCheck is True, We ONLY want the bits from Rook to King, not opposite direction
			if upFlag and self.X - i >= 0 and (IsForCalculatingThreats or self.IsOkayForKingSafety(theboard, FriendlyKing, self.X-i, self.Y)) and (not IsForCheck or (checkInY and abs(self.X - EnemyKingX) >= abs(self.X - i - EnemyKingX))) and (checkDirectThreats == None or coorToBitVector(self.X-i, self.Y, "-K" if self.color == "black" else "+K") in checkDirectThreats):
				if theboard[self.X - i][self.Y] == "XX":
					action_string = str(self.X) + str(self.Y) + str(self.X - i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					threated_bits.append(coorToBitVector(self.X - i, self.Y, "+K" if self.color == "black" else "-K"))

				elif upFlag and (self.color == "white" and theboard[self.X - i][self.Y][0] == "-" or self.color == "black" and theboard[self.X - i][self.Y][0] == "+"):
					action_string = str(self.X) + str(self.Y) + str(self.X - i) + str(self.Y) + self.notation[1]
					available_actions.append(ad.actions[action_string])
					upFlag = False
					#If for check threats, don't label enemy pieces as threat except the enemy king of course (from the rook)
					if not IsForCheck or theboard[self.X - i][self.Y][1] == "K" : threated_bits.append(coorToBitVector(self.X - i, self.Y, "+K" if self.color == "black" else "-K"))
					
					#The square behind the enemy king, will also be threatened, so we add it. (If we're calculating for checks)
					if IsForCheck and self.X - i - 1 >=0:
						behindKingBit.append(coorToBitVector(self.X - i - 1, self.Y, "+K" if self.color == "black" else "-K"))

				else:
					upFlag = False
					#If for check threats, don't label our pieces as threat (from the rook)
					if not IsForCheck: threated_bits.append(coorToBitVector(self.X - i, self.Y, "+K" if self.color == "black" else "-K"))
					

		if IsForCheck:
			return threated_bits, behindKingBit
		if IsForCalculatingThreats:
			return threated_bits
		return available_actions

	#Check if the piece threats the given coordinate (False if you give self coordinates)
	def DoesItThreatSquare(self, theboard, coorX, coorY):
		IsSameX = True if self.X == coorX else False
		IsSameY = True if self.Y == coorY else False

		if IsSameX == False and IsSameY == False:
			return False

		leftFlag = True
		rightFlag = True
		upFlag = True
		downFlag = True

		if IsSameX:
			for i in range(1,3):
				if (leftFlag and self.Y - i >=0 and	self.Y - i == coorY):
					return True
				elif leftFlag and (self.Y - i < 0 or theboard[self.X][self.Y - i] != "XX"):
					leftFlag = False
					
				if (rightFlag and self.Y + i < 3 and self.Y + i == coorY):
					return True
				elif rightFlag and (self.Y + i > 2 or theboard[self.X][self.Y + i] != "XX"):
					rightFlag = False
		if IsSameY:
			for i in range(1,6):
				if (upFlag and self.X - i >=0 and self.X - i == coorX):
					return True
				elif upFlag and (self.X - i < 0 or theboard[self.X - i][self.Y] != "XX"):
					upFlag = False
					
				if (downFlag and self.X + i < 6 and self.X + i == coorX):
					return True
				elif downFlag and (self.X + i > 5 or theboard[self.X + i][self.Y] != "XX"):
					downFlag = False


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
				if (targetY < self.Y and isYGreaterThanKing == 1 or targetY > self.Y and isYGreaterThanKing == -1) and theboard[self.X][targetY] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if (targetY <= self.Y and isYGreaterThanKing == 1 or targetY >= self.Y and isYGreaterThanKing == -1):
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[self.X][targetY][0] == "+" and self.color == "white" or theboard[self.X][targetY][0] == "-" and self.color == "black":
					return True
				#If it is empty, pass (The square from piece to opposite direction of king)
				elif theboard[self.X][targetY] == "XX":
					continue
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[self.X][targetY][1] == "R" and not isInBetween(self.X, targetY, FriendlyKing.X, FriendlyKing.Y, candidateX, candidateY):
						return False
					else:
						return True

		elif slope == 10:
			targetX = FriendlyKing.X
			while True:
				targetX += isXGreaterThanKing
				#If we're out of board
				if targetX > 5 or targetX < 0:
					return True
				#If there is a piece between current piece and king, then there is no pin which means it's safe to move
				if (targetX < self.X and isXGreaterThanKing == 1 or targetX > self.X and isXGreaterThanKing == -1) and theboard[targetX][self.Y] != "XX":
					return True
				#Pass the current piece or inbetween empty squares (It's empty since it didn't get caught by the if block above)
				if (targetX <= self.X and isXGreaterThanKing == 1 or targetX >= self.X and isXGreaterThanKing == -1):
					continue
				#If the first piece on the way is friendly (from piece to opposite direction of king), then it's safe
				if theboard[targetX][self.Y][0] == "+" and self.color == "white" or theboard[targetX][self.Y][0] == "-" and self.color == "black":
					return True
				#If it is empty, pass (The square from piece to opposite direction of king)
				elif theboard[targetX][self.Y] == "XX":
					continue
				#There is an enemy piece on the way! There MAY be a pin. Check the enemy piece.
				else:
					#If it is an enemy rook, then we are in pin. Add Queen here when switched to full board.
					if theboard[targetX][self.Y][1] == "R" and not isInBetween(targetX, self.Y, FriendlyKing.X, FriendlyKing.Y, candidateX, candidateY):
						return False
					else:
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

#Decides if the given square is in between of other given 2 squares
#For now, calculates only for straight lines. Add diagonals when bishop added.
def isInBetween(op1x, op1y, op2x, op2y, inx, iny):

	IsSameX = True if op1x == op2x else False
	IsSameY = True if op1y == op2y else False

	if IsSameX == False and IsSameY == False:
		return False

	if IsSameX:
		'''if abs(op1y - op2y) > abs(op1y - iny):
			return True'''
		if inx == op2x:	#or inx == op1x
			return True
		return False
	if IsSameY:
		if iny == op2y:	#or iny == op1y
			return True
		return False












