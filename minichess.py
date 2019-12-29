import numpy as np 
import random
import torch
import normalizer #bit vector representation'a geçtikten sonra kullanılmıyor. ilerde silinecek.

UP = 0
DOWN = 1
RIGHT = 2
LEFT = 3
UP_RIGHT = 4
DOWN_RIGHT = 5
DOWN_LEFT = 6
UP_LEFT = 7
#normalizer = normalizer.MinMaxNormalizer(0,7)

#MiniChess class'ı Board'ı temsil eder. 
class MiniChess():
	def __init__(self, device):
		self.board =  	[
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
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]	#704-768 (beyaz vezir)

		#Colour, Totalmovecount, P1-castling, P2-castling, No-progress count sonradan eklenecek

		self.BlackKingBitonBoard = 4
		self.WhiteKingBitonBoard = 60
		self.BlackKingX = 0
		self.BlackKingY = 5
		self.WhiteKingX = 7
		self.WhiteKingY = 5
		self.available_actions = []
		self.BlackKing = King("black")
		self.WhiteKing = King("white")
		self.ThreatedSquares = []
		self.PossibleMoves = []
		self.device = device
		self.terminal = False

	#Tahtayı başlangıç pozisyonuna getirir. Sonuç olarak, state number'ı döndürür.
	def reset(self):
		self.board =  	[
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
							0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0]	#704-768 (beyaz vezir)

		self.BlackKingBitonBoard = 4
		self.WhiteKingBitonBoard = 60
		self.BlackKingX = 0
		self.BlackKingY = 5
		self.WhiteKingX = 7
		self.WhiteKingY = 5
		self.available_actions = []
		self.BlackKing.reset()
		self.WhiteKing.reset()
		self.ThreatedSquares = []
		self.PossibleMoves = []
		self.terminal = False

		return None

	#Şah'a bir adım attırır ve tahtayı düzenler. Ayrıca Şah objesinin step fonksiyonunu çağırır ve onun da kendisini düzenlemesini sağlar.
	def step(self, action):
		self.board[self.KingX][self.KingY] = "XX"	#Şaha adım attırmadan önce bulunduğu yeri boşluk ile dolduruyoruz.
		self.bitVectorBoard[self.KingBitonBoard] = 0

		reward = 0
		terminal = False

		if action == UP:
			self.WhiteKingX = self.WhiteKingX - 1
			self.WhiteKingBitonBoard -= 8
		elif action == DOWN:
			self.WhiteKingX = self.WhiteKingX + 1
			self.WhiteKingBitonBoard += 8
		elif action == RIGHT:
			self.WhiteKingY = self.WhiteKingY + 1
			self.WhiteKingBitonBoard += 1
		elif action == LEFT:
			self.WhiteKingY = self.WhiteKingY - 1
			self.WhiteKingBitonBoard -= 1
		elif action == UP_RIGHT:
			self.WhiteKingX = self.WhiteKingX - 1
			self.WhiteKingY = self.WhiteKingY + 1
			self.WhiteKingBitonBoard -= 7
		elif action == DOWN_RIGHT:
			self.WhiteKingX = self.WhiteKingX + 1
			self.WhiteKingY = self.WhiteKingY + 1
			self.WhiteKingBitonBoard += 9
		elif action == DOWN_LEFT:
			self.WhiteKingX = self.WhiteKingX + 1
			self.WhiteKingY = self.WhiteKingY - 1
			self.WhiteKingBitonBoard += 7
		elif action == UP_LEFT:
			self.WhiteKingX = self.WhiteKingX - 1
			self.WhiteKingY = self.WhiteKingY - 1
			self.WhiteKingBitonBoard -= 9

		self.board[self.WhiteKingX][self.WhiteKingY] = "K"	#Şahın yeni konumuna notasyonunu ekliyoruz.
		self.bitVectorBoard[self.WhiteKingBitonBoard] = 1
		self.WhiteKing.step(self.WhiteKingBitonBoard)		#Şah objesine adım attırıyoruz.

		#Şimdilik amacımız Şah'ın en üste çıkması olduğundan, X koordinatı 0 ise oyun biter.
		if 	self.WhiteKingBitonBoard < 8:	#self.WhiteKingX == 0:
			reward += 10000
			terminal = True
		else:
			reward -= 10;

		return None, reward, terminal

	#Tehdit edilen kareler hesaplanıyor
	def calculateThreatedSquares(self):
		#Burada her bir karşı alet için hesap yapılacak. Şimdilik default pozisyondaki tehdit edilen kareleri hardcoded verelim.
		ThreatedSquares = [16, 17, 18, 19, 20, 21, 22, 23]	
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

	#Main'den ilk bu çağrılır
	#Sıra: C_A_A -> C_P_M -> King.P_M
	def calculate_available_actions(self):
		possibleMoves = self.calculatePossibleMoves()
		return self.available_actions

	#Tensor coming in, tensor coming out
	def take_action(self, action):
		_, reward, self.terminal = self.step(action.item())
		return torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0), self.terminal

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
			self.KingBitonBoard = 60
		elif color == "black":
			self.notation = "-K"
			self.KingBitonBoard = 4

	def reset(self):
		if self.color == "white":
			self.KingBitonBoard = 60
		elif self.color == "black":
			self.KingBitonBoard = 4

	#MiniChess'teki step fonksiyonu tarafından çağrılır. Şah'ın pozisyonunu düzenler.
	def step(self, KingBitonBoard):
		self.KingBitonBoard = KingBitonBoard

	#Legal hamlelerin koordinat ikililerinin listesini döndürür. Örn:  [[1,2], [1,3]]
	def possibleMoves(self, ThreatedSquares):
		#2d list to keep track of possile moves of the King
		possibleMoves = []
		available_actions = []
		top = False
		bottom = False
		right = False

		#Üst kontrol
		if 	self.KingBitonBoard > 7 :	#Tek renk şah için çalışır durumda. Diğer şahın bit aralık değerleri farklı olacak.
			possibleMoves.append(self.KingBitonBoard - 8) or available_actions.append(UP) if (self.KingBitonBoard - 8) not in ThreatedSquares else 1==1
			top = True
		#Sağ Kontrol
		if 	(self.KingBitonBoard + 1) % 8 != 0:		#self.Y + 1 <= 7:
			possibleMoves.append(self.KingBitonBoard + 1) or available_actions.append(RIGHT) if (self.KingBitonBoard + 1) not in ThreatedSquares else 1==1
			right = True
			if top:	#Sag-ust
				possibleMoves.append(self.KingBitonBoard - 7) or available_actions.append(UP_RIGHT) if (self.KingBitonBoard - 7) not in ThreatedSquares else 1==1
		#Alt Kontrol
		if 	self.KingBitonBoard < 56:		#self.X + 1 <= 7:
			possibleMoves.append(self.KingBitonBoard + 8) or available_actions.append(DOWN) if (self.KingBitonBoard + 8) not in ThreatedSquares else 1==1
			bottom = True
			if right:	#Sag-alt
				possibleMoves.append(self.KingBitonBoard + 9) or available_actions.append(DOWN_RIGHT) if (self.KingBitonBoard + 9) not in ThreatedSquares else 1==1
		#Sol Kontrol
		if 	self.KingBitonBoard % 8 != 0:		#self.Y - 1 >= 0:
			possibleMoves.append(self.KingBitonBoard - 1) or available_actions.append(LEFT) if (self.KingBitonBoard - 1) not in ThreatedSquares else 1==1
			if top:	#Sol-üst
				possibleMoves.append(self.KingBitonBoard - 9) or available_actions.append(UP_LEFT) if (self.KingBitonBoard - 9) not in ThreatedSquares else 1==1
			if bottom:	#Sol-alt
				possibleMoves.append(self.KingBitonBoard + 7) or available_actions.append(DOWN_LEFT) if (self.KingBitonBoard + 7) not in ThreatedSquares else 1==1


		return possibleMoves, available_actions


