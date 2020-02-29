
class MinMaxNormalizer():
	def __init__(self, minNoProgress, maxNoProgress, minMoveCount, maxMoveCount):
		self.minNoProgress = minNoProgress
		self.maxNoProgress = maxNoProgress
		self.minMoveCount = minMoveCount
		self.maxMoveCount = maxMoveCount

	def normalizeNoProgress(self, count):
		number = (count-self.minNoProgress) / (self.maxNoProgress - self.minNoProgress)
		number = float("{0:.3f}".format(number))
		return number

	def normalizeMoveCount(self, count):
		number = (count-self.minMoveCount) / (self.maxMoveCount - self.minMoveCount)
		number = float("{0:.3f}".format(number))
		return number


