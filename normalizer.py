
class MinMaxNormalizer():
	def __init__(self, minnumber, maxnumber):
		self.min = minnumber
		self.max = maxnumber

	def normalize(self, state):
		outstate = []
		for index, val in enumerate(state):
			number = (state[index]-self.min) / (self.max - self.min)
			number = float("{0:.3f}".format(number))
			outstate.append(number)

		return outstate