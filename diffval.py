class DiffVal:
	def __init__(self, val):
		self.val = val
	def __setattr__(self, name, value):
		assert(name == 'val', "no, you can't do that")
		if hasattr(self, 'val'):
			self.__dict__['prev_val'] = self.val
			self.__dict__['val'] = value
		else:
			self.__dict__['prev_val'] = value
			self.__dict__['val'] = value
	def delta(self):
		return self.val - self.prev_val