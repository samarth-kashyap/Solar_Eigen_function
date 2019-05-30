from time import clock
	
class timestamp:
	def __init__(self):
		self.last_time = clock()
		self.this_time = self.last_time
	def lap(self, label = None):
		self.this_time = clock()
		if(label != None):
			print(label + ': ' + str(self.this_time - self.last_time))
		self.last_time = self.this_time
		
