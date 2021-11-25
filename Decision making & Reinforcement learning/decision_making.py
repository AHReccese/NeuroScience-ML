import numpy as np

np.random.seed(0)
num_objects = 10
num_stimuli = 4

class Shadlen:

	def __init__(self):
		self.actions_correctness = []
		self.step = 0
		self.criterion = 0.5
		self.stimuli = self.generate_stimuli(num_stimuli)
		self.state = self.to_one_hot([self.step] + self.stimuli)
	
	def to_one_hot(self, array, N=10):
		array = np.array(array).astype(int)
		labels = np.zeros((array.shape[0], N), dtype=np.float32)
		labels[np.arange(array.shape[0]), array] = 1.
		return labels
		
	def generate_stimuli(self, N_s):
		stimuli = []
		for i in range(N_s):
			stimuli.append(np.random.randint(0, num_objects))
		return stimuli
		
	def reset(self):
		self.__init__()
		return self.state
		
	def response(self, action):
		if self.step == 1:
			GT = float((np.sum(self.stimuli) / (num_objects * num_stimuli)) < self.criterion)
		else:
			GT = float((np.sum(self.stimuli) / (num_objects * num_stimuli)) > self.criterion)
			
		self.actions_correctness.append(GT == action)
		self.step += 1
		
		finished = False
		
		if self.step == 1:
			reward = 0.
			
		elif self.step == 2:
			if self.actions_correctness[-2]:
				reward = 0.
			else:
				finished = True
				if self.actions_correctness[-1]:
					reward = 1.2
				else:
					reward = -2.

		elif self.step == 3:		
			finished = True

			if self.actions_correctness[-2]:
				if self.actions_correctness[-1]:
					reward = 3.5
				else:
					reward = 0.5
			else:
				if self.actions_correctness[-1]:
					reward = 0.
				else:
					reward = -1.					
		else:
			print("StepErr!")
			return

		self.stimuli = self.generate_stimuli(num_stimuli)
		self.state = self.to_one_hot([self.step] + self.stimuli)
		
		return reward, self.state, finished
	