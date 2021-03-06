import numpy as np


class meanstdFilter(object):
	def __init__(self, epsilon = 1e-4, shape = ()):
		self.mean = np.zeros(shape, 'float32')
		self.var = np.ones(shape, 'float32')
		self.count = epsilon

	def update(self, x):
		if x.ndim < 2:
			x = x[np.newaxis, :]
		batch_mean = np.mean(x, axis=0)
		batch_var = np.var(x, axis=0)
		batch_count = x.shape[0]
		self.update_from_moments(batch_mean, batch_var, batch_count)

	def update_from_moments(self, batch_mean, batch_var, batch_count):
		delta = batch_mean -self.mean
		total_count = self.count + batch_count

		new_mean = self.mean + delta * batch_count / total_count
		m_a = self.var * (self.count)
		m_b = batch_var * (batch_count)
		M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
		new_var = M2 / (self.count + batch_count)

		new_count = batch_count + self.count

		self.mean = new_mean
		self.var = new_var
		self.count = new_count

	def filtting(self, x):
		return (x - self.mean) / (np.sqrt(self.var))