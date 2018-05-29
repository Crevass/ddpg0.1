import numpy as np
import random

class Memory(object):
	def __init__(self, Capacity):
		self.memo_a = []
		self.memo_s = []
		self.memo_r = []
		self.memo_s_ = []
		self.memo_end = []
		self.size = 0
		self.pointer = 0
		self.capacity = Capacity
	def save(self, s, a, r, s_, end):
		if (self.size < self.capacity):
			self.memo_s.append(s)
			self.memo_a.append(a)
			self.memo_r.append(r)
			self.memo_s_.append(s_)
			self.memo_end.append(end)
			self.size = len(self.memo_r)
		else:
			self.memo_s[self.pointer] = s
			self.memo_a[self.pointer] = a
			self.memo_r[self.pointer] = r
			self.memo_s_[self.pointer] = s_
			self.memo_end[self.pointer] = end
			self.size = len(self.memo_r)
		self.pointer = (self.pointer + 1) % self.capacity
		assert self.size <= self.capacity, 'Memory Error'
		assert (self.pointer >= 0 and self.pointer < self.capacity), 'Memory Error'

	def sample(self, batch_size):
		assert self.size >= batch_size, 'Not enough transitions'
		indices = random.sample(range(self.size), batch_size)
		bs, ba, br, bs_, be = [], [], [], [], []
		
		for i in range(batch_size):
			bs.append(self.memo_s[indices[i]])
			ba.append(self.memo_a[indices[i]])
			br.append(self.memo_r[indices[i]])
			bs_.append(self.memo_s_[indices[i]])
			be.append(self.memo_end[indices[i]])

		bs = np.vstack(bs)
		ba = np.vstack(ba)
		br = np.vstack(br)
		bs_ = np.vstack(bs_)
		be = np.vstack(be)
		return bs, ba, br, bs_, be
		
	def reset(self):
		self.memo_a.clear()
		self.memo_s.clear()
		self.memo_r.clear()
		self.memo_s_.clear()
		self.memo_end.clear()
		self.size = 0
		self.pointer = 0
	def get_size(self):
		return len(self.memo_a)