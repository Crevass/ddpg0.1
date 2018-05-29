import tensorflow as tf
import numpy as np

L1NUM = 64
L2NUM = 64
L3NUM = 64
L4NUM = 64

ACTIVITY_SCALE = 0
L2_SCALE = 0
GAMMA = 0.99

LR_C = 0.0002
LR_A = 0.0001

TAU_A = 0.2
TAU_C = 0.2



class DDPG(object):
	def __init__(self, a_dim, s_dim, a_bound, sess):
		self.s1 = tf.placeholder(tf.float32, [None, s_dim])
		self.s2 = tf.placeholder(tf.float32, [None, s_dim])
		self.r = tf.placeholder(tf.float32, [None, 1])
		self.terminate = tf.placeholder(tf.float32, [None, 1])
		self.sess = sess


		self.actor1 = self._build_actor(a_dim, s_dim, a_bound, self.s1, True, 'Actor1')
		self.actor2 = self._build_actor(a_dim, s_dim, a_bound, self.s2, False, 'Actor2')
	
		self.critic1 = self._build_critic(a_dim, s_dim, self.actor1, self.s1, True, 'Critic1')
		self.critic2 = self._build_critic(a_dim, s_dim, self.actor2, self.s2, False, 'Critic2')

		self.a_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor1')
		self.a_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor2')

		self.c_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic1')
		self.c_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic2')

		self.loss_q = self._q_loss(self.r, self.critic1, self.critic2, self.terminate)
		self.loss_a = -tf.reduce_mean(self.critic1, name='a_loss')
		
		with tf.name_scope('C_train'):
			self.c_train = tf.train.AdamOptimizer(LR_C).minimize(self.loss_q, var_list = self.c_params1)
		with tf.name_scope('A_train'):
			self.a_train = tf.train.AdamOptimizer(LR_A).minimize(self.loss_a, var_list = self.a_params1)

		self.sample_action = tf.clip_by_value(tf.squeeze(self.actor1), -a_bound, a_bound)
		with tf.name_scope('Update_A'):
			self.update_old_a = [old.assign(old*(1-TAU_A) + new*TAU_A) for old, new in zip(self.a_params2, self.a_params1)]
		with tf.name_scope('Update_C'):
			self.update_old_q = [old.assign(old*(1-TAU_C) + new*TAU_C) for old, new in zip(self.c_params2, self.c_params1)]


	def _dense_layer(self, input_layer, input_dim, output_dim, act, use_bias, trainable, scope):
		with tf.variable_scope(scope):
			w = self.weight_variable(input_dim, output_dim, trainable)
			tf.summary.histogram('weights', w)
			mul = tf.matmul(input_layer, w)
			if use_bias:
				b = self.bias_variable(output_dim, trainable)
				tf.summary.histogram('bias', b)
				with tf.name_scope('pre_active'):
					preactive = tf.add(mul, b)
			else:
				with tf.name_scope('pre_active'):
					preactive = mul
			with tf.name_scope('activation'):
				if (act != None):
					activation = act(preactive)
				else:
					activation = preactive
		return activation

	def weight_variable(self, input_dim, output_dim, trainable):
		init = tf.truncated_normal(shape=[input_dim, output_dim], stddev=0.1)
		return tf.Variable(initial_value=init, trainable=trainable, name='weights')

	def bias_variable(self, output_dim, trainable):
		init = tf.constant(0.1, shape=[1, output_dim])
		return tf.Variable(initial_value=init, trainable=trainable, name='bias')
	"""
	def _create_res(self, input_layer, num):
		res = tf.layers.dense(
			input_layer,
			num,
			activation=None,
			kernel_initializer=tf.constant_initializer(1.0),
			bias_initializer=None,
			#kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=kernel_scale),
			#activity_regularizer=tf.contrib.layers.l2_regularizer(scale=activity_scale),
			trainable=False
			)
		return res
	"""
	def _dropout_layer(self, input_layer, ratio):
		out = tf.layers.dropout(input_layer, rate=ratio)
		return dropout_layer

	def _build_actor(self, a_dim, s_dim, a_bound, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1 = self._dense_layer(observation, s_dim, L1NUM, tf.nn.tanh, True, trainable, 'l1')
			l2 = self._dense_layer(l1, L1NUM, L2NUM, tf.nn.tanh, True, trainable, 'l2')
			raw_a = self._dense_layer(l2, L2NUM, a_dim, tf.nn.tanh, True, trainable, 'raw_a')
			scaled_a = tf.multiply(raw_a, a_bound, name='scaled_a')
		return scaled_a


	def _build_critic(self, a_dim, s_dim, action, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1_a = self._dense_layer(action, a_dim, L1NUM, tf.nn.tanh, True, trainable, 'l1_a')
			l1_s = self._dense_layer(observation, s_dim, L1NUM, tf.nn.tanh, True, trainable, 'l1_s')
			l1 = tf.concat([l1_a, l1_s], axis = 1, name='l1')
			l2 = self._dense_layer(l1, L1NUM + L1NUM, L2NUM, tf.nn.tanh, True, trainable, 'l2')
			q = self._dense_layer(l2, L2NUM, 1, None, False, trainable, 'q')
		return q

	def _q_loss(self, r, q, next_q, terminate):
		with tf.name_scope('target_q'):
			target_q = tf.add(tf.multiply(tf.multiply(next_q, GAMMA), terminate), r)
		with tf.name_scope('q_loss'):
			TD_error = tf.reduce_mean(tf.square(target_q - q))
		return TD_error

	def choose_action(self, s):
		s = s[np.newaxis, :]
		a = self.sess.run(self.sample_action, feed_dict={self.s1: s})
		return a

	def train(self, bs, ba, br, bs_, be):
		self.sess.run(self.update_old_q)
		self.sess.run(self.update_old_a)
		self.sess.run(self.c_train,
			feed_dict={self.s1: bs, self.actor1: ba, self.r: br, self.s2: bs_, self.terminate: be})
		self.sess.run(self.a_train, feed_dict={self.s1: bs})
	