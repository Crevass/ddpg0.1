import tensorflow as tf
import numpy as np

L1NUM = 64
L2NUM = 64
L3NUM = 64
L4NUM = 64

ACTIVITY_SCALE = 0
L2_SCALE = 0
GAMMA = 0.95

LR_C = 0.0002
LR_A = 0.0001

TAU = 0.05

class DDPG(object):
	def __init__(self, a_dim, s_dim, a_bound, sess):
		self.s1 = tf.placeholder(tf.float32, [None, s_dim])
		self.s2 = tf.placeholder(tf.float32, [None, s_dim])
		self.r = tf.placeholder(tf.float32, [None, 1])
		self.sess = sess

		self.actor1 = self._build_actor(a_dim, self.s1, a_bound, True, 'a1')
		self.actor2 = self._build_actor(a_dim, self.s2, a_bound, False, 'a2')

		self.critic1 = self._build_critic(self.actor1, self.s1, True, 'Critic1')
		self.critic2 = self._build_critic(self.actor2, self.s2, False, 'Critic2')

		self.a_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'a1')
		self.a_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'a2')

		self.c_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic1')
		self.c_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic2')

		self.loss_q = self._q_loss(self.r, self.critic1, self.critic2)
		self.loss_a = -tf.reduce_mean(self.critic1)
		
		self.c_train = tf.train.AdamOptimizer(LR_C).minimize(self.loss_q, var_list = self.c_params1)
		self.a_train = tf.train.AdamOptimizer(LR_A).minimize(self.loss_a, var_list = self.a_params1)

		self.sample_action = tf.clip_by_value(tf.squeeze(self.actor1), -a_bound, a_bound)
		self.update_old_a = [old.assign(old*(1-TAU) + new*TAU) for old, new in zip(self.a_params2, self.a_params1)]
		self.update_old_q = [old.assign(old*(1-TAU) + new*TAU) for old, new in zip(self.c_params2, self.c_params1)]
		self.sess.run(tf.global_variables_initializer())
		self.sess.run(self.update_old_a)
		self.sess.run(self.update_old_q)

	def _create_dense(self, input_layer, num, acti_func, kernel_scale=0, activity_scale=0, trainable=True):
		dense_layer = tf.layers.dense(
			input_layer,
			num,
			activation=acti_func,
			kernel_initializer=tf.random_normal_initializer(0.05, 0.3),
			bias_initializer=tf.random_normal_initializer(0.05, 0.3),
			#kernel_regularizer=tf.contrib.layers.l2_regularizer(scale=kernel_scale),
			#activity_regularizer=tf.contrib.layers.l2_regularizer(scale=activity_scale),
			trainable=trainable
			)
		return dense_layer

	def _create_drop(self, input_layer, ratio):
		dropout_layer = tf.layers.dropout(input_layer, rate=ratio)
		return dropout_layer
	
	def _build_actor(self, a_dim, observation, a_bound, trainable, scope):
		with tf.variable_scope(scope):
			l1 = self._create_dense(observation, L1NUM, tf.nn.tanh, L2_SCALE, ACTIVITY_SCALE, trainable)
			#l1_drop = self._create_drop(l1, 0.5)
			l2 = self._create_dense(l1, L2NUM, tf.nn.tanh, L2_SCALE, ACTIVITY_SCALE, trainable)
			#l2_drop = self._create_drop(l2, 0.5)
			a = self._create_dense(l2, a_dim, tf.nn.tanh, L2_SCALE, ACTIVITY_SCALE, trainable)
			scaled_a = tf.multiply(a, a_bound)
		return scaled_a

	def _build_critic(self, action, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1_a = self._create_dense(action, L1NUM, tf.nn.tanh, L2_SCALE, ACTIVITY_SCALE, trainable)
			l1_s = self._create_dense(observation, L1NUM, tf.nn.tanh, L2_SCALE, ACTIVITY_SCALE, trainable)
			l1 = tf.concat([l1_a, l1_s], axis = 1)
			l2 = self._create_dense(l1, L2NUM, tf.nn.tanh, L2_SCALE, ACTIVITY_SCALE, trainable)
			q = self._create_dense(l2, 1, None, L2_SCALE, ACTIVITY_SCALE, trainable)
		return q

	def _q_loss(self, r, q, next_q):
		target_q = GAMMA * next_q + r
		TD_error = tf.reduce_mean(tf.square(target_q - q))
		return TD_error

	def choose_action(self, s):
		s = s[np.newaxis, :]
		a = self.sess.run(self.sample_action, feed_dict={self.s1: s})
		return a

	def train(self, bs, ba, br, bs_):
		self.sess.run(self.update_old_q)
		self.sess.run(self.update_old_a)
		self.sess.run(self.c_train, feed_dict={self.s1: bs, self.actor1: ba, self.r: br, self.s2: bs_})
		self.sess.run(self.a_train, feed_dict={self.s1: bs})
	