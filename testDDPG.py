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
	def __init__(self, a_dim, s_dim, a_bound, sess, max_grad_norm=0.5):
		self.s1 = tf.placeholder(tf.float32, [None, s_dim], name='obs1')
		self.s2 = tf.placeholder(tf.float32, [None, s_dim], name='obs2')
		self.r = tf.placeholder(tf.float32, [None, 1], name='reward')
		self.terminate = tf.placeholder(tf.float32, [None, 1], name='done_flag')
		self.sess = sess


		self.actor1 = self._build_actor1(a_dim, s_dim, a_bound, self.s1, True, 'Actor1')
		self.actor2 = self._build_actor1(a_dim, s_dim, a_bound, self.s2, False, 'Actor2')
	
		self.critic1 = self._build_critic1(a_dim, s_dim, self.actor1, self.s1, True, 'Critic1')
		self.critic2 = self._build_critic1(a_dim, s_dim, self.actor2, self.s2, False, 'Critic2')

		self.a_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor1')
		self.a_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Actor2')

		self.c_params1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic1')
		self.c_params2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'Critic2')

		self.loss_q = self._q_loss1(self.r, self.critic1, self.critic2, self.terminate)
		self.loss_a = -tf.reduce_mean(self.critic1, name='a_loss')
		

		a_grads = tf.gradients(self.loss_a, self.a_params1)
		c_grads = tf.gradients(self.loss_q, self.c_params1)

		if max_grad_norm is not None:
			a_grads, _a_grad_norm = tf.clip_by_global_norm(a_grads, max_grad_norm)
			c_grads, _c_grad_norm = tf.clip_by_global_norm(c_grads, max_grad_norm)
		a_grads = list(zip(a_grads, self.a_params1))
		c_grads = list(zip(c_grads, self.c_params1))
		a_optimizer = tf.train.AdamOptimizer(learning_rate=LR_A, epsilon=1e-5)
		c_optimizer = tf.train.AdamOptimizer(learning_rate=LR_C, epsilon=1e-5)

		with tf.name_scope('C_train'):
			self.c_train = c_optimizer.apply_gradients(c_grads)
		with tf.name_scope('A_train'):
			self.a_train = a_optimizer.apply_gradients(a_grads)

		with tf.name_scope('sample_action'):
			self.sample_action = tf.clip_by_value(tf.squeeze(self.actor1, axis=0), -a_bound, a_bound)

		with tf.name_scope('Update_A'):
			self.update_old_a = [old.assign(old*(1-TAU_A) + new*TAU_A) for old, new in zip(self.a_params2, self.a_params1)]
		with tf.name_scope('Update_C'):
			self.update_old_q = [old.assign(old*(1-TAU_C) + new*TAU_C) for old, new in zip(self.c_params2, self.c_params1)]




###################################### style 1 ###########################################################
	def variable_summaries(self, var, scope):
		with tf.name_scope(scope):
			mean = tf.reduce_mean(var)
			tf.summary.scalar('mean', mean)
			#with tf.name_scope('stddev'):
			#	stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
			#tf.summary.scalar('stddev', stddev)
			#tf.summary.scalar('max', tf.reduce_max(var))
			#tf.summary.scalar('min', tf.reduce_min(var))
			tf.summary.histogram('histogram', var)

	def _dense_layer(self, input_layer, input_dim, output_dim, act, use_bias, trainable, scope):
		with tf.variable_scope(scope):
			w = self.weight_variable(input_dim, output_dim, trainable)
			self.variable_summaries(w, 'weights')
			mul = tf.matmul(input_layer, w)
			if use_bias:
				b = self.bias_variable(output_dim, trainable)
				self.variable_summaries(b, 'bias')
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
		return tf.get_variable(
			name='weights',
			shape=[input_dim, output_dim],
			initializer=tf.truncated_normal_initializer(stddev=0.1),
			trainable=trainable)

	def bias_variable(self, output_dim, trainable):
		return tf.get_variable(
			name='bias',
			shape=[1,output_dim],
			initializer=tf.constant_initializer(0.1),
			trainable=trainable)

	def _dropout_layer(self, input_layer, ratio):
		out = tf.layers.dropout(input_layer, rate=ratio)
		return dropout_layer

	def _build_actor1(self, a_dim, s_dim, a_bound, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1 = self._dense_layer(observation, s_dim, L1NUM, tf.nn.tanh, True, trainable, 'l1')
			l2 = self._dense_layer(l1, L1NUM, L2NUM, tf.nn.tanh, True, trainable, 'l2')
			raw_a = self._dense_layer(l2, L2NUM, a_dim, None, True, trainable, 'raw_a')
			scale = tf.get_variable(
				name='action_scale',
				shape=[1, a_dim],
				initializer=tf.constant_initializer(0.0),
				trainable=trainable)
			scaled_a = raw_a + scale * tf.random_normal(tf.shape(scale))
		return scaled_a

	def _build_critic1(self, a_dim, s_dim, action, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1_a = self._dense_layer(action, a_dim, L1NUM, tf.nn.tanh, True, trainable, 'l1_a')
			l1_s = self._dense_layer(observation, s_dim, L1NUM, tf.nn.tanh, True, trainable, 'l1_s')
			l1 = tf.concat([l1_a, l1_s], axis = 1, name='l1')
			l2 = self._dense_layer(l1, L1NUM + L1NUM, L2NUM, tf.nn.tanh, True, trainable, 'l2')
			q = self._dense_layer(l2, L2NUM, 1, None, False, trainable, 'q')
			self.variable_summaries(q, 'predict_q')
		return q

############################################################################################################


#################################### style 2 ###############################################################

	def _create_dense(self, input_layer, num, acti_func, trainable, name):
		dense_layer = tf.layers.dense(
			input_layer,
			num,
			activation=acti_func,
			kernel_initializer=tf.truncated_normal_initializer(mean=0.0, stddev=0.1),
			bias_initializer=tf.constant_initializer(0.1),
			trainable=trainable,
			name=name
			)
		return dense_layer

	def _build_actor2(self, a_dim, s_dim, a_bound, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1 = self._create_dense(observation, L1NUM, tf.nn.tanh, trainable, 'l1')
			l2 = self._create_dense(l1, L2NUM, tf.nn.tanh, trainable, 'l2')
			raw_a = self._create_dense(l2, a_dim, tf.nn.tanh, trainable, 'raw_a')
			scaled_a = tf.multiply(raw_a, a_bound, name='scaled_a')
		return scaled_a

	def _build_critic2(self, a_dim, s_dim, action, observation, trainable, scope):
		with tf.variable_scope(scope):
			l1_a = self._create_dense(action, L1NUM, tf.nn.tanh, trainable, 'l1_a')
			l1_s = self._create_dense(observation, L1NUM, tf.nn.tanh, trainable, 'l1_s')
			l1 = tf.concat([l1_a, l1_s], axis = 1, name='l1')
			l2 = self._create_dense(l1, L2NUM, tf.nn.tanh, trainable, 'l2')
			q = self._create_dense(l2, 1, None, trainable, 'q')
		return q
############################################################################################################


	def _q_loss1(self, r, q, next_q, terminate):
		self.variable_summaries(r, 'reward')
		self.variable_summaries(terminate, 'terminate')
		with tf.name_scope('target_q'):
			target_q = tf.add(tf.multiply(tf.multiply(next_q, GAMMA), terminate), r)
			self.variable_summaries(target_q, 'target_q')
		with tf.name_scope('q_loss'):
			TD_error = tf.reduce_mean(tf.square(target_q - q))
			self.variable_summaries(TD_error, 'TD_error')
		return TD_error

	def _q_loss2(self, r, q, next_q, terminate):
		with tf.name_scope('target_q'):
			target_q = tf.add(tf.multiply(next_q, GAMMA), r)
			self.variable_summaries(target_q, 'target_q')
		with tf.name_scope('q_loss'):
			TD_error = tf.reduce_mean(tf.square(target_q - q))
			self.variable_summaries(TD_error, 'TD_error')
		return TD_error


	def choose_action(self, s):
		s = s[np.newaxis, :]
		a = self.sess.run(self.sample_action, feed_dict={self.s1: s})
		return a

	def train1(self, bs, ba, br, bs_, be):
		self.sess.run(self.update_old_q)
		self.sess.run(self.update_old_a)
		self.sess.run(self.c_train,
			feed_dict={self.s1: bs, self.actor1: ba, self.r: br, self.s2: bs_, self.terminate: be})
		self.sess.run(self.a_train, feed_dict={self.s1: bs})
	
	def train2(self, bs, ba, br, bs_, be):
		self.sess.run(self.update_old_q)
		self.sess.run(self.update_old_a)
		self.sess.run(self.c_train,
			feed_dict={self.s1: bs, self.actor1: ba, self.r: br, self.s2: bs_})
		self.sess.run(self.a_train, feed_dict={self.s1: bs})