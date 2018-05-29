import numpy as np
from Core_large_res import DDPG
from Memory import Memory
import tensorflow as tf
import random
import gym
import time

SAVE_PATH = '/home/wang/Research/MODELS/DDPG-MODEL/'
#SUB_PATH = 'Pendulum-v0'
SUB_PATH = ENV_NAME = 'LunarLanderContinuous-v2'
#SUB_PATH = ENV_NAME = 'Pendulum-v0'

BATCH_SIZE = 32
MEMO_CAPACITY = 100000

MAX_EPISODE = 1000000
TERMINATE_REWARD = -200

HORIZON = 1000

VIDEO_SAVE_TURN = 100

def save_or_not(episode):
	if episode % VIDEO_SAVE_TURN == 0:
		return True
	else:
		return False

def get_true(episode):
	return True


if __name__ == "__main__":
	env = gym.make(ENV_NAME)
	A_DIM = env.action_space.shape[0]
	S_DIM = env.observation_space.shape[0]
	A_BOUND = env.action_space.high

	gpu_option = tf.GPUOptions(per_process_gpu_memory_fraction = 0.4, allow_growth = True)
	sess = tf.Session(config = tf.ConfigProto(gpu_options = gpu_option, log_device_placement = False))

	env = gym.wrappers.Monitor(
		env = env, directory = SAVE_PATH + 'videos/' + SUB_PATH, video_callable = save_or_not, force = True)
	
	agent = DDPG(A_DIM, S_DIM, A_BOUND, sess)
	
	memory = Memory(MEMO_CAPACITY)

	merged = tf.summary.merge_all()
	writer = tf.summary.FileWriter(SAVE_PATH + 'log', sess.graph)
	sess.run(tf.global_variables_initializer())

	recent100 = np.zeros(100)
	write_pointer = 0
	for episode in range(MAX_EPISODE):
		s = env.reset()
		ep_reward = 0
		step_pointer = 0
		end = 1
		while True:
			write_pointer += 1
			a = agent.choose_action(s)

			a = np.clip(np.random.normal(a, 0.3), -A_BOUND, A_BOUND)

			s_, r, done, _ = env.step(a)
			if done:
				end = 0
			else:
				end = 1

			scaled_r = r/10

			memory.save(s, a, scaled_r, s_, end)
			if (write_pointer % 10 == 0):
				summary = sess.run(merged)
				writer.add_summary(summary, write_pointer)

			if (memory.get_size() > HORIZON):
				bs, ba, br, bs_, be = memory.sample(BATCH_SIZE)
				agent.train(bs, ba, br, bs_, be)
			s = s_
			ep_reward += r
			step_pointer += 1
			if done:
				break

		recent100[episode % 100] = ep_reward
		average_reward = recent100.mean()
		
		print('EP: %i ' %episode, 'R: %.2f ' %ep_reward, 'Avg: %.2f ' %average_reward, 'STEP: %i ' %step_pointer)

		if ((average_reward >= TERMINATE_REWARD) and (episode > 100)):
			print('Reward Achived, finish trainning')
			break
	env.close()
	print('Start Testing........\n')

	env = gym.make(ENV_NAME)
	env = gym.wrappers.Monitor(env = env, directory = SAVE_PATH + 'videos/' + SUB_PATH + '/last_100', video_callable = get_true, force = True)

	for test_ep in range(100):
		s = env.reset()
		ep_reward = 0
		step_pointer = 0
		while True:
			a = agent.choose_action(s)
			s, r, done, _ = env.step(a)
			ep_reward += r
			step_pointer += 1
			if done:
				break
		print('TEST EP: %i ' %test_ep, 'R: %.2f ' %ep_reward, 'STEP: %i' %step_pointer)
	env.close()
