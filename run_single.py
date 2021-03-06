import gym
import ma_gym 
import random
import datetime
import numpy as np
import tensorflow as tf


class QMix():
    def __init__(self, env, num_s, num_a, lr=0.0001, gamma=0.99, replace_target_iter=2000,
                 memory_size=200000, batch_size=1024, epsilon=1, epsilon_decay=0.0001):
        self.n_agents = 2
        self.env = env
        self.name = "qmix"
        self.num_global_s = 2*num_s
        self.num_s = num_s
        self.num_a = num_a
        self.lr = lr
        self.gamma = gamma
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.1
        self.learn_step_cnt = 0  # total learning step
        self.episode_cnt = 0
        self.memory = []
        self.memory_counter = 0
        self._build_net()

        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/eval_net')

        e_params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/mixing_net' + '/eval_hyper')
        t_params += tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name + '/mixing_net' + '/target_hyper')

        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()
        self.sess.run(tf.global_variables_initializer())

        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time
        self.summary_writer = tf.summary.FileWriter(train_log_dir, self.sess.graph)

    def _build_net(self):  # we use parameter sharing among agents
        with tf.variable_scope(self.name):
            # ------------------ all inputs ------------------------
            self.S = tf.placeholder(tf.float32, [None, self.num_global_s], name='S')  # input Global State
            self.s = tf.placeholder(tf.float32, [None, self.num_s], name='s1')  # input state for agent1
            self.S_ = tf.placeholder(tf.float32, [None, self.num_global_s], name='S_')  # input Next Global State
            self.s_ = tf.placeholder(tf.float32, [None, self.num_s], name='s1_')  # input next state for agent1
            self.R = tf.placeholder(tf.float32, [None, ], name='R')  # input Reward
            self.a = tf.placeholder(tf.float32, [None, self.num_a], name='a')  # input Action onehot for agent1
            self.done = tf.placeholder(tf.float32, [None, ], name='done')  # input Done info ???

            self.q_m_ = tf.placeholder(tf.float32, [None, ], name='q_value_next_max')
            self.q_target = tf.placeholder(tf.float32, [None, ], name='q_tot_target')

            w_initializer, b_initializer = tf.random_normal_initializer(0., 0.1), tf.constant_initializer(0.0)

            # ------------------ build evaluate_net ------------------
            with tf.variable_scope('eval_net'):
                a_fc1 = tf.layers.dense(self.s, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='agent_fc1_e')
                # a_fc2 = tf.layers.dense(a_fc1, 128, tf.nn.relu, kernel_initializer=w_initializer,
                #                         bias_initializer=b_initializer, name='agent_fc2_e')
                # a_fc3 = tf.layers.dense(a_fc2, 64, tf.nn.relu, kernel_initializer=w_initializer,
                #                         bias_initializer=b_initializer, name='agent_fc3_e')
                self.q_eval = tf.layers.dense(a_fc1, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_e')

            # ------------------ build target_net ------------------
            with tf.variable_scope('target_net'):
                a_fc1_ = tf.layers.dense(self.s_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                                         bias_initializer=b_initializer, name='agent_fc1_t')
                # a_fc2_ = tf.layers.dense(a_fc1_, 128, tf.nn.relu, kernel_initializer=w_initializer,
                #                          bias_initializer=b_initializer, name='agent_fc2_t')
                # a_fc3_ = tf.layers.dense(a_fc2_, 64, tf.nn.relu, kernel_initializer=w_initializer,
                #                          bias_initializer=b_initializer, name='agent_fc3_t')
                self.q_next = tf.layers.dense(a_fc1_, self.num_a, kernel_initializer=w_initializer,
                                              bias_initializer=b_initializer, name='q_t')

            # [batch*n_agents, 1]
            self.q_selected = tf.reduce_sum(tf.multiply(self.q_eval, self.a), axis=1)

            # ------------------ build mixing_net ------------------
            with tf.variable_scope('mixing_net'):
                # [batch, n_agents]
                self.q_concat = tf.reshape(self.q_selected, [-1, self.n_agents])
                self.q_concat_ =tf.reshape(self.q_m_, [-1, self.n_agents]) 

                with tf.variable_scope('eval_hyper'):
                    with tf.variable_scope('hyper_layer1'):
                        non_abs_w1 = tf.layers.dense(inputs=self.S, units=self.n_agents*32, kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer, name='non_abs_w1')
                        self.w1 = tf.reshape(tf.abs(non_abs_w1), shape=[-1, self.n_agents, 32], name='w1')
                        self.b1 = tf.layers.dense(inputs=self.S, units=32, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='non_abs_b1')
                    with tf.variable_scope('hyper_layer2'):
                        non_abs_w2 = tf.layers.dense(inputs=self.S, units=32 * 1, kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer, name='non_abs_w2')
                        self.w2 = tf.reshape(tf.abs(non_abs_w2), shape=[-1, 32, 1], name='w2')
                        bef_b2 = tf.layers.dense(inputs=self.S, units=32, activation=tf.nn.relu,
                                                kernel_initializer=w_initializer, bias_initializer=b_initializer, name='bef_b2')
                        self.b2 = tf.layers.dense(inputs=bef_b2, units=1, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='non_abs_b2')

                with tf.variable_scope('target_hyper'):
                    with tf.variable_scope('hyper_layer1'):
                        non_abs_w1_ = tf.layers.dense(inputs=self.S_, units=self.n_agents*32, kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer, name='non_abs_w1')
                        self.w1_ = tf.reshape(tf.abs(non_abs_w1_), shape=[-1, self.n_agents, 32], name='w1')
                        self.b1_ = tf.layers.dense(inputs=self.S_, units=32, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='non_abs_b1')

                    with tf.variable_scope('hyper_layer2'):
                        non_abs_w2_ = tf.layers.dense(inputs=self.S_, units=32 * 1, kernel_initializer=w_initializer,
                                                    bias_initializer=b_initializer, name='non_abs_w2')
                        self.w2_ = tf.reshape(tf.abs(non_abs_w2_), shape=[-1, 32, 1], name='w2')
                        bef_b2_ = tf.layers.dense(inputs=self.S_, units=32, activation=tf.nn.relu,
                                                kernel_initializer=w_initializer, bias_initializer=b_initializer, name='bef_b2')
                        self.b2_ = tf.layers.dense(inputs=bef_b2_, units=1, kernel_initializer=w_initializer,
                                                bias_initializer=b_initializer, name='non_abs_b2')


                with tf.variable_scope('layer_mix_eval'):
                    lin1 = tf.matmul(tf.reshape(self.q_concat, shape=[-1, 1, self.n_agents]), self.w1) + tf.reshape(self.b1, shape=[-1, 1, 32])
                    a1 = tf.nn.elu(lin1, name='a1')
                    self.Q_tot = tf.reshape(tf.matmul(a1, self.w2), shape=[-1, 1]) + self.b2

                with tf.variable_scope('layer_mix_target'):
                    lin1_ = tf.matmul(tf.reshape(self.q_concat_, shape=[-1, 1, self.n_agents]), self.w1_) + tf.reshape(self.b1_, shape=[-1, 1, 32])
                    a1_ = tf.nn.elu(lin1_, name='a1_')
                    self.Q_tot_ = tf.reshape(tf.matmul(a1_, self.w2_), shape=[-1, 1]) + self.b2_

            # todo: add q_target, loss, train_op
            # with tf.variable_scope('q_target'):
                
            with tf.variable_scope('loss'):
                # self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.Q_tot, name='TD_error'))
                self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_selected, name='TD_error'))
            with tf.variable_scope('train'):
                self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def act(self, state):
        if np.random.uniform() > self.epsilon :# pick the argmax action
            s = np.array(state)
            if len(s.shape) < 2:
                s = np.array(state)[np.newaxis, :]
            q_eval = self.sess.run(self.q_eval, feed_dict={self.s: s})
            action = np.argmax(q_eval, axis=-1)[0]
        else:  # pick random action
            action = self.env.action_space.sample() 
        return action

    def store(self, EXP):
        self.memory_counter += 1
        if len(self.memory) > self.memory_size:
            # random replacement
            index = np.random.randint(0, self.memory_size)
            self.memory[index] = EXP
        else:
            self.memory.append(EXP)

    def learn(self):
        if len(self.memory) < self.batch_size :
            return
        # sample batch exp from memory
        if self.learn_step_cnt % 10000 == 0:
            print(self.name, 'update ----> learn_step_cnt', self.learn_step_cnt)
        batch_exp = random.sample(self.memory, self.batch_size)
        S, s, a, R, S_, s_, done = [[] for _ in range(7)]
        for exp in batch_exp:
            S.append(exp[0])
            s.append([exp[1] , exp[2]])
            a.append([exp[3] , exp[4]])
            R.append(exp[5])
            S_.append(exp[6])
            s_.append([exp[7], exp[8]])
            done.append(exp[9])

        # to get q_tot
        s = np.stack(s)
        a = np.stack(a)
        s_ = np.stack(s_)

        s.shape = (self.batch_size*self.n_agents, self.num_s)
        s_.shape = (self.batch_size*self.n_agents, self.num_s)
        
        actions_1hot = np.zeros([self.batch_size, self.n_agents, self.num_a], dtype=int)
        grid = np.indices((self.batch_size, self.n_agents))
        actions_1hot[grid[0], grid[1], a] = 1
        actions_1hot.shape = (self.batch_size*self.n_agents, self.num_a)

        # to get q_tot_
        q_ = self.sess.run(self.q_next, feed_dict={self.s_: s_})
        q_m_ = np.max(q_, axis=1)
        q_tot_ = self.sess.run(self.Q_tot_, feed_dict={self.S_: S_, self.q_m_: q_m_})
        R = np.repeat(np.expand_dims(np.array(R), axis=-1), 2, axis=1).reshape(2048)
        done = np.repeat(np.expand_dims(np.array(done), axis=-1), 2, axis=1).reshape(2048)
        q_target = R + (1 - np.array(done)) * self.gamma * q_m_ # np.squeeze(q_tot_, axis=-1)
        # update
        _, cost = self.sess.run([self._train_op, self.loss],
                                feed_dict={self.S: S, self.s:s, self.a: actions_1hot,
                                           self.q_target: q_target, self.done: done})
        # print('cost', cost)

        self.write_summary_scalar('loss', cost, self.learn_step_cnt)
        self.write_summary_scalar('epsilon', self.epsilon, self.learn_step_cnt)
        self.write_summary_scalar('memory_cnt', self.memory_counter, self.learn_step_cnt)
        self.epsilon = max(self.epsilon - self.epsilon_decay, self.epsilon_min)  # decay epsilon
        self.learn_step_cnt += 1

        # check to do the soft replacement of target net
        if self.learn_step_cnt % self.replace_target_iter == 0 and self.learn_step_cnt:
            self.sess.run(self.target_replace_op)


    def train(self):
        
        for i in range(50000):        
            # done_n = [False for _ in range(env.n_agents)]
            done_n = False
            ep_reward = 0
            obs = env.reset()
            while not done_n:
                # env.render()
                action = self.act(obs)
                obs_n, reward_n, done_n, info = env.step(action)
                # import pdb; pdb.set_trace()
                ep_reward += reward_n
                obs_glob = [obs.tolist() + obs.tolist()]
                obs_glob_next = [obs_n.tolist() + obs_n.tolist()]
                self.store(obs_glob + [obs.tolist()] + [obs.tolist()] + [action, action] + [reward_n] + obs_glob_next + [obs_n.tolist()] + [obs_n.tolist()] + [done_n])
                obs = obs_n
                for i in range(4):
                    self.learn()
            self.write_summary_scalar("ep_reward", ep_reward, self.learn_step_cnt)

    def write_summary_scalar(self, tag, value, iteration):
        self.summary_writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag=tag, simple_value=value)]), iteration)


env = gym.make('CartPole-v0')
# alg = QMix(env, env.observation_space[0].shape[0], env.action_space[0].n)
alg = QMix(env, env.observation_space.shape[0], env.action_space.n)

alg.train()
