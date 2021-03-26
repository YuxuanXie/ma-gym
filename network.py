import tensorflow as tf


def get_variable(name, shape):

    return tf.get_variable(name, shape, tf.float32,
                           tf.initializers.truncated_normal(0,0.01))

def Qmix_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer):
    """
    Args:
        agent_qs: shape [batch, n_agents]
        state: shape [batch, state_dim]
        state_dim: integer
        n_agents: integer
        n_h_mixer: integer
    """
    agent_qs_reshaped = tf.reshape(agent_qs, [-1, 1, n_agents])

    # n_h_mixer * n_agents because result will be reshaped into matrix
    hyper_w_1 = get_variable('hyper_w_1', [state_dim, n_h_mixer*n_agents]) 
    hyper_w_final = get_variable('hyper_w_final', [state_dim, n_h_mixer])

    hyper_b_1 = tf.get_variable('hyper_b_1', [state_dim, n_h_mixer])

    hyper_b_final_l1 = tf.layers.dense(inputs=state, units=n_h_mixer, activation=tf.nn.relu,
                                       use_bias=False, name='hyper_b_final_l1')
    hyper_b_final = tf.layers.dense(inputs=hyper_b_final_l1, units=1, activation=None,
                                    use_bias=False, name='hyper_b_final')

    # First layer
    w1 = tf.abs(tf.matmul(state, hyper_w_1))
    b1 = tf.matmul(state, hyper_b_1)
    w1_reshaped = tf.reshape(w1, [-1, n_agents, n_h_mixer]) # reshape into batch of matrices
    b1_reshaped = tf.reshape(b1, [-1, 1, n_h_mixer])
    # [batch, 1, n_h_mixer]
    hidden = tf.nn.elu(tf.matmul(agent_qs_reshaped, w1_reshaped) + b1_reshaped)
    
    # Second layer
    w_final = tf.abs(tf.matmul(state, hyper_w_final))
    w_final_reshaped = tf.reshape(w_final, [-1, n_h_mixer, 1]) # reshape into batch of matrices
    b_final_reshaped = tf.reshape(hyper_b_final, [-1, 1, 1])

    # [batch, 1, 1]
    y = tf.matmul(hidden, w_final_reshaped) + b_final_reshaped

    q_tot = tf.reshape(y, [-1, 1])

    return q_tot


def si_weight(states, actions, n_agents):

    data = tf.stack([states, actions], axis=-1)
    with tf.variable_scope('adv_hyer'):
        all_head_key = tf.layers.dense(inputs=state, 1, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='all_head_key')
        all_head_agents = tf.layers.dense(inputs=state, n_agents, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='all_head_agents')
        all_head_action = tf.layers.dense(inputs=data, n_agents, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='all_head_action')

    head_attend_weights = []
    for curr_head_key, curr_head_agents, curr_head_action in zip(all_head_key, all_head_agents, all_head_action):
        x_key = tf.repeat(tf.abs(curr_head_key), n_agents) + 1e-10
        x_agents = tf.sigmoid(curr_head_agents)
        x_action = tf.sigmoid(curr_head_action)
        weights = x_key * x_agents * x_action
        head_attend_weights.append(weights)

    head_attend = tf.stack(head_attend_weights, axis=1)
    head_attend = tf.reshape(head_attend, shape=(-1, self.num_kernel, self.n_agents))
    head_attend = tf.reduce_sum(head_attend, axis=1)

    return head_attend

def calc_v(agent_qs, n_agents):
    agent_qs = tf.reshape(agent_qs shape=[-1, n_agents])
    v_tot = tf.reduce_sum(agent_qs, axis=-1)
    return v_tot    

def calc_adv(agent_qs, states, actions, max_q_i, n_agents):
    # states = states.reshape(-1, state_dim)
    # actions = actions.reshape(-1, action_dim)
    # agent_qs = agent_qs.view(-1, n_agents)
    # max_q_i = max_q_i.view(-1, n_agents)

    adv_q = agent_qs - max_q_i

    adv_w_final = si_weight(states, actions, n_agents)
    adv_w_final = tf.reshape(adv_w_final, shape=[-1, n_agents])

    adv_tot = tf.reduce_sum(adv_q * (adv_w_final - 1.), axis=1)



def calc(agent_qs, state, state_dim, n_agents, n_h_mixer=32,  max_q_i=None, actions=None, is_v=False):
    if is_v:
        v_tot = calc_v(agent_qs)
        return v_tot
    else:
        adv_tot = calc_adv(agent_qs, states, actions, max_q_i)
        return adv_tot

def Qplex_mixer(agent_qs, state, state_dim, n_agents, n_h_mixer=32,  max_q_i=None, actions=None, is_v=False):
    """
    Args:
        agent_qs: shape [batch, n_agents]
        state: shape [batch, state_dim]
        state_dim: integer
        n_agents: integer
        n_h_mixer: integer
        max_q_i: shape [batch, n_agents]
    """
    bs = agent_qs.shape[0]
    
    with tf.variable_scope('hyper_w_final'):
        non_abs_w_final1 = tf.layers.dense(inputs=state, units=n_agents*32, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='hyper_w_final_1')
        non_abs_w_final2 = tf.layers.dense(inputs=non_abs_w_final1, units= n_agents, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='hyper_w_final_2')

        w_final = tf.reshape(tf.abs(non_abs_w_final2), shape=[-1, n_agents]) + 1e-10

    with tf.variable_scope('V'):
        non_abs_v1 = tf.layers.dense(inputs=state, units=n_agents*32, tf.nn.relu, kernel_initializer=w_initializer,
                                        bias_initializer=b_initializer, name='v_1')
        non_abs_v_final = tf.layers.dense(inputs=non_abs_v1, units= n_agents, kernel_initializer=w_initializer,
                                    bias_initializer=b_initializer, name='v_2')

        v = tf.reshape(non_abs_v_final, shape=[-1, n_agents])

    agent_qs = tf.matmul(w_final, agent_qs) + v 

    if not is_v:
        max_q_i = tf.reshape(max_q_i, shape=[-1, n_agents])
        max_q_i = tf.matmul(w_final, max_q_i) + v 
    
    y = self.calc(agent_qs, states, actions=actions, max_q_i=max_q_i, is_v=is_v)
    v_tot = y.view(bs, -1, 1)

    return v_tot