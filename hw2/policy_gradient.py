
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import gym
import logz
import scipy.signal
import os
import time
import inspect
from multiprocessing import Process


# In[137]:
#============================================================================================#
# Utilities
#============================================================================================#
def build_mlp(
        input_placeholder, 
        output_size,
        scope, 
        n_layers=2, 
        size=64, 
        activation=tf.tanh,
        output_activation=None
        ):
    #========================================================================================#
    #                           ----------SECTION 3----------
    # Network building
    #
    # Your code should make a feedforward neural network (also called a multilayer perceptron)
    # with 'n_layers' hidden layers of size 'size' units. 
    # 
    # The output layer should have size 'output_size' and activation 'output_activation'.
    #
    # Hint: use tf.layers.dense
    #========================================================================================#
    print('build_mlp[%s] with n_layers:%d, size:%d, output_size:%d' % (scope, n_layers, size, output_size))
    prev = input_placeholder
    hidden_layers = [size] * n_layers
    with tf.variable_scope(scope):
        # YOUR_CODE_HERE
        for i, n in enumerate(hidden_layers):
            out = tf.contrib.layers.fully_connected(inputs=prev, \
                                                    num_outputs=n, \
                                                    activation_fn=activation,\
                                                    normalizer_fn=None)
            prev = out

        out = tf.contrib.layers.fully_connected(inputs=out,\
                                                num_outputs=output_size,\
                                                activation_fn=output_activation)

    return out
    
def pathlength(path):
    return len(path["reward"])


# In[140]:
def prepare_training_data(trajectories, reward_to_go=False, gamma=1.0):
    paths = trajectories
    import time
    s = time.time()
    for p in paths:
        R = 0
        Rs = []
        rews = p['reward']
        totalr = 0
        for i in range(len(rews) - 1, -1, -1):
            r = rews[i]
            totalr += r
            R = r + gamma * R
            Rs.insert(0, R)
    
        p['totalr'] = totalr
        if reward_to_go is True:
            p['R'] = Rs
        else:
            p['R'] = [Rs[0]] * len(Rs)

    """
    for p in paths:
        rews = np.array(p['reward'])
        seq = np.power(gamma, np.arange(len(rews)))
        Rs = rews * seq
        Rs = np.cumsum(Rs)
        
        p['totalr'] = Rs[-1]
        if reward_to_go is True:
            p['R'] = Rs.tolist()
        else:
            p['R'] = [Rs[-1]] * len(Rs)
    """
    
    ob_no = np.concatenate([path["observation"] for path in paths])
    ac_na = np.concatenate([path["action"] for path in paths])    
    q_n = np.concatenate([path["R"] for path in paths])
    rewards = np.array([path["totalr"] for path in paths])
    return {'ob_no': ob_no, 'ac_na': ac_na, 'q_n': q_n, 'rewards': rewards}

# In[149]:
def train_PG(exp_name='',
            env_name='CartPole-v0',
            n_iter=100, 
            gamma=1.0, 
            min_timesteps_per_batch=1000, 
            max_path_length=None,
            learning_rate=5e-3, 
            reward_to_go=True, 
            animate=True, 
            logdir=None, 
            normalize_advantages=True,
            nn_baseline=False, 
            seed=0,
            # network arguments
            n_layers=1,
            size=32,
            log_interval=10):
    tf.reset_default_graph()
    
    start = time.time()

    # Configure output directory for logging
    logz.configure_output_dir(logdir)
    
    # Log experimental parameters
    args = inspect.getargspec(train_PG)[0]
    locals_ = locals()
    params = {k: locals_[k] if k in locals_ else None for k in args}
    logz.save_params(params)

    # Set random seeds
    tf.set_random_seed(seed)
    np.random.seed(seed)

    # Make the gym environment
    env = gym.make(env_name)
    
    # Is this env continuous, or discrete?
    discrete = isinstance(env.action_space, gym.spaces.Discrete)

    # Maximum length for episodes
    max_path_length = max_path_length or env.spec.max_episode_steps

    #========================================================================================#
    # Notes on notation:
    # 
    # Symbolic variables have the prefix sy_, to distinguish them from the numerical values
    # that are computed later in the function
    # 
    # Prefixes and suffixes:
    # ob - observation 
    # ac - action
    # _no - this tensor should have shape (batch size /n/, observation dim)
    # _na - this tensor should have shape (batch size /n/, action dim)
    # _n  - this tensor should have shape (batch size /n/)
    # 
    # Note: batch size /n/ is defined at runtime, and until then, the shape for that axis
    # is None
    #========================================================================================#

    # Observation and action sizes
    ob_dim = env.observation_space.shape[0]
    ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
    
    sy_ob_no = tf.placeholder(shape=[None, ob_dim], name="ob", dtype=tf.float32)
    if discrete:
        sy_ac_na = tf.placeholder(shape=[None], name="ac", dtype=tf.int32) 
    else:
        sy_ac_na = tf.placeholder(shape=[None, ac_dim], name="ac", dtype=tf.float32)
    sy_adv_n = tf.placeholder(shape=[None], name="advantage_score", dtype=tf.float32)
    
    #build policy net
    if discrete:
        sy_logits_na = build_mlp(sy_ob_no, 
                                ac_dim,
                                'policy_net', 
                                n_layers=n_layers, 
                                size=size, 
                                activation=tf.tanh,
                                output_activation=None)
        sy_sampled_ac = tf.squeeze(tf.multinomial(sy_logits_na, 1)) # Hint: Use the tf.multinomial op
        log_prob = tf.nn.log_softmax(sy_logits_na)
        sy_logprob_n = tf.reduce_sum(tf.one_hot(sy_ac_na, ac_dim, dtype=tf.float32) * log_prob, axis=1)
    else:
        sy_mean = build_mlp(sy_ob_no, 
                            ac_dim,
                            'policy_net_mean', 
                            n_layers=n_layers, 
                            size=size, 
                            activation=tf.tanh,
                            output_activation=None)
        # logstd should just be a trainable variable, not a network output.
        sy_logstd = tf.Variable(tf.ones(ac_dim, dtype=tf.float32), name='policy_net_sigma')
        dist = tf.contrib.distributions.MultivariateNormalDiag(sy_mean, sy_logstd)
        sy_sampled_ac = dist.sample()
        sy_logprob_n = dist.log_prob(sy_sampled_ac)  # Hint: Use the log probability under a multivariate gaussian. 
        
    #build optimizer
    loss = -tf.reduce_sum(sy_logprob_n * sy_adv_n, axis=0)
    update_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    if nn_baseline:
        baseline_prediction = tf.squeeze(build_mlp(
                                sy_ob_no, 
                                1, 
                                "nn_baseline",
                                n_layers=n_layers,
                                size=size))
        # Define placeholders for targets, a loss function and an update op for fitting a 
        # neural network baseline. These will be used to fit the neural network baseline. 
        # YOUR_CODE_HERE
        sy_base_n = tf.placeholder(shape=[None], name='baseline_target', dtype=tf.float32)
        baseline_loss = tf.reduce_mean(tf.square(baseline_prediction - sy_base_n))
        baseline_update_op = tf.train.AdamOptimizer(learning_rate).minimize(baseline_loss)
        
    tf_config = tf.ConfigProto(inter_op_parallelism_threads=1, intra_op_parallelism_threads=1) 
    sess = tf.Session(config=tf_config)
    sess.__enter__() # equivalent to `with sess:`

    #with tf.Session() as sess:
    if True:
        tf.global_variables_initializer().run() #pylint: disable=E1101
        
        #========================================================================================#
        # Training Loop
        #========================================================================================#
        total_timesteps = 0
        for itr in range(n_iter):
            #print("********** Iteration %i ************"%itr)

            # Collect paths until we have enough timesteps
            timesteps_this_batch = 0
            paths = []
            while True:
                ob = env.reset()
                obs, acs, rewards = [], [], []
                animate_this_episode=(len(paths)==0 and (itr % 10 == 0) and animate)
                steps = 0
                while True:
                    if animate_this_episode:
                        #env.render()
                        #time.sleep(0.05)
                        pass
                    obs.append(ob)
                    ac = sess.run(sy_sampled_ac, feed_dict={sy_ob_no : ob[None]})
                    acs.append(ac)
                    ob, rew, done, _ = env.step(ac)
                    rewards.append(rew)
                    steps += 1
                    if done or steps > max_path_length:
                        break
                path = {"observation" : np.array(obs), 
                        "reward" : np.array(rewards), 
                        "action" : np.vstack(acs)}
                paths.append(path)
                timesteps_this_batch += pathlength(path)
                if timesteps_this_batch > min_timesteps_per_batch:
                    break
            total_timesteps += timesteps_this_batch

            # Build arrays for observation, action for the policy gradient update by concatenating 
            # across paths
            train_data = prepare_training_data(paths, reward_to_go, gamma)
            ob_no = train_data['ob_no']
            ac_na = train_data['ac_na']
            rewards = train_data['rewards']
            q_n = train_data['q_n']
            
            # YOUR_CODE_HERE
            #====================================================================================#
            #                           ----------SECTION 5----------
            # Computing Baselines
            #====================================================================================#
            if nn_baseline:
                # If nn_baseline is True, use your neural network to predict reward-to-go
                # at each timestep for each trajectory, and save the result in a variable 'b_n'
                # like 'ob_no', 'ac_na', and 'q_n'.
                #
                # Hint #bl1: rescale the output from the nn_baseline to match the statistics
                # (mean and std) of the current or previous batch of Q-values. (Goes with Hint
                # #bl2 below.)

                b_n = sess.run([baseline_prediction], feed_dict={sy_ob_no: ob_no})[0]
                adv_n = q_n - (b_n + np.mean(q_n)) * np.std(q_n)
            else:
                adv_n = q_n.copy()

            #====================================================================================#
            #                           ----------SECTION 4----------
            # Advantage Normalization
            #====================================================================================#

            if normalize_advantages:
                # On the next line, implement a trick which is known empirically to reduce variance
                # in policy gradient methods: normalize adv_n to have mean zero and std=1. 
                # YOUR_CODE_HERE
                #pass
                adv_n = (adv_n - np.mean(adv_n))/np.std(adv_n)


            #====================================================================================#
            #                           ----------SECTION 5----------
            # Optimizing Neural Network Baseline
            #====================================================================================#
            basel_loss = -1
            if nn_baseline:
                # ----------SECTION 5----------
                # If a neural network baseline is used, set up the targets and the inputs for the 
                # baseline. 
                # 
                # Fit it to the current batch in order to use for the next iteration. Use the 
                # baseline_update_op you defined earlier.
                #
                # Hint #bl2: Instead of trying to target raw Q-values directly, rescale the 
                # targets to have mean zero and std=1. (Goes with Hint #bl1 above.)

                # YOUR_CODE_HERE
                #pass
                basel_loss, _ = sess.run([baseline_loss, baseline_update_op], 
                                         feed_dict={sy_ob_no: ob_no, sy_base_n: (q_n - np.mean(q_n))/np.std(q_n)})
            
            train_loss, _ = sess.run([loss, update_op], feed_dict={sy_ob_no: ob_no, sy_ac_na: ac_na, sy_adv_n: adv_n})
                
            if itr % log_interval == 0 or itr == n_iter - 1:
                print('iter:%d, train_loss:%.4f, baseline_loss:%.4f, eposodes:%d, reward mean:%.4f, std:%.4f'                       % (itr, train_loss, basel_loss, len(paths), np.mean(rewards), np.std(rewards)))

            # Log diagnostics
            returns = [path["reward"].sum() for path in paths]
            ep_lengths = [pathlength(path) for path in paths]
            logz.log_tabular("Time", time.time() - start)
            logz.log_tabular("Iteration", itr)
            logz.log_tabular("AverageReturn", np.mean(returns))
            logz.log_tabular("StdReturn", np.std(returns))
            logz.log_tabular("MaxReturn", np.max(returns))
            logz.log_tabular("MinReturn", np.min(returns))
            logz.log_tabular("EpLenMean", np.mean(ep_lengths))
            logz.log_tabular("EpLenStd", np.std(ep_lengths))
            logz.log_tabular("TimestepsThisBatch", timesteps_this_batch)
            logz.log_tabular("TimestepsSoFar", total_timesteps)
            logz.dump_tabular()
            logz.pickle_tf_vars()


# In[150]:
def train_v1():
    env_name = 'Humanoid-v1' #'CartPole-v0' # 
    train_PG(exp_name='test_PG',
            env_name=env_name,
            n_iter=100, 
            gamma=0.99, 
            min_timesteps_per_batch=256, 
            max_path_length=1000,
            learning_rate=5e-3, 
            reward_to_go=True, 
            animate=False, 
            logdir=None, 
            normalize_advantages=True,
            nn_baseline=True, 
            seed=0,
            # network arguments
            n_layers=3,
            size=500,
            log_interval=10)
    

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('env_name', type=str)
    parser.add_argument('--exp_name', type=str, default='vpg')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--discount', type=float, default=1.0)
    parser.add_argument('--n_iter', '-n', type=int, default=100)
    parser.add_argument('--batch_size', '-b', type=int, default=1000)
    parser.add_argument('--ep_len', '-ep', type=float, default=-1.)
    parser.add_argument('--learning_rate', '-lr', type=float, default=5e-3)
    parser.add_argument('--reward_to_go', '-rtg', action='store_true')
    parser.add_argument('--dont_normalize_advantages', '-dna', action='store_true')
    parser.add_argument('--nn_baseline', '-bl', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--n_experiments', '-e', type=int, default=1)
    parser.add_argument('--n_layers', '-l', type=int, default=1)
    parser.add_argument('--size', '-s', type=int, default=32)
    args = parser.parse_args()

    if not(os.path.exists('data')):
        os.makedirs('data')
    logdir = args.exp_name + '_' + args.env_name + '_' + time.strftime("%d-%m-%Y_%H-%M-%S")
    logdir = os.path.join('data', logdir)
    if not(os.path.exists(logdir)):
        os.makedirs(logdir)

    max_path_length = args.ep_len if args.ep_len > 0 else None

    for e in range(args.n_experiments):
        seed = args.seed + 10*e
        print('Running experiment with seed %d'%seed)
        def train_func():
            train_PG(
                exp_name=args.exp_name,
                env_name=args.env_name,
                n_iter=args.n_iter,
                gamma=args.discount,
                min_timesteps_per_batch=args.batch_size,
                max_path_length=max_path_length,
                learning_rate=args.learning_rate,
                reward_to_go=args.reward_to_go,
                animate=args.render,
                logdir=os.path.join(logdir,'%d'%seed),
                normalize_advantages=not(args.dont_normalize_advantages),
                nn_baseline=args.nn_baseline, 
                seed=seed,
                n_layers=args.n_layers,
                size=args.size
                )
        # Awkward hacky process runs, because Tensorflow does not like
        # repeatedly calling train_PG in the same thread.
        p = Process(target=train_func, args=tuple())
        p.start()
        p.join()
     

if __name__ == "__main__":
    #train_v1()
    main()

