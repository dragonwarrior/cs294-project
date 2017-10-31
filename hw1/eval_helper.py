import os
import time
import pickle
import numpy as np
import gym

def eval_policy(policy_f, envname):
    import gym
    max_steps = 200
    env = gym.make(args.envname)

    returns = []
    observations = []
    actions = []
    render = False
    for i in range(args.num_rollouts):
        obs = env.reset()
        done = False
        totalr = 0.
        steps = 0
        while not done:
            action = policy_f(obs[None,:])
            observations.append(obs)
            actions.append(action)
            obs, r, done, _ = env.step(action)
            totalr += r
            steps += 1
            if render:
                #time.sleep(0.1)
                env.render()

            if steps % 100 == 0: 
                print("%i/%i" % (steps, max_steps))

            if steps >= max_steps:
                break

        returns.append(totalr)

    #print('returns', returns)
    print('mean return', np.mean(returns))
    print('std of return', np.std(returns))

