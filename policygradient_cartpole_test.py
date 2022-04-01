"""
Usage:
    policygradient_cartpole_test.py [options]

Options:
    --model-save-path=<file>            model save path [default: ./]
    --test-episodes=<int>               how many episodes to test the model [default: 100]
    --render=<bool>                     whether to render when playing game [default: False]
    --result-save-path=<file>           save the result into a csv [default: ./]
"""

from docopt import docopt

import tensorflow as tf
import gym
import time
from tensorflow.keras.layers import Dense
import numpy as np
import os
import pandas as pd


#### try the game
def test_CartPole(model_name,env,n_episodes=10,render=False):
    sum_step = 0
    max_step_list = []
    for _ in range(1, n_episodes+1):
        obs = env.reset()
        for i in range(1, 210):
            left_proba = model(obs[np.newaxis])
            action_i = tf.cast(tf.random.uniform([1]) > left_proba,tf.float32)
            obs, reward, done, info = env.step(int(action_i[0][0]))
            if render:
                time.sleep(0.01)
                env.render()
            if done:
                print(model_name+': episode %3d,max_step = %3d'%(_,i))
                sum_step = sum_step + i
                max_step_list.append(i)
                env.close()
                break
    print('CartPole:mean max step = %.2f'%(sum_step / n_episodes))
    print('--------')
    print('\n')
    return([sum_step / n_episodes] + max_step_list)


if __name__ == '__main__':
    args = docopt(__doc__)
    saved_model_path = args['--model-save-path']
    test_episodes = int(args['--test-episodes'])
    render = eval(args["--render"])
    save_result_path = args["--result-save-path"]
    print(args)
    ####
    game_name = "CartPole-v0"
    env = gym.make(game_name)
    all_model = os.listdir(saved_model_path)
    all_model.sort()
    all_result = []
    for model_ in all_model:
        if 'h5' in model_:
            model = tf.keras.models.load_model(saved_model_path+model_)
            result = test_CartPole(model_, env, n_episodes=test_episodes, render=render)
            result_model_ = [model_] + result
            all_result.append(result_model_)
    df = pd.DataFrame(all_result)
    df.columns = ['model_name'] + ['mean_max_step'] + ["epi_%d" % x for x in range(1, test_episodes + 1)]
    df.sort_values(by='mean_max_step', ascending=False, inplace=True)
    csv_name = save_result_path + 'pg_cartpole_test_result.csv'
    df.to_csv(csv_name, index=0)
    print('**result csv file saved in ' + csv_name)
