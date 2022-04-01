"""
Usage:
    policygradient_cartpole_train.py [options]

Options:
    --hidden-size=<int>                 size of hidden state [default: 8]
    --batch-size=<int>                  batch-size [default: 32]
    --max-iteration=<int>               number of update the model [default: 300]
    --episodes=<int>                     episode per iteration [default: 10]
    --discount-rate=<float>             discount rate of rewards [default: 0.98]
    --lr=<float>                        learning rate [default: 0.005]
    --model-save-every=<int>            save model frequency [default: 20]
    --model-save-path=<file>            model save path [default: ./]
"""

from docopt import docopt
import os

import tensorflow as tf
import gym
import numpy as np
from tensorflow.keras.layers import Dense

game_name = "CartPole-v0"
env = gym.make(game_name)

#### discount rewards
def discount_reward(rewards, discount_rate):
    '''
    :param rewards:  list:a list of rewards
    :param gradients:  float in [0,1]
    :return:  list:a discounted reward list
    '''
    if len(rewards) == 0:
        return rewards
    else:
        for i in range(len(rewards)-2, -1, -1):
            rewards[i] = rewards[i] + rewards[i+1] * discount_rate
        return rewards

def discount_and_normalize_rewards(all_rewards, discount_rate):
    '''
    :param all_rewards:   list of list of rewards
    :param discount_rate:  float in [0,1]
    :return: same structure with all_rewards,but processed
    '''
    new_all_rewards = [discount_reward(_, discount_rate) for _ in all_rewards]
    flatten = np.concatenate(new_all_rewards)
    mean = flatten.mean()
    std = flatten.std()
    new_all_rewards = [(np.array(_) - mean) / std for _ in new_all_rewards]
    return new_all_rewards

#### build a nn model
def nn(hidden_size=8):
    model = tf.keras.models.Sequential()
    model.add(Dense(hidden_size, activation='relu', input_dim=4))
    model.add(Dense(1, activation=tf.keras.activations.sigmoid))
    return model

####
def play_one_step(env,obs, model, loss_fn):
    with tf.GradientTape() as tape:
        obs = tf.expand_dims(obs, axis=0)
        ## pole向左偏地概率或是程度
        left_proba = model(obs)
        ## pole向左偏地越严重，action则更倾向于向左移动
        action = tf.cast(tf.random.uniform([1,1]) > left_proba, tf.float32)
        ## y_target可以理解成希望pole的倾斜方向：实际向左，则希望向右
        y_target = tf.constant([[1.0]])-action
        ## ****损失函数****
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    grads = tape.gradient(loss, model.trainable_variables)
    obs, reward, done, info = env.step(int(action[0][0].numpy()))
    return obs, reward, done, grads

def play_multi_episodes(env, n_episodes, model, loss_fn):
    all_rewards = []
    all_gradients = []
    for j in range(n_episodes):
        obs = env.reset()
        reward_i = []
        grads_i = []
        while 1:
            obs, reward, done, grads = play_one_step(env, obs, model, loss_fn)
            reward_i.append(reward)
            grads_i.append(grads)
            if done:
                break
        all_rewards.append(reward_i)
        all_gradients.append(grads_i)
    return all_rewards, all_gradients

#### train an agent
def train(args):
    hidden_size = int(args['--hidden-size'])
    n_iteration = int(args['--max-iteration'])
    n_episodes = int(args['--episodes'])
    discount_rate = float(args['--discount-rate'])
    lr = float(args['--lr'])
    save_model_every = int(args['--model-save-every'])
    save_path = args['--model-save-path']
    ##
    model = nn(hidden_size)
    loss_fn = tf.losses.binary_crossentropy
    optimizer = tf.keras.optimizers.Adam(lr)
    #### play and train
    for i in range(1, n_iteration+1):
        print('update model : iteration = %4d/%4d'%(i,n_iteration))
        all_rewards, all_gradients = play_multi_episodes(env, n_episodes, model, loss_fn)
        all_rewards = discount_and_normalize_rewards(all_rewards, discount_rate)
        mean_grads = []
        for j in range(len(model.trainable_variables)):
            mean_grads_ = tf.reduce_mean(
                [all_gradients[episode][step][j] * reward
                    for episode,rewards in enumerate(all_rewards)
                        for step, reward in enumerate(rewards)], axis=0)
            mean_grads.append(mean_grads_)
        optimizer.apply_gradients(zip(mean_grads,model.trainable_variables))
        #### save model
        if i % save_model_every == 0:
            model_name = save_path + 'pg_cartpole_%04d.h5' % i
            model.save(model_name)
            print('**model saved in ' + model_name)

if __name__ == '__main__':
    args = docopt(__doc__)
    save_path = args['--model-save-path']
    if not os.path.exists(save_path):
        assert os.path.exists(save_path)==True, "--model-save-path is NOT exists"
    train(args)
