import env
from collections import deque
import numpy as np
import gym
import keras.backend as K
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.initializers import RandomUniform, VarianceScaling
from keras import optimizers

from ddpg import memory

ENV_NAME = 'Pendulum-v0'


def create_actor(n_states, n_actions):
    state_input = Input(shape=(n_states,))

    h = Dense(16, activation='relu')(state_input)
    h = Dense(16, activation='relu')(h)
    h = Dense(16, activation='relu')(h)
    out = Dense(n_actions,  activation='linear')(h)
    return Model(inputs=[state_input], outputs=[out])


def create_critic(n_states, n_actions):
    state_input = Input(shape=(n_states,))
    action_input = Input(shape=(n_actions,))

    h = Dense(32, activation='relu')(concatenate([state_input, action_input]))
    h = Dense(32, activation='relu')(h)
    h = Dense(32, activation='relu')(h)
    out = Dense(1, activation='linear')(h)

    return Model(inputs=[state_input, action_input], outputs=out)


if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    n_actions = env.action_space.shape[0]
    n_states = env.observation_space.shape[0]
    n_episodes = 2000

    actor, tgt_actor = create_actor(n_states, n_actions), create_actor(n_states, n_actions)
    critic, tgt_critic = create_critic(n_states, n_actions), create_critic(n_states, n_actions)

    action_limits = [env.action_space.low, env.action_space.high]

    agent = ddpg.DDPGAgent(actor, tgt_actor, critic, tgt_critic, action_limits)

    s_test = np.reshape(env.reset(), [-1, n_states])
    rewards = deque(100)
    for i_episode in range(n_episodes):
        if (i_episode + 1) % 200 == 0:
            raw_input("Enter!")
            agent.learning_phase = False
            s = env.reset()
            r_sum = 0
            for t in range(500):
                env.render()
                a = agent.act(s)
                s_new, r, _, _ = env.step(a)
                agent.sense(s, a, r, s_new)
                agent.train_step()
                s = s_new
                r_sum += r
            print r_sum / 500
            rewards.add(r_sum / 500)
            agent.learning_phase = True
        else:
            agent.new_episode()
            s = env.reset()
            r_sum = 0
            for t in range(200):
                env.render()
                a = agent.act(s)
                s_new, r, _, _ = env.step(a)
                agent.sense(s, a, r, s_new)
                agent.train_step()
                r_sum += r
                s = s_new
            rewards.add(r_sum / 200)

        print i_episode, r_sum / 200, \
            agent.tgt_critic(s_test, agent.tgt_actor(s_test)), \
            agent.critic(s_test, agent.tgt_actor(s_test)), \
            agent.tgt_actor(s_test), \
            agent.actor(s_test), \
            sum(rewards) / len(rewards)
        # np.linalg.norm(agent.tgt_actor.get_weights()[5]), \
        # np.linalg.norm(agent.tgt_critic.get_weights()[5]), \
