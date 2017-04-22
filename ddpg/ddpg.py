from actor import Actor
from critic import Critic
from memory import ReplayBuffer
from process import OrnsteinUhlenbeck

import numpy as np
import gym


class DDPG:

    def __init__(self, state_space, action_space):
        self.rb_size = 1e6
        self.min_train_step = 100
        self.minibatch_size = 64
        self.gamma = 0.99
        self.tau = 0.001

        self.step = 0
        self.state_space = state_space
        self.action_space = action_space
        self.critic = Critic(state_space, action_space)
        self.tgt_critic = Critic(state_space, action_space)
        self.tgt_critic.set_weights(self.critic.get_weights())
        self.actor = Actor(state_space, action_space, self.critic.model)
        self.tgt_actor = Actor(state_space, action_space, self.critic.model)
        self.tgt_actor.set_weights(self.actor.get_weights())

        self.buffer = ReplayBuffer(self.rb_size)
        self.process = OrnsteinUhlenbeck(x0=np.zeros(action_space), theta=0.15, mu=0, sigma=0.2)

    def sense(self, s, a, r, s_new):
        self.buffer.add((s, a, r, s_new))

    def act(self, s):
        return self.tgt_actor(s) + self.process()

    def train_step(self):
        self.step += 1
        if self.step < self.min_train_step:
            if self.step % (self.min_train_step // 10) == 0:
                print "Still gathering data."
            return

        minibatch = self.buffer.sample(self.minibatch_size)
        ys = np.zeros(len(minibatch))
        states = np.zeros([len(minibatch), self.state_space])
        actions = np.zeros([len(minibatch), self.action_space])

        # Must be a faster way.
        for i in range(len(minibatch)):
            states[i], actions[i], r, s_new = minibatch[i]
            ys[i] = r + self.gamma * self.tgt_critic(s_new, self.tgt_actor(s_new))

        self.critic.step(states, actions, ys)
        self.actor.step(states)

        critic_weights = self.critic.get_weights()
        tgt_critic_weights = self.tgt_critic.get_weights()
        actor_weights = self.actor.get_weights()
        tgt_actor_weights = self.tgt_actor.get_weights()

        for i in range(len(critic_weights)):
            tgt_critic_weights[i] = (1 - self.tau) * tgt_critic_weights[i] + \
                self.tau * critic_weights[i]
        self.tgt_critic.set_weights(tgt_critic_weights)

        for i in range(len(actor_weights)):
            tgt_actor_weights[i] = (1 - self.tau) * tgt_actor_weights[i] + \
                self.tau * actor_weights[i]
        self.tgt_actor.set_weights(tgt_actor_weights)

ENV_NAME = 'Pendulum-v0'

if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    n_actions = env.action_space.shape[0]
    n_states = env.observation_space.shape[0]
    print n_states, n_actions

    agent = DDPG(n_states, n_actions)
    for i_episode in range(10000):
        s = env.reset()
        for t in range(100):
            env.render()
            s = np.reshape(s, (-1, n_states))
            a = agent.act(s)
            s_new, r, done, _ = env.step(a)
            agent.sense(s, a, r, np.reshape(s_new, (-1, n_states)))
            agent.train_step()
            if done:
                print("Episode finished after {} timesteps".format(t+1))
                break


