import env
import gym
import keras.backend as K
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras import optimizers

from ddpg import actor, critic, process
from ddpg.agents import ddpg_agent

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
    n_episodes = 10000
    n_steps = 200
    render_period = 50

    actor, tgt_actor = create_actor(n_states, n_actions), create_actor(n_states, n_actions)
    critic, tgt_critic = create_critic(n_states, n_actions), create_critic(n_states, n_actions)

    action_limits = [env.action_space.low, env.action_space.high]

    agent = ddpg_agent.DDPGAgent(actor, tgt_actor, critic, tgt_critic, action_limits,
                                 actor_lr=1e-3, critic_decay=0, rb_size=1e5, tau=1e-3)
    agent.train(env, n_episodes, n_steps, render_period)

    print "Saving weights..."
    actor.save_weights("actor.model", overwrite=True)
    critic.save_weights("critic.model", overwrite=True)
    print "Done."
