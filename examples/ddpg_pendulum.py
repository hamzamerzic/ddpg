import env
import gym
import keras.backend as K
from keras.initializers import RandomUniform, VarianceScaling
from keras.layers import Input, Dense, concatenate, Lambda
from keras.models import Model
from keras import optimizers

from ddpg import actor, critic, process
from ddpg.agents import ddpg_agent

ENV_NAME = 'Pendulum-v0'


def create_actor(n_states, n_actions):
    state_input = Input(shape=(n_states,))

    w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
    h1 = Dense(400, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(state_input)
    h2 = Dense(300, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(h1)

    w_init = RandomUniform(-3e-3, 3e-3)
    out = Dense(n_actions, kernel_initializer=w_init,
                bias_initializer=w_init, activation='tanh')(h2)
    out = Lambda(lambda x: 2 * x)(out)  # Since the output range is -2 to 2.

    return Model(inputs=[state_input], outputs=[out])


def create_critic(n_states, n_actions):
    state_input = Input(shape=(n_states,))
    action_input = Input(shape=(n_actions,))

    w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
    h1 = Dense(400, kernel_initializer=w_init,
               bias_initializer=w_init, activation='relu')(state_input)
    x = concatenate([h1, action_input])
    h2 = Dense(300, kernel_initializer=w_init, bias_initializer=w_init, activation='relu')(x)

    w_init = RandomUniform(-3e-3, 3e-3)
    out = Dense(1, kernel_initializer=w_init, bias_initializer=w_init, activation='linear')(h2)

    return Model(inputs=[state_input, action_input], outputs=out)


if __name__ == '__main__':
    env = gym.make(ENV_NAME)

    n_actions = env.action_space.shape[0]
    n_states = env.observation_space.shape[0]
    n_episodes = 1000
    n_steps = 200
    render_period = 20

    actor, tgt_actor = create_actor(n_states, n_actions), create_actor(n_states, n_actions)
    critic, tgt_critic = create_critic(n_states, n_actions), create_critic(n_states, n_actions)

    action_limits = [env.action_space.low, env.action_space.high]

    agent = ddpg_agent.DDPGAgent(actor, tgt_actor, critic, tgt_critic, action_limits,
                                 critic_decay=0)
    agent.train(env, n_episodes, n_steps, render_period)

    print "Storing the logs..."
    agent.dump_logs("logs.pkl")

    print "Performing 5 evaluation steps..."
    agent.eval(env, 5)

    print "Saving the model..."
    actor.save("actor.model", overwrite=True)
    critic.save("critic.model", overwrite=True)
