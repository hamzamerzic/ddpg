import keras.backend as K
from keras.layers import Input, Dense, concatenate
from keras.models import Model
from keras.initializers import RandomUniform, VarianceScaling
from keras import optimizers


class Critic:

    def __init__(self, state_space, action_space):
        self.lr = 1e-3
        self.decay = 1e-2

        self.state_input = Input(shape=(state_space,))
        self.action_input = Input(shape=(action_space,))

        w_init = VarianceScaling(scale=1./3, mode='fan_in', distribution='uniform')
        h1 = Dense(400, kernel_initializer=w_init,
                   bias_initializer=w_init, activation='relu')(self.state_input)
        x = concatenate([h1, self.action_input])
        h2 = Dense(300, kernel_initializer=w_init, bias_initializer=w_init, activation='relu')(x)

        w_init = RandomUniform(-3e-3, 3e-3)
        out = Dense(1, kernel_initializer=w_init, bias_initializer=w_init, activation='linear')(x)

        self.model = Model(inputs=[self.state_input, self.action_input], outputs=out)
        self.opt = optimizers.Adam(lr=self.lr, decay=self.decay)
        self.model.compile(self.opt, loss='mse')

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def step(self, states, actions, ys):
        self.model.train_on_batch([states, actions], ys)

    def __call__(self, states, actions):
        return self.model.predict_on_batch([states, actions])
