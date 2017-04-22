import keras.backend as K
from keras.layers import Input, Dense
from keras.models import Model
from keras.initializers import RandomUniform, VarianceScaling
from keras import optimizers


class Actor:

    def __init__(self, state_space, action_space, critic_model):
        self.lr = 1e-4

        state_input = Input(shape=(state_space,))

        w_init = VarianceScaling(
            scale=1./3, mode='fan_in', distribution='uniform')
        h1 = Dense(400, kernel_initializer=w_init,
                   bias_initializer=w_init, activation='relu')(state_input)
        h2 = Dense(300, kernel_initializer=w_init,
                   bias_initializer=w_init, activation='relu')(h1)

        w_init = RandomUniform(-3e-3, 3e-3)
        out = Dense(action_space, kernel_initializer=w_init,
                    bias_initializer=w_init, activation='tanh')(h2)

        self.model = Model(inputs=state_input, outputs=out)
        self.opt = optimizers.Adam(self.lr)

        critic_out = critic_model([state_input, out])

        updates = self.opt.get_updates(
            params=self.model.trainable_weights, constraints=self.model.constraints,
            loss=[critic_out])
        self.train_step = K.function(inputs=[state_input], outputs=[], updates=updates)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def step(self, states):
        self.train_step([states])

    def __call__(self, state):
        return self.model.predict_on_batch(state)
