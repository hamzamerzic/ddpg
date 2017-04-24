from collections import deque

# TODO: Clean up and add visualizations.


class Agent(object):

    def __init__(self, warmup_steps):
        self.step = 0
        self.learning_phase = True
        self.warmup_steps = warmup_steps

    def sense(self, s, a, r, s_new):
        raise NotImplementedError()

    def act(self, s):
        raise NotImplementedError()

    def new_episode(self):
        raise NotImplementedError()

    def train(self, env, n_episodes, n_steps, render_period=None):
        try:
            rewards = deque(maxlen=100)
            for i_episode in range(1, n_episodes + 1):
                self.new_episode()

                if render_period is not None and i_episode % render_period == 0:
                    self.learning_phase = False
                    render = True
                else:
                    self.learning_phase = True
                    render = False

                s = env.reset()
                r_sum = 0
                for t in range(n_steps):
                    if render:
                        env.render()
                    a = self.act(s)
                    s_new, r, _, done = env.step(a)
                    self.sense(s, a, r, s_new)
                    self.train_step()
                    r_sum += r
                    s = s_new

                rewards.append(r_sum / n_steps)
                print "Ep %4d" % (i_episode), " last reward ", rewards[-1], \
                    " reward mean ", sum(rewards) / len(rewards)
        except KeyboardInterrupt:
            print "Training interrupted by the user!"
