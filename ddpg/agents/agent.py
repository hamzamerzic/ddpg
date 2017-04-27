from collections import deque
from timeit import default_timer as timer
try:
    import cPickle as pickle
except:
    import pickle


class Agent(object):

    def __init__(self, warmup_episodes, logging=True):
        self.learning_phase = True
        self.warmup_episodes = warmup_episodes if warmup_episodes is not None else 0
        self.logging = logging
        if logging:
            self.logs = []

    def sense(self, s, a, r, s_new):
        raise NotImplementedError()

    def act(self, s):
        raise NotImplementedError()

    def new_episode(self, episode):
        raise NotImplementedError()

    def train_step(self):
        raise NotImplementedError()

    def add_log(self, agent_returns):
        if not self.logging:
            raise RuntimeError('Logging disabled!')
        if not self.logs:
            self.logs.append({})
            self.logs[-1]['episode'] = 1  # Initial episode.
        for name, value in agent_returns:
            if name not in self.logs[-1]:
                self.logs[-1][name] = []
            self.logs[-1][name].append(value)

    def dump_logs(self, filename):
        if not self.logging:
            raise RuntimeError('Logging disabled!')
        pickle.dump(self.logs, open(filename, "wb"))

    def train(self, env, n_episodes, n_steps, render_period=None, reward_window=100):
        start = timer()
        try:
            rewards = deque(maxlen=reward_window)
            for episode in range(1, n_episodes + 1):
                self.new_episode()
                s = env.reset()

                r_sum = 0
                for t in range(n_steps):
                    if render_period is not None and episode % render_period == 0:
                        env.render()
                    a = self.act(s)
                    s_new, r, _, done = env.step(a)
                    self.sense(s, a, r, s_new)
                    if episode > self.warmup_episodes:
                        self.train_step()
                    r_sum += r
                    s = s_new
                    if done:
                        break

                rewards.append(float(r_sum) / t)
                print "Ep %4d, last reward %.5f, mean reward %.5f." % (episode, rewards[-1],
                                                                      sum(rewards) / len(rewards))
        except KeyboardInterrupt:
            print "Training interrupted by the user!"

        end = timer()
        duration = end - start
        print "Performed %d episodes. Elapsed time %f. Average time per episode %f." % \
            (episode - 1, duration, duration / (episode - 1))

    def eval(self, env, n_episodes=1, n_steps=200, render=True):
        learning_phase = self.learning_phase
        self.learning_phase = False

        for episode in range(n_episodes):
            print "Ep %4d" % (episode)
            done = False
            s = env.reset()

            for _ in range(n_steps):
                if render:
                    env.render()
                a = self.act(s)
                s_new, _, _, done = env.step(a)
                s = s_new
                if done:
                    break

        self.learning_phase = learning_phase
