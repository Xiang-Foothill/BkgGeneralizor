import numpy as np
import matplotlib.pyplot as plt

class continuous_MC():

    def __init__(self, states, pi):
        """@states: a string array representing the state space of the continuous-time
        markov chain, with the ith element representing the name of the ith state
        @pi: the wanted stationary distribution of the markov chain, should have the same length
        as states

        the initial state will be set to state 0
        """

        if abs(np.sum(pi) - 1) >= 0.01:
            raise ValueError("invalid values fo stationary distribution. The distribution must sum up to 1")

        self.states = states
        self.pi = pi
        self.Q = self.Q_generator()

        self.cur = 0
        self.t = 0

        self.next_update = np.random.exponential(scale = 1 / - self.Q[self.cur, self.cur])
    
    def Q_generator(self):
        """a helper function that generatest the Q matrix based on the desired stationary distribution"""
        n = len(self.states)

        Q = np.ones(shape = (n, n))
        Q = Q / 30

        for i in range(n):
            for j in range(n):
                Q[i][j] = Q[i][j] / self.pi[i] if i != j else 0
        
        for i in range(n):
            Q[i][i] = - np.sum(Q[i])
        
        return Q
    
    def update_chain(self, t):
        """@t: the time"""
        self.t = t

        n = len(self.states)

        if self.t >= self.next_update:
            # choose the next state
            values = np.arange(n)
            probs = self.Q[self.cur] / (- self.Q[self.cur, self.cur])
            probs[self.cur] = 0

            self.cur = np.random.choice(values, size = 1, p = probs)[0]

            self.next_update = self.t + np.random.exponential(scale = 1 / - self.Q[self.cur, self.cur])
        
        return self.states[self.cur]
    
def test():
    states = [0, 1, 2, 3]
    pi = [0.5, 0.4, 0.05, 0.05]
    MC = continuous_MC(states = states, pi = pi)
    dt = 0.001
    t = 0

    state_freq = [0, 0, 0, 0]
    time_series = []

    while t < 200:
        t += dt
        cur_state = MC.update_chain(t)

        state_freq[cur_state] = state_freq[cur_state] + 1
        time_series.append(cur_state)
    
    state_freq = state_freq / np.sum(state_freq)
    print(state_freq)
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.hist(time_series, bins=10, color='skyblue', edgecolor='black')
    ax1.set_xlabel('states')
    ax1.set_ylabel('Frequency')

    ax2.plot(time_series)
    ax2.set_xlabel("time / ms")
    ax2.set_ylabel("state")
    plt.show()


if __name__ == "__main__":
    test()



    
    