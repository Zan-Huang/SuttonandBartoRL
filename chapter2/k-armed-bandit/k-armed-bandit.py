import numpy as np
from numpy.random import seed
from numpy.random import randint
from numpy.random import normal

import random

from scipy.stats import norm
import matplotlib.pyplot as plt

seed(2)

class Bandit:
    def __init__(self, k, rounds):
        self.arms = []
        self.size = k
        self.rounds = rounds

    def generate_bandit(self):
        values = np.random.uniform(low = -3, high = 3, size =(self.size, ))
        print("average rewards:", end = " ")
        print(values)
        for i in values:
            single_arm = normal(loc=i, scale=1, size=self.rounds)
            self.arms.append(single_arm)
        return values

    def print_bandit(self):
        for i in range(self.size):
            print(self.arms[i])

    def return_arms(self):
        return_array = np.vstack(self.arms)
        return_array = return_array.T
        print("returning array of shape:", end = " ")
        print(return_array.shape)
        return return_array

    def plot_reward_distribution(self):
        print("plotting reward distribution")
        plt.style.use('_mpl-gallery')
        plot_array = np.vstack(self.arms)
        fig, ax = plt.subplots()
        ax.eventplot(plot_array, orientation="vertical", linewidth=0.2)
        ax.set(ylim=(-7, 7), yticks=np.arange(-7, 7))
        plt.show()


class epsilon_greedy_action:
    def __init__(self, epsilon, bandit_size, bandit_input, optimal_vector):
        self.epsilon = epsilon
        self.q_vector = np.zeros(bandit_size)
        self.n = np.zeros(bandit_size)
        self.accum_reward = 0
        self.bandit_input = bandit_input
        self.opt_choice_val = 0
        self.optimal_vector = optimal_vector

    def run_step(self, step):
        opt_choice = np.argmax(self.optimal_vector)
        random_val = random.random()
        if(random_val < self.epsilon):
            random_int = randint(0, self.q_vector.size)
            if(self.n[random_int] != 0):
                self.q_vector[random_int] = self.q_vector[random_int] + (self.bandit_input[step][random_int]  - self.q_vector[random_int])/self.n[random_int]
            else:
                self.q_vector[random_int] = self.bandit_input[step][random_int]
            self.accum_reward += self.bandit_input[step][random_int]

            if(random_int == opt_choice):
                self.opt_choice_val += 1
            self.n[random_int] += 1

        else:
            select_int = np.argmax(self.q_vector)
            if(self.n[select_int] != 0):
                self.q_vector[select_int] = self.q_vector[select_int] + (self.bandit_input[step][select_int] - self.q_vector[select_int])/self.n[select_int]
            else:
                self.q_vector[select_int] = self.bandit_input[step][select_int]
            self.accum_reward += self.bandit_input[step][select_int]
            if(opt_choice == select_int):
                self.opt_choice_val += 1
            self.n[select_int] += 1

    def return_accum_reward(self):
        return self.accum_reward

    def return_opt_number(self):
        return self.opt_choice_val


bandit1 = Bandit(10, 1000)
opt_val = bandit1.generate_bandit()
#bandit1.print_bandit()
print(bandit1.return_arms())
#bandit1.plot_reward_distribution()

model = epsilon_greedy_action(0.1, 10, bandit1.return_arms(), opt_val)
list_x = []
list_y = []

opt_y = []

for i in range(999):
    model.run_step(i)
    avg_reward = model.return_accum_reward() / (i+1)
    opt_y.append(model.return_opt_number() / (i+1))
    list_x.append(i)
    list_y.append(avg_reward)

xpoints = np.array(list_x)
ypoints = np.array(list_y)
plt.plot(xpoints, ypoints)

yopt = np.array(opt_y)
plt.plot(xpoints, opt_y)

print(model.return_accum_reward())
plt.show()
