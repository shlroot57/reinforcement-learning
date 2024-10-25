import random

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

N = 190
p = 0.49

def sample(N,p,pi,timestep_max,number):

    episodes = []
    for _ in range(number):
        timestep = 0
        episode = []
        s = np.random.randint(1,N-1)
        while 1<= s < N and timestep <= timestep_max:
            timestep += 1
            #根据策略选择动作
            if pi == 'all_in':
                if s < 0.5*N:
                    a = s
                else:
                    a = N-s
            elif pi == 'one_dollar':
                a = 1
            elif pi =='two_dollar':
                if s == 1 and s == N-1:
                    a = 1
                else:
                    a = 2
            #转移状态
            if np.random.rand()<p:
                s_next =s + a
            else:
                s_next =s - a

            if s_next >= N:
                s_next = N
                r = 1
            else:
                r = 0

            episode.append((s,a,r,s_next))
            s = s_next
        episodes.append(episode)
    return episodes

def MC(episodes,N,gamma):
    V = np.zeros(N+1)
    NS = np.zeros(N+1)
    for episode in episodes:
        G=0
        for i in range(len(episode)-1, -1, -1):
            (s,a,r,s_next) = episode[i]
            G = r + G*gamma
            NS[s] = NS[s] + 1
            V[s] = V[s] + (G-V[s] / NS[s])
    return V

timestep_max = 1000

episodes_allin = sample(N,p,'all_in',timestep_max,10000)
episodes_one = sample(N,p,'one_dollar',timestep_max,10000)
episodes_two = sample(N,p,'two_dollar',timestep_max,10000)
values_allin = MC(episodes_allin,N,0.9)
values_one = MC(episodes_one,N,0.9)
values_two = MC(episodes_two,N,0.9)



plt.figure(figsize=(20, 10))

plt.subplot(1, 3, 1)
plt.plot(values_allin,label='all_in')
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.title('all_in')

plt.subplot(1, 3, 2)
plt.plot(values_one,label='one_dollar')
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.title('one_dollar')

plt.subplot(1, 3, 3)
plt.plot(values_two,label='two_dollar')
plt.xlabel('Capital')
plt.ylabel('Value estimates')
plt.title('two_dollar')

plt.savefig('../figure_gamblers_problem_MC.png')
plt.close()
