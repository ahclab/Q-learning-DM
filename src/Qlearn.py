#! python
### Q-learning of dialogue manager ###
### Koichiro Yoshino Feb 1, 2018   ###

import numpy as np
import random
import simulator

states = simulator.states
state_actions = simulator.state_actions
stateaction_sysuttr = {}
actions = []
confs   = [0.0,0.2,0.4,0.6,0.8,1.0]
#confs = [0.0,1.0]

for state in states:
    for stateactionframe in state_actions[state]:
        if stateactionframe not in actions:
            actions.append(stateactionframe)

states_nda  = np.array(states)
actions_nda = np.array(actions)
confs_nda   = np.array(confs)

qval = np.zeros((len(confs),len(states),len(actions)))

def best_action(state,conf):
    conf = round(conf*5)/5
    #conf = round(conf)
    cpos = np.where(confs_nda == round(maxblf*5)/5)[0][0]
    #cpos = np.where(confs_nda == round(maxblf))[0][0]
    spos = np.where(states_nda == state)[0][0]
    apos = np.where(qval[cpos][spos] == max(qval[cpos][spos]))[0][0]
    action = actions[apos]
    return action
    

# learning rate, randomness, discount factor
alpha   = 0.01
epsilon = 0.001
gamma   = 0.5

diff = 0
for dial in range(100000):

    turns = 1
    
    if dial % 500 == 0:
        totalqval = np.average(qval)
        print(dial,totalqval,diff-totalqval)
        diff = totalqval
    
    goal  = simulator.goal_sampling()
    
    state = simulator.state_sampling(goal,-1)
    conf  = random.random()
    obs   = simulator.noise_sampling(state,conf)
    noisystate = simulator.noisy_state(obs[1])
    noisyobs = simulator.noise_sampling(noisystate,conf)
    blf   = noisyobs
    maxblf = np.max(blf[1])
    maxblfstate = states_nda[np.where(blf[1]==maxblf)[0][0]]
    
    erand = random.random()
    if epsilon < erand:
        action = best_action(maxblfstate,maxblf)
    else:
        action = random.choice(actions)
        
    reward = -1
    if goal == state and action == "goal":
        reward += 10
    if goal == state and action == "do":
        reward += -5
    if goal != state and action == "goal":
        reward += -50
    if goal != state and action == "do":
        reward += 1
        
    cpos = np.where(confs_nda == round(maxblf*5)/5)[0][0]
    apos = np.where(actions_nda == action)[0][0]
    spos = np.where(states_nda == maxblfstate)[0][0]

    while True:        

#        print(goal,state,conf)
        
        if action == "confirm":
            nstate = state
            noise = 1-conf
            nconf = 1-(noise/2)
        else:
            nstate = simulator.state_sampling(goal,state)
            nconf = random.random()
            
        nobs   = simulator.noise_sampling(nstate,nconf)

        nnoisystate = simulator.noisy_state(nobs[1])
        nnoisyobs = simulator.noise_sampling(nnoisystate,nconf)
        nblf   = simulator.beliefupdate(nnoisyobs,blf)
        nmaxblf = np.max(nblf[1])
        nmaxblfstate = states_nda[np.where(nblf[1]==nmaxblf)[0][0]]
        nbestaction = best_action(nmaxblfstate,nmaxblf)
                
        ncpos = np.where(confs_nda == round(nmaxblf*5)/5)[0][0]
        #ncpos = np.where(confs_nda == round(nmaxblf))[0][0]
        napos = np.where(actions_nda == nbestaction)[0][0]
        nspos = np.where(states_nda == nmaxblfstate)[0][0]
        
        if action == "goal":
            qval[cpos][spos][apos] = (1-alpha) * qval[cpos][spos][apos] + alpha * reward
        else:        
            qval[cpos][spos][apos] = (1-alpha) * qval[cpos][spos][apos] + alpha * (reward + gamma * qval[ncpos][nspos][napos])

        if action == "goal":
            break

        turns += 1
        if turns == 6:
            break
        
        erand = random.random()
        if epsilon < erand:
            naction = best_action(nmaxblfstate,nmaxblf)
        else:
            naction = random.choice(actions)

        reward = -1
        if goal == nstate and naction == "goal":
            reward += 10
        if goal == nstate and naction == "do":
            reward += -5
        if goal != nstate and naction == "goal":
            reward += -50
        if goal != nstate and naction == "do":
            reward += 1
        if turns == 5:
            reward += -50

        conf  = nconf
        state = nstate
        obs   = nobs
        blf   = nblf
        maxblf = nmaxblf
        action = naction
        
        cpos = ncpos
        apos = napos
        spos = nspos


print(qval)
for conf in confs:
    for state in states:
        print(conf,state,best_action(state,conf))
