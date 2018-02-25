#! python
### User simulator of dialogue manager ###
### Koichiro Yoshino, Jan 28, 2018     ###


import json
import numpy as np
import random

f = open("../data/dialogframe.txt")
dialogframe = json.load(f)
f.close

tasks = dialogframe.keys()

states     = []
slotvals   = []
goals      = []
allactions = []
state_actions  = {}
state_parents  = {}
state_children = {}
state_pos = {}

pos = 0
for task in tasks:
    for taskstate in dialogframe[task]:
        state   = taskstate['state-index']
        goalp   = taskstate['goal-possible']
        slotval = taskstate['slot-value']
        actions = taskstate['actions']
        parents = taskstate['parents']
        children= taskstate['children']
        
        states.append(state)
        slotvals.append(slotval)
        if goalp == 1:
            goals.append(state)

        state_actions[state]  = actions
        state_parents[state]  = parents
        state_children[state] = children
        state_pos[state] = pos
        pos += 1

        for action in actions:
            allactions.append(str(state)+":"+action)

states_nda = np.array(states,dtype=int)
goals_nda  = np.array(goals,dtype=int)

def goal_sampling():
    goal = np.random.choice(goals)
    return goal

def possible_states(goal,cstate):
    goal_parents = state_parents[goal]
    cstate_children = state_children[cstate]
    statecandidates = list(set(goal_parents) & set(cstate_children))
    if goal == cstate:
        statecandidates.append(goal)
    else:
        if goal in cstate_children:
            statecandidates.append(goal)
            statecandidates.append(cstate)
    return(statecandidates)

def possible_goals(goals,cstate):
    cstate_children = state_children[cstate]
    goalcandidates = list(set(goals) & set(cstate_children))
    if cstate in goals:
        goalcandidates.append(cstate)
    return(goalcandidates)
    
def state_sampling(goal,cstate):
    nstates = possible_states(goal,cstate)
    nstate = random.choice(nstates)
    return nstate

def noise_sampling(state,confidence):
    noise  = 1.0 - confidence
    prob = np.random.rand(len(states))
    pos    = state_pos[state]
    prob[pos] = 0.0
    bsum   = np.sum(prob)
    prob = (prob * noise) / bsum
    prob[pos] = confidence
    return(states_nda,prob)

def noisy_state(prob):
    noisystate = np.random.choice(states_nda,p=prob)
    return noisystate

goal_matrix = np.zeros((len(states),len(states)))
for state in states:
    spos    = state_pos[state]
    pgoals = possible_goals(goals,state)
    prob = 1.0/float(len(pgoals))
    for pgoal in pgoals:
        gpos    = state_pos[pgoal]
        goal_matrix[gpos][spos] = prob

simulator_tensor = np.zeros((len(states),len(states),len(states)))
for caction in states:
    cpos = state_pos[caction]
    for goal in goals:
        gpos = state_pos[goal]
        pstates = possible_states(goal,caction)
        if pstates != []:
            prob = 1.0/(float(len(pstates)))
            for nstate in pstates:
                npos = state_pos[nstate]
                simulator_tensor[cpos][npos][gpos] = prob
        else:
            prob = 1.0/(float(len(states)))
            for nstate in states:
                npos = state_pos[nstate]
                simulator_tensor[cpos][npos][gpos] = prob

statetrans_tensor = np.zeros((len(states),len(states),len(states)))
for x in range(len(simulator_tensor)):
#    print(x)
    statetrans_tensor[x] = np.dot(simulator_tensor[x],goal_matrix)
    
def statetransition(belief,caction):
    cpos = state_pos[caction]
    statetrans_matrix = statetrans_tensor[cpos]
    transition = np.dot(statetrans_matrix,belief[1])
    return transition

def beliefupdate(observation,transition):
    nprob = observation[1] * transition[1]
    nprob += 0.00001
    psum  = np.sum(nprob)
    nprob = nprob / psum
    return(observation[0],nprob)


# test
goal = goal_sampling()
print("goal",goal)

pstate = state_sampling(goal,-1)
print("1st State:",pstate,"(from -1)")
pobs   = noise_sampling(pstate,0.4)
pblf   = (pobs)
print("1st Observation:",pobs,"(from -1)")
print("1st Belief:",pblf,"(from -1)")
ctrans = statetransition(pblf,pstate)

cstate = state_sampling(goal,pstate)
print("2nd State:",cstate)
cobs   = noise_sampling(cstate,0.8)
print("2nd Observation",cobs)
cblf   = beliefupdate(cobs,pblf)
print("2nd Belief:",cblf)
