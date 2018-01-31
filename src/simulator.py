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

states   = []
slotvals = []
goals    = []
state_actions  = {}
allactions     = {}
state_parents  = {}
state_children = {}

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

        for action in actions:
            print(action)
            allactions.append(state+":"+action)

print(allactions)
print(goals)
states_nda = np.array(states,dtype=str)

def goal_sampling():
    return random.shuffle(goals)




