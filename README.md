# Q-learning-DM
This is an example of dialogue manager based on reinforcement learning.
The learning algorithm is based on Q-learning.
The system converts the task structure that is written in data/dialogueframe.txt into probability distribution to work the belief update and user simulator.
Details are written in the following paper.

Koichiro Yoshino, Shinji Watanabe, Jonathan Le Roux, John R. Hersh
Statistical Dialogue Management using Intention Dependency Graph
In Proc. IJCNLP2013.
http://aclweb.org/anthology//I/I13/I13-1127.pdf

## Required
- python (3.5 or later)
- numpy (1.13.3 or later)

## How to run

python Qlearn.py

python simulator.py also work for the test of user simulator.
