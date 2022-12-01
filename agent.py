import torch
import numpy as np
import random
from collections import deque
from game import SnakeGame, Direction, Point

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001



class Agent:
    def __init__(self):
        #store more paramters like num games, epsilon for randomness
        #gamma memory(use deque), model and trainer
        pass

    def get_state(self,game):
        # state = [ds, dr,du,dd,movedir,fl,fr,fu,fd] 9 tuple)
        pass

    def remember(self, state, action, reward, next_state, done):
        # append to deque
        pass

    def train_long_memory(self):
        #fetch batch using random.sample
        pass

    def train_short_memory(self, state, action, reward, next_state, done):
        # call train_step()
        pass

    def get_action(self, state):
        # random moves tradeoff between exploration and pediction
        pass

def train():
    # lists for keeping track of scores
    # mean scores and final scores
    # total score, best score
    # agent 
    # game object
    # training loop
    #   get old state from game
    #   get move based on state
    #   perform move and get new state
    #   set new state
    #   train short memory
    #   remember and store in memory
    #   check if done -> train long memory, plot result, reset game
    #   save model if new high score and plot
    pass

if __name__ == '__main__':
    train()