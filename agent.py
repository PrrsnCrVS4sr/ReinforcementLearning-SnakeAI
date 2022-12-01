import torch
import numpy as np
import random
from collections import deque
from game import SnakeGame, Direction, Point, BLOCK_SIZE

MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001



class Agent:
    def __init__(self):
        #store more paramters like num games, epsilon for randomness
        #gamma memory(use deque), model and trainer
        self.n_games = 0
        self.epsilon = 0.9
        self.gamme = 0.9
        self.memory = deque(maxlen=MAX_MEM)
        self.model = None
        self.trainer = None
       

    def get_state(self,game):
        # state = [ds, dr,dl,movedir,fl,fr,fu,fd] 12 tuple)
        head = game.head
        food = game.food
        direction = game.direction

        point_l = Point(head.x-BLOCK_SIZE,head.y)
        point_r = Point(head.x +BLOCK_SIZE, head.y)
        point_u = Point(head.x,head.y-BLOCK_SIZE)
        point_d = Point(head.x, head.y + BLOCK_SIZE)

        dir_l = (direction == Direction.LEFT)
        dir_r = (direction == Direction.RIGHT)
        dir_u = (direction == Direction.LEFT)
        dir_d = (direction == Direction.LEFT)
        state = [
            #STRAIGHT
                (direction  == Direction.LEFT and game.is_collision(point_l)) or
                (direction == Direction.RIGHT and game.is_collision(point_r)) or
                (direction == Direction.UP and game.is_collision(point_u)) or 
                (direction == Direction.DOWN and game.is_collision(point_d)),
            #RIGHT
                (direction  == Direction.LEFT and game.is_collision(point_u)) or
                (direction == Direction.RIGHT and game.is_collision(point_d)) or
                (direction == Direction.UP and game.is_collision(point_r)) or 
                (direction == Direction.DOWN and game.is_collision(point_l)),
            #LEFT
                (direction  == Direction.LEFT and game.is_collision(point_d)) or
                (direction == Direction.RIGHT and game.is_collision(point_u)) or
                (direction == Direction.UP and game.is_collision(point_l)) or 
                (direction == Direction.DOWN and game.is_collision(point_r)),

                dir_l,
                dir_r,
                dir_u,
                dir_d,
                

                (head.x > food.x),
                (head.x < food.x),
                (head.y < food.y),
                (head.y > food.y)
                
                ]
        return state

    def remember(self, state, action, reward, next_state, done):
        # append to deque
        self.memory.append((state, action, reward, next_state, done))
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
    scores  = []
    mean_scores = []
    final_scores = []
    total_score = 0
    best_score = 0

    agent = Agent()
    game = SnakeGame()

    while True:
        old_state = agent.get_state(game= game)
        action = agent.get_action(state = old_state)
        reward, done, score = game.play_step(action)
        new_state = agent.get_state(game = game)

        agent.train_short_memory(old_state,action,reward,new_state,done)

        agent.remember(old_state,action,reward,new_state,done)

        # scores.append(score)
        if done:
            game.start()
            agent.n_games +=1
            agent.train_long_memory()
            scores.append(score)
            total_score = score

            if total_score > best_score:
                best_score = total_score
        print(f"{agent.n_games}, {score}, {best_score}")
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