import torch
import numpy as np
import random
from collections import deque
from game import SnakeGame, Direction, Point, BLOCK_SIZE
from model import Linear_QNet,QTrainer
from utils import plot


MAX_MEM = 100_000
BATCH_SIZE = 1000
LR = 0.001



class Agent:
    def __init__(self):
        #store more paramters like num games, epsilon for randomness
        #gamma memory(use deque), model and trainer
        self.n_games = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEM)
        self.model = Linear_QNet(11,256,3)
        self.trainer = QTrainer(self.model,LR,self.gamma)
       

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
        return np.array(state,dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # append to deque
        self.memory.append((state, action, reward, next_state, done))
        

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_batch = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_batch = self.memory
        states, actions, rewards, next_states, dones = zip(*mini_batch)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 -self.n_games
        final_move = [0,0,0]
        if random.randint(0,200) < self.epsilon:
            idx = random.randint(0,2)
            final_move[idx] = 1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            idx = torch.argmax(self.model(state0)).item()
            final_move[idx] = 1

        return final_move

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

            if score > best_score:
                best_score = score
            
            print(f"{agent.n_games}, {score}, {best_score}")

            scores.append(score)
            total_score += score 
            mean_score = total_score/agent.n_games
            mean_scores.append(mean_score)

            plot(scores=scores,mean_scores=mean_scores)
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