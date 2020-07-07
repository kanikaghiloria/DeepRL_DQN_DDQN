from __future__ import print_function

# import tensorflow as tf
import tensorflow.compat.v1 as tf
from gym.wrappers import Monitor

tf.disable_v2_behavior()
import cv2
import sys
import math
import os

import atexit

from game import wrapped_flappy_bird

sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque


from collections import defaultdict
import json
import random
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

import gym
GAME = 'bird'

class FlappyBirdQL(gym.Wrapper):
    '''Game Environment fo Q-Learning'''

    def __init__(self, env, rounding = 0):
        super.__init__(env)
        self.rounding = rounding

    def save_output(self, outputdir = None):
        if outputdir:
            self.env = Monitor(self.env, directory = outputdir, force = True)

    def step(self, action):
        '''Enables the agent to take the action and observe next state and reward'''
        _, reward, terminal, _ = self.env.step(action)
        state = self.getGameState()
        if not terminal:
            reward = reward + 0.5
        else:
            reward = reward - 1000
        if reward >=1:
            reward = 5
        return state, reward, terminal, {}

    def getGameState(self):
        '''Returns current game state'''
        gameState = self.env.game_state.getGameState()
        hor_dist_to_next_pipe = gameState['next_pipe_dist_to_player']
        ver_dist_to_next_pipe = gameState['next_pipe_bottom_y'] - gameState['player_y']
        if self.rounding != 0:
            hor_dist_to_next_pipe = discretize(hor_dist_to_next_pipe, self.rounding)
            ver_dist_to_next_pipe = discretize(ver_dist_to_next_pipe, self.rounding)

        state = []
        state.append('player_vel' + ' ' + str(gameState['player_vel']))
        state.append('hor_dist_to_next_pipe' + ' ' + str(hor_dist_to_next_pipe))
        state.append('ver_dist_to_next_pipe' + ' ' + str(ver_dist_to_next_pipe))
        return ' '.join(state)


class Transform(object):
    ''' A class that preprocesses the images of the game screen. '''

    def __init__(self):
        ''' Initializes the class. '''
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transform = transforms.Compose([
            transforms.Lambda(lambda x: x[:404]),
            transforms.ToPILImage(),
            transforms.Grayscale(),
            transforms.Resize((80, 80)),
            transforms.Lambda(lambda x:
                              cv2.threshold(np.array(x), 128, 255, cv2.THRESH_BINARY)[1]),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.unsqueeze(0).to(self.device)),
        ])

    def process(self, img):
        '''
        Transforms the input image.

        Args:
            img (ndarray): Imput image.

        Returns:
            Tensor: A transformed image.
        '''
        return self.transform(img)


class FlappyBirdAgent(object):
    ''' Template Agent. '''

    def __init__(self, actions):
        '''
        Initializes the agent.

        Args:
            actions (list): Possible action values.
        '''
        self.actions = actions

    def act(self, state):
        '''
        Returns the next action for the current state.

        Args:
            state (list): The current state.
        '''
        raise NotImplementedError("Override this.")

    def train(self, numIters):
        '''
        Trains the agent.

        Args:
            numIters (int): The number of training iterations.
        '''
        raise NotImplementedError("Override this.")

    def test(self, numIters):
        '''
        Evaluates the agent.

        Args:
            numIters (int): The number of evaluation iterations.
        '''
        raise NotImplementedError("Override this.")

    def saveOutput(self):
        ''' Saves the scores. '''
        raise NotImplementedError("Override this.")

class QLearning(FlappyBirdAgent):
    def __init(self, actions, probFlap = 0.5, rounding = 10):
        super().__init__(actions)
        self.probFlap = probFlap
        self.q_values = defaultdict(float)
        self.env = FlappyBirdQL(gym.make('FlappyBird-v0'), rounding = rounding)

    def act(self, state):
        def randomAct():
            if random.random() < self.probFlap:
                return 0
            return 1
        if random.random() < self.epsilon:
            return randomAct()

        q_values = [self.q_values.get((state, a), 0) for a in self.actions]
        if q_values[0] < q_values[1]:
            return 1
        if q_values[0] > q_values[1]:
            return 0
        else:
            return randomAct()

    def saveQValues(self):
        save = {key[0] + ' action ' + str(key[1]) : self.q_values[key] for key in self.q_values}
        with open ("logs_" + GAME +'/QLearning/qValues.json', 'w') as fp:
            json.dump(save, fp)

    def loadQValues(self):
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)
        with open("logs_" + GAME +'/QLearning/qValues.json') as f:
            load = json.load(f)
            self.q_values = {parseKey(key): load[key] for key in load}

    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1, eta = 0.9, epsilonDecay = False,
              etaDecay = False, evalPerIters = 250, numItersEval = 1000):
        self.epsilon = epsilon
        self.initialEpsilon = epsilon
        self.discount = discount
        self.eta = eta
        self.epsilonDecay = epsilonDecay
        self.etaDecay = etaDecay
        self.evalPerIters = evalPerIters
        self.numItersEval = numItersEval
        self.env.seed(random.randint(0, 100))

        done = False
        maxScore = 0
        maxReward = 0

        ctr = 0

        while "flappy_bird" != "angry_bird":
            if ctr % 50 == 0 or ctr == numIters - 1:
                print("Iteration: ", ctr)
            self.epsilon = self.initialEpsilon / (ctr + 1) if self.epsilonDecay \
                else self.initialEpsilon

            score = 0
            totalReward = 0
            ob = self.env.reset()
            gameIter = []
            state = self.env.getGameState()

            while True:
                action = self.act(state)
                nextState, reward, done, _ = self.env.step(action)
                gameIter.append((state, action, reward, nextState))
                state = nextState

                self.env.render()

                totalReward += reward
                if reward >= 1:
                    score += 1
                if done:
                    break

            if score > maxScore:
                maxScore = score
            if totalReward > maxReward:
                maxReward = totalReward

            if (order == 'forward'):
                for (state, action, reward, nextState) in gameIter:
                    self.updateQ(state, action, reward, nextState)
            else:
                for (state, action, reward, nextState) in gameIter[::-1]:
                    self.updateQ(state, action, reward, nextState)
            if self.etaDecay:
                self.eta *= (ctr + 1) / (ctr + 2)

            if (ctr + 1) % self.evalPerIters == 0:
                output = self.test(numIters=self.numItersEval)
                self.saveOutput(output, ctr + 1)
                self.saveQValues()
            ctr = ctr +1
        self.env.close()
        print("Max Score Train: ", maxScore)
        print("Max Reward Train: ", maxReward)
        print()

    def test(self, numIters=20000):
            '''
            Evaluates the agent.

            Args:
                numIters (int): The number of evaluation iterations.

            Returns:
                dict: A set of scores.
            '''
            self.epsilon = 0
            self.env.seed(0)
            done = False
            maxScore = 0
            maxReward = 0
            output = defaultdict(int)

            for i in range(numIters):
                score = 0
                totalReward = 0
                ob = self.env.reset()
                state = self.env.getGameState()

                while True:
                    action = self.act(state)
                    state, reward, done, _ = self.env.step(action)
                    #                self.env.render()  # Uncomment it to display graphics.
                    totalReward += reward
                    if reward >= 1:
                        score += 1
                    if done:
                        break

                output[score] += 1
                if score > maxScore: maxScore = score
                if totalReward > maxReward: maxReward = totalReward

            self.env.close()
            print("Max Score Test: ", maxScore)
            print("Max Reward Test: ", maxReward)
            print()
            return output

    def updateQ(self, state, action, reward, nextState):
        nextQValues = [self.q_values.get((nextState, nextAction), 0) for nextAction in self.actions]
        nextValue = max (nextQValues)
        self.q_values[(state, action)] = (1 - self.eta) * self.q_values.get((state, action), 0) \
                                         + self.eta * (reward + self.discount * nextValue)

    def saveOutput(self, output, iter):
        if not os.path.isdir("logs_" + GAME +'/QLearning/scores'):
            os.mkdir("logs_" + GAME +'/QLearning/scores')
        with open("logs_" + GAME +'/QLearning/scores/scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)

def discretize(num, rounding):
    '''
    Discretizes the input num base on the value rounding.

    Args:
        num (int): An input value.
        rounding (int): The level of discretization.

    Returns:
        int: A discretized output value.
    '''
    return rounding * math.floor(num / rounding)


def dotProduct(v1, v2):
    '''
    Computes the dot product between two feature vectors v1 and v2.

    Args:
        v1, v2 (dict): Two input vectors.

    Returns:
        dict: A dot product.
    '''
    if len(v1) < len(v2):
        return dotProduct(v2, v1)
    return sum(v1.get(key, 0) * value for key, value in v2.items())

def increment(v1, scale, v2):
    '''
    Executes v1 += scale * v2 for feature vectors.

    Args:
        v1, v2 (dict): Two input vectors.
        scale (float): A scale value.
    '''
    for key, value in v2.items():
        v1[key] = v1.get(key, 0) + value * scale

outStatement = ''

# GAME = 'bird' # the name of the game being played for log files
# ACTIONS = 2 # number of valid actions
# GAMMA = 0.99 # decay rate of past observations

def main():
    r = 10
    agent = QLearning(actions=[0,1])
    agent.train(order = 'backward', numIters = 20000, epsilon = 0.1, discount = 1, eta = 0.8, epsilonDecay = False,
              etaDecay = False, evalPerIters = 250, numItersEval = 1000)
    agent.saveQValues()

if __name__ == "__main__":
    main()