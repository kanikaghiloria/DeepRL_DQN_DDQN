'''
This program implements a Q-Learning Agent.
'''

import os, sys
sys.path.append('../game')
sys.path.append('../utils')

from collections import defaultdict
import json
import random
import numpy as np

import gym
from agents.TemplateAgent import FlappyBirdAgent
from game.FlappyBirdGame import FlappyBirdNormal

import warnings
warnings.filterwarnings('ignore')
        
# a_file = open("logs/scores/a.txt", 'a+')
a_file = open("D:\\RL\\FlapAI-Bird\\logs\\scores\\scores_file.txt", 'w')
h_file = open("D:\\RL\\FlapAI-Bird\\logs\\scores\\test.txt", 'w')
class QLearningAgent(FlappyBirdAgent):
    ''' Q-Learning Agent. '''
    
    def __init__(self, actions, probFlap = 0.5, rounding = None):
        '''
        Initializes the agent.
        
        Args:
            actions (list): Possible action values.
            probFlap (float): The probability of flapping when choosing
                              the next action randomly.
            rounding (int): The level of discretization.
        '''
        super().__init__(actions)
        self.probFlap = probFlap
        self.qValues = defaultdict(float)
        self.env = FlappyBirdNormal(gym.make('FlappyBird-v0'), rounding = rounding)

    def act(self, state):
        '''
        Returns the next action for the current state.
        
        Args:
            state (str): The current state.
            
        Returns:
            int: 0 or 1.
        '''
        def randomAct():
            if random.random() < self.probFlap:
                return 0
            return 1
            
        if random.random() < self.epsilon:
            return randomAct()
            
        qValues = [self.qValues.get((state, action), 0) for action in self.actions]
            
        if qValues[0] < qValues[1]:
            return 1
        elif qValues[0] > qValues[1]:
            return 0
        else:
            return randomAct()
            
    def saveQValues(self):
        ''' Saves the Q-values. '''
        toSave = {key[0] + ' action ' + str(key[1]) : self.qValues[key] for key in self.qValues}
        # with open('logs/qValues.json', 'w') as fp:
        with open('D:\\RL\\FlapAI-Bird\\logs\\scores\\qValues.json', 'w') as fp:
            json.dump(toSave, fp)
            fp.writelines("\n")
            
    def loadQValues(self):
        ''' Loads the Q-values. '''
        def parseKey(key):
            state = key[:-9]
            action = int(key[-1])
            return (state, action)

        with open('qValues.json') as fp:
            toLoad = json.load(fp)
            self.qValues = {parseKey(key) : toLoad[key] for key in toLoad}

    def train(self, order = 'forward', numIters = 20000, epsilon = 0.1, discount = 1,
              eta = 0.9, epsilonDecay = False, etaDecay = False, evalPerIters = 250,
              numItersEval = 1000):
        '''
        Trains the agent.
        
        Args:
            order (str): The order of updates, 'forward' or 'backward'.
            numIters (int): The number of training iterations.
            epsilon (float): The epsilon value.
            discount (float): The discount factor.
            eta (float): The eta value.
            epsilonDecay (bool): Whether to use epsilon decay.
            etaDecay (bool): Whether to use eta decay.
            evalPerIters (int): The number of iterations between two evaluation calls.
            numItersEval (int): The number of evaluation iterations.
        '''
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
        timestep=0
        
        for i in range(numIters):
            # if i % 50 == 0 or i == numIters - 1:
            #     print("Iter: ", i)
            print("Iter: ", i, " // timestep: ", timestep)
            # i = 4
            self.epsilon = self.initialEpsilon / (i + 1) if self.epsilonDecay \
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
                self.env.render()  # Uncomment it to display graphics.
                totalReward += reward
                timestep = timestep + 1
                if reward >= 1:
                    score += 1
                if done:
                    break
            
            if score > maxScore: maxScore = score
            if totalReward > maxReward: maxReward = totalReward
            
            if order == 'forward':
                for (state, action, reward, nextState) in gameIter:
                    self.updateQ(state, action, reward, nextState)
            else:
                for (state, action, reward, nextState) in gameIter[::-1]:
                    self.updateQ(state, action, reward, nextState)
                
            if self.etaDecay:
                self.eta *= (i + 1) / (i + 2)
            
            if (i + 1) % self.evalPerIters == 0:
                # print("enter here")
                with open('D:\\RL\\FlapAI-Bird\\logs\\scores\\test.txt', 'a+') as fp:
                    s = "Iter: ", str(i+1), " // Timestep: ", str(timestep)," // "
                    fp.writelines(s)
                    # fp.writelines("\n")
                output = self.test(numIters = self.numItersEval)
                self.saveOutput(output, i + 1, timestep)
                self.saveQValues()
                
        self.env.close()
        print("Max Score Train: ", maxScore)
        print("Max Reward Train: ", maxReward)
        print()
    
    def test(self, numIters = 20000):
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

            print ("i test: ", i)
            while True:
                action = self.act(state)
                state, reward, done, _ = self.env.step(action)
                self.env.render()  # Uncomment it to display graphics.
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
        with open('D:\\RL\\FlapAI-Bird\\logs\\scores\\test.txt', 'a') as fp:
            s = "Max Score Test: ", str(maxScore)," // Max Reward Test: ", str(maxReward)
            fp.writelines(s)
            fp.writelines("\n")
        return output
            
    def updateQ(self, state, action, reward, nextState):
        '''
        Updates the Q-values based on an observation.
        
        Args:
            state, nextState (str): Two states.
            action (int): 0 or 1.
            reward (int): The reward value.
        '''
        nextQValues = [self.qValues.get((nextState, nextAction), 0) for nextAction in self.actions]
        nextValue = max(nextQValues)
        self.qValues[(state, action)] = (1 - self.eta) * self.qValues.get((state, action), 0) \
                                        + self.eta * (reward + self.discount * nextValue)
        
    def saveOutput(self, output, iter, timestep):
        '''
        Saves the scores.
        
        Args:
            output (dict): A set of scores.
            iter (int): Current iteration.
        '''
        # if not os.path.isdir('logs/scores'):
        #     os.mkdir('logs/scores')
        # with open('./logs/scores/scores_{}.json'.format(iter), 'w') as fp:
        #     json.dump(output, fp)
        with open('D:\\RL\\FlapAI-Bird\\logs\\scores\\scores_{}.json'.format(iter), 'w') as fp:
            json.dump(output, fp)
            fp.writelines("\n")
        outStatement = "ITER: ", str(iter)," // TIMESTEP: ", str(timestep)," // output: ",str(output), "\n"

        # with open('/logs/scores/scores_file', 'a') as fout:
        with open("D:\\RL\\FlapAI-Bird\\logs\\scores\\scores_file.txt", 'a') as fout:
            fout.writelines(outStatement)

def main():
    agent = QLearningAgent(actions=[0, 1], rounding=10, probFlap=0.1)
    agent.train(order='backward', eta=0.8, numIters = 20000, evalPerIters=250, numItersEval = 1000)
    agent.saveQValues()

if __name__ == '__main__':
    main()
