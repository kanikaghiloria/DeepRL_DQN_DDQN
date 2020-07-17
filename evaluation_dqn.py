#!/usr/bin/env python
from __future__ import print_function

# import tensorflow as tf
from statistics import mean

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import cv2
import sys
import atexit

from game import wrapped_flappy_bird

sys.path.append("game/")
import game.wrapped_flappy_bird as game
import random
import numpy as np
from collections import deque

outStatement = ''

GAME = 'bird' # the name of the game being played for log files
ACTIONS = 2 # number of valid actions
GAMMA = 0.99 # decay rate of past observations
DIFFICULTY = 'general'
AGENT = 'DQN'
NUMEBEROFGAMES = 100

############################### UNCOMMENT THIS SECTION FOR TESTING
outputFile = "/output_test.txt"
# a_file = open("logs_" + GAME + "/" + AGENT + "/" + DIFFICULTY + "/readout_test.txt", 'a+')
# h_file = open("logs_" + GAME + "/" + AGENT + "/" + DIFFICULTY + "/hidden_test.txt", 'a+')
out_file = open("logs_" + GAME + "/" + AGENT + "/" + DIFFICULTY + "/output_test.txt", 'a+')
###############################

FRAME_PER_ACTION = 1

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.01, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W, stride):
    return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")

def createNetwork():
    # network weights
    W_conv1 = weight_variable([8, 8, 4, 32])
    b_conv1 = bias_variable([32])

    W_conv2 = weight_variable([4, 4, 32, 64])
    b_conv2 = bias_variable([64])

    W_conv3 = weight_variable([3, 3, 64, 64])
    b_conv3 = bias_variable([64])

    W_fc1 = weight_variable([1600, 512])
    b_fc1 = bias_variable([512])

    W_fc2 = weight_variable([512, ACTIONS])
    b_fc2 = bias_variable([ACTIONS])

    # input layer
    s = tf.placeholder("float", [None, 80, 80, 4])

    # hidden layers
    h_conv1 = tf.nn.relu(conv2d(s, W_conv1, 4) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)

    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2, 2) + b_conv2)
    #h_pool2 = max_pool_2x2(h_conv2)

    h_conv3 = tf.nn.relu(conv2d(h_conv2, W_conv3, 1) + b_conv3)
    #h_pool3 = max_pool_2x2(h_conv3)

    #h_pool3_flat = tf.reshape(h_pool3, [-1, 256])
    h_conv3_flat = tf.reshape(h_conv3, [-1, 1600])

    h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat, W_fc1) + b_fc1)

    # readout layer
    readout = tf.matmul(h_fc1, W_fc2) + b_fc2

    return s, readout, h_fc1

def testNetwork(s, readout, h_fc1, sess):
    # define the cost function

    with open("logs_" + GAME + "/" + AGENT + "/" + DIFFICULTY + outputFile, 'a') as fout:
        fout.writelines("================================= NEW EVALUATION =================================")
        fout.writelines("\n")

    a = tf.placeholder("float", [None, ACTIONS])
    y = tf.placeholder("float", [None])
    readout_action = tf.reduce_sum(tf.multiply(readout, a), reduction_indices=1)
    cost = tf.reduce_mean(tf.square(y - readout_action))
    train_step = tf.train.AdamOptimizer(1e-6).minimize(cost)

    # open up a game state to communicate with emulator
    game_state = game.GameState(difficulty=DIFFICULTY)

    do_nothing = np.zeros(ACTIONS)
    do_nothing[0] = 1
    x_t, r_0, terminal, s1, s2 = game_state.frame_step(do_nothing)
    x_t = cv2.cvtColor(cv2.resize(x_t, (80, 80)), cv2.COLOR_BGR2GRAY)
    ret, x_t = cv2.threshold(x_t,1,255,cv2.THRESH_BINARY)
    s_t = np.stack((x_t, x_t, x_t, x_t), axis=2)

    # saving and loading networks
    saver = tf.train.Saver()
    sess.run(tf.initialize_all_variables())
    checkpoint = tf.train.get_checkpoint_state("saved_networks/" + AGENT + "/" + DIFFICULTY + "/")


    ############################### UNCOMMENT THIS SECTION FOR TESTING
    if checkpoint and checkpoint.model_checkpoint_path:
        saver.restore(sess, checkpoint.model_checkpoint_path)
        print("Successfully loaded:", checkpoint.model_checkpoint_path)
    else:
        print("Could not find old network weights")

    t = 0
    scores = []
    currentGame = 1
    # while "flappy bird" != "angry bird":
    while currentGame <= NUMEBEROFGAMES:
        # choose an action epsilon greedily
        readout_t = readout.eval(feed_dict={s : [s_t]})[0]
        a_t = np.zeros([ACTIONS])
        action_index = 0
        if t % FRAME_PER_ACTION == 0:
            action_index = np.argmax(readout_t)
            a_t[action_index] = 1
        else:
            a_t[0] = 1 # do nothing
        # run the selected action and observe next state and reward
        # x_t1_colored, r_t, terminal = game_state.frame_step(a_t)
        x_t1_colored, r_t, terminal, score, final_score = game_state.frame_step(a_t)
        # if(score > 0):
        #     print (score)
        x_t1 = cv2.cvtColor(cv2.resize(x_t1_colored, (80, 80)), cv2.COLOR_BGR2GRAY)
        ret, x_t1 = cv2.threshold(x_t1, 1, 255, cv2.THRESH_BINARY)
        x_t1 = np.reshape(x_t1, (80, 80, 1))
        #s_t1 = np.append(x_t1, s_t[:,:,1:], axis = 2)
        s_t1 = np.append(x_t1, s_t[:, :, :3], axis=2)

        # update the old values
        s_t = s_t1
        t += 1

        # print info
        state = "test"
        print("TIMESTEP: ", t, "/ GAME: ", currentGame, "/ STATE: ", state, "/ ACTION: ", action_index, "/ SCORE: ", score)
        # write info to files

        if terminal:
            outStatement = "TIMESTEP: ", str(t), "/ GAME: ", str(currentGame), "/ ACTION: ", \
                           str(action_index), "/ FINAL SCORE: ", str(final_score), "\n"
            with open("logs_" + GAME + "/" + AGENT + "/" + DIFFICULTY + outputFile, 'a') as fout:
                fout.writelines(outStatement)
            scores.append(final_score)
            currentGame = currentGame + 1

    maxScore = max(scores)
    minScore = min(scores)
    meanScore = mean(scores)
    lastOutStatement = "Total TIMESTEP: ", str(t), "/ total Games: ", str(NUMEBEROFGAMES), "/ Maximum Score: ", \
                       str(maxScore), "/ Minimum Score: ", str(minScore), "/ Average Score: ", str(meanScore), "\n"
    with open("logs_" + GAME + "/" + AGENT + "/" + DIFFICULTY + outputFile, 'a') as fout:
        fout.writelines(lastOutStatement)



def playGame():
    sess = tf.InteractiveSession()
    s, readout, h_fc1 = createNetwork()
    testNetwork(s, readout, h_fc1, sess)

def printLine():
    print ("outStatement: ", outStatement)

def main():
    # try:
    playGame()
    atexit.register(printLine)


if __name__ == "__main__":
    main()

