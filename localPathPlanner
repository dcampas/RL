import numpy as np
import math
import gym
import random
import time
import os
import pygame
from operator import add, sub

random.seed(1)

#Possible ways to determine and gauge risks:
# 1. Use distance between the drone and the risk
# 2. Use the velocity of the risk relative to the drone and/or the drone relative to the risk
# 3. Predetermine risks by using the time of day
# 4. Use the number of risks of one type present
# 5. Use the elevation/height of the risks
#Possible types of risks:
# 1. Cars
# 2. People/pets
# 3. Buildings
# 4. Birds
# 5. Water
# 6. Wind (speed and direction)

##############################FUNCTIONS##############################

#Get the number of rows and columns of the environment
def stSpaceShp(env):
    stateSpaceShape = np.shape(env)
    return stateSpaceShape

#Obtain the possible number of states
def stSpaceSize(stateSpaceShape):
    stateSpaceSize = stateSpaceShape[0] * stateSpaceShape[1]
    return stateSpaceSize

#Create a Q-Table with all values initialized to 0
def qTable(stateSpaceSize, actionSpaceSize):
    q_table = np.zeros((stateSpaceSize, actionSpaceSize))
    return q_table

#Create a function that converts a [row,col] state tuple into a single integer that defines the state
#Use this formula to convert from non-zero [row,col] state to 0-based single integer:(row - 1) * width) + (col - 1)
def stateNum(state, gridWidth):
    row = state[0]
    col = state[1]
    stateNum = ((row) * gridWidth) + (col)
    return stateNum

#Create a function that updates the position of the drone based on its action (for use after training)
def updatePos(action,state,stateSpaceShape):
    if action == 0 and (state[0] != 0):
        state[0] -= 1
    elif action == 1 and (state[1] != stateSpaceShape[1]-1):
        state[1] += 1
    elif action == 2 and (state[0] != stateSpaceShape[0]-1):
        state[0] += 1
    elif action == 3 and (state[1] != 0):
        state[1] -= 1
    return state

#Create a function that updates the position of the drone and penalizes the drone for not moving (for use during training)
def updatePosReward(action,state,stateSpaceShape):
    if (action == 0) and (state[0] != 0):
        state[0] -= 1
        posReward = 0
    elif (action == 1) and (state[1] != stateSpaceShape[1]-1):
        state[1] += 1
        posReward = 0
    elif (action == 2) and (state[0] != stateSpaceShape[0]-1):
        state[0] += 1
        posReward = 0
    elif (action == 3) and (state[1] != 0):
        state[1] -= 1
        posReward = 0
    else:
        posReward = -1
    return state, posReward

#Create a function that calculates the distance between the drone and the goal and assigns the corresponding reward; SEEMS LIKE THIS IS NOT NEEDED
def getDistReward(state,stateSpaceShape,goal):
    coordinateDist = [goal[1] - state[1], goal[0] - state[0]]
    dist = math.sqrt((coordinateDist[0])**2 + (coordinateDist[1])**2)

    if dist != 0:
        distReward = -(dist * 0.001)
    else:
        distReward = 1
    
    return distReward

    #^Model distance as linear or exponential function

#Create a function that calculates the drone's reward based on the risk level of the area it moves into
def getRiskReward(env,state):
    if env[state[0], state[1]] == 1:
        riskReward = -1
    else:
        riskReward = 0

    return riskReward

#Write a function that counts the number of steps from start of episode to end of episode; it will use this number as a multiplier to decrease the total rewards as more steps are needed to reach the goal
def timeReward(steps, rewardsCurrentState):
    if done == False:
        rewardsCurrentState -= (steps * 0.2)
    return rewardsCurrentState

#Create a function that gives a reward if the goal has been reached during training
def getFinishReward(env,state,goal):
    if (state[0] == goal[0]) and (state[1] == goal[1]):
        finishReward = 1
    else:
        finishReward = 0
    return finishReward

#Create a function that ends the Pygame simulation if the goal has been reached
def episodeEnd(state,done,goal,goalReached):
    if (state[0] == goal[0]) and (state[1] == goal[1]):
        done = True
        goalReached += 1
    return done, goalReached

#Create a function that obtains the action that yields the 2nd highest Q-Value of a Q-Table; this will be used to select an alternate path in case of back-and-forth movement
def altPath(stateNum,qTable):
    altPathOptionsDictionary = {qTable[stateNum,0]:0, qTable[stateNum,1]:1, qTable[stateNum,2]:2, qTable[stateNum,3]:3}
    altPathOptionsQValues = np.array([qTable[stateNum, 0], qTable[stateNum, 1], qTable[stateNum, 2], qTable[stateNum, 3]])
    altPathOptionsQValues = sorted(altPathOptionsQValues, reverse=True)
    action = altPathOptionsQValues[1]
    action = altPathOptionsDictionary.get(action)
    return action

#Create a function that generates a Pygame environment representative of the numpy array used for training
def createEnv(envA,envB,envC,envD,rows,cols,screen,gridLocationSize):
    for row in range(rows):
        for col in range(cols):
            if envA[row,col] == 0 and envB[row,col] == 0 and envC[row,col] == 0 and envD[row,col] == 0:
                pygame.draw.rect(screen, white, (col * gridLocationSize, row * gridLocationSize, gridLocationSize, gridLocationSize))
            elif envA[row,col] == 1:
                pygame.draw.rect(screen, yellow, (col * gridLocationSize, row * gridLocationSize, gridLocationSize, gridLocationSize))
            elif envB[row,col] == 1:
                pygame.draw.rect(screen, orange, (col * gridLocationSize, row * gridLocationSize, gridLocationSize, gridLocationSize))   
            elif envC[row,col] == 1:
                pygame.draw.rect(screen, blue, (col * gridLocationSize, row * gridLocationSize, gridLocationSize, gridLocationSize))        
            elif envD[row,col] == 1:
                pygame.draw.rect(screen, red, (col * gridLocationSize, row * gridLocationSize, gridLocationSize, gridLocationSize))

#Create a function that calculates the weight of each risk based on how prevalent it is
def getWeights(envA,envB,envC,envD,rows,cols):
    envs = [envA,envB,envC,envD]
    weights = []
    envNum = 0
    for env in envs:
        weights.append(0)
        for row in range(rows):
            for col in range(cols):
                if env[row,col] == 1:
                    weights[envNum] += 1
        envNum += 1
    return weights

#Create a function that assigns weights to each risk, combines the Q-Tables, and selects a weighted path to follow
def weightedTable(qTableA,qTableB,qTableC,qTableD,weightA,weightB,weightC,stSpaceSizeA,actionSpaceSize):
    qTable = np.zeros((stSpaceSizeA, actionSpaceSize))
    qTableA = qTableA * weightA
    qTableB = qTableB * weightB
    qTableC = qTableC * weightC
    qTableD = qTableD * weightD
    qTable += qTableA
    qTable += qTableB
    qTable += qTableC
    qTable += qTableD
    return qTable

##############################ENVIRONMENTS##############################

envA = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,1,0,1,0,0,0,0,0],[0,0,0,1,1,1,1,0,0,0],[0,0,0,1,1,1,1,0,0,0],[0,0,0,1,1,1,1,1,0,0],[0,0,0,1,1,1,1,0,0,0],[0,0,0,1,1,1,0,0,0,0],[0,0,1,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
envB = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,1,1,1,1],[0,0,0,0,0,1,1,1,1,1],[0,0,0,0,0,1,1,1,1,1]])
envC = np.array([[1,1,1,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
envD = np.array([[0,0,0,0,0,0,1,1,1,0],[0,0,0,0,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# envA = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,0,1,1],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# envB = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0],[0,0,0,1,1,0,0,0,0,0]])
# envC = np.array([[0,1,1,0,0,1,1,1,0,0],[0,1,1,0,0,1,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,1,1,0,0],[0,0,0,0,0,0,0,1,1,1],[0,0,0,0,0,0,0,1,1,1]])
# envD = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,1,0,0,0,0,0,0,0,0],[0,1,1,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# envA = np.array([[1,1,1,1,1,1,1,1,1,0],[0,0,0,0,0,0,1,1,1,0],[0,0,0,0,0,0,1,1,1,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# envB = np.array([[0,0,0,0,0,0,0,0,0,0],[1,1,1,1,1,1,0,0,0,0],[0,0,0,0,0,1,0,0,0,0],[0,0,0,1,0,1,1,1,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,1,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# envC = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[1,1,0,1,0,0,0,0,0,0],[1,1,0,1,0,0,0,0,0,0],[1,1,0,1,0,0,0,0,0,0],[1,1,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0]])
# envD = np.array([[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,1],[0,0,0,0,0,0,0,0,0,0],[0,0,0,0,0,1,1,1,1,1],[0,0,0,0,0,1,1,1,1,1]])

##############################CONSTANTS##############################

goal = [0,9] #set the desired goal for all environments

#Number of actions is 4 (up(0), right(1), down(2), left(3))
actionSpaceSize = 4

#Set the desired number of episodes and the maximum number of steps per episode
numEpisodes = 100000
maxStepsPerEpisode = 200

#Set the desired learning rate and discount rate
learningRate = 0.0025
discountRate = 0.9 #higher value means that the agent prioritizes immediate rewards over long-term rewards that it can expect to receive

#Set the desired exploration rate, maximum exploration rate, minimum exploration rate, and exploration decay rate
explorationRate = 1
maxExplorationRate = 1
minExplorationRate = 0.01
explorationDecayRate = 0.00025

#Initialize Pygame before starting an environment simulation
pygame.init()

#Create RGB tuples to represent different colors
red = (255,0,0)
orange = (255,165,0)
yellow = (255,255,0)
greenyellow = (173,255,47)
green = (0,255,0)
blue = (0,0,255)
white = (255,255,255)
black = (0,0,0)

##############################TRAINING LOOP A##############################

#Initialize an array that will contain and separate the rewards gained in each episode
rewardsAllEpisodes = []

#Create a variable to keep track of how many times the goal is reached
goalReached = 0

stSpaceShpA = stSpaceShp(envA)
stSpaceSizeA = stSpaceSize(stSpaceShpA)
rowsA = stSpaceShpA[0]
colsA = stSpaceShpA[1]
qTableA = qTable(stSpaceSizeA, actionSpaceSize)

for episode in range(numEpisodes):

    state = [stSpaceShpA[0]-1,0] #set the initial state
    gridWidth = stSpaceShpA[1]
    stateNum = ((state[0]) * gridWidth) + (state[1])
    #print('initial state: ', state)
    #print('initial state in integer form: ', stateNum)
    steps = 0

    done = False
    rewardsCurrentEpisode = 0

    #Dictate how the program selects an action and use the Bellman equation and a weighted sum of the previous Q value to update the Q-Table after an action is taken
    #This loop is executed until the maximum number of steps is reached or until the 'agent' reaches the goal
    for step in range(maxStepsPerEpisode):

        steps += 1

        rewardsCurrentState = 0

        #Generate a random number used to decide if the drone will explore or exploit the environment
        explorationRateThreshold = random.uniform(0, 1)

        #Take an action according to the Q-table to exploit the environment
        if explorationRateThreshold > explorationRate:
            action = np.argmax(qTableA[stateNum, :])

        #Take a random action to explore the environment
        else:
            action = random.choice(range(0,actionSpaceSize))
        
        #print('action: ', action)

        #Generate variables that represent the state prior to implementing the selected action; these will be used to update the Q-Table
        prevState = state
        prevstateNum = ((state[0]) * gridWidth) + (state[1])

        #Change the position of the state according to the action selected; adjust the reward accordingly
        state, posReward = updatePosReward(action,state,stSpaceShpA)
        #print('state due to action: ', state)
        rewardsCurrentState += posReward

        #Check the distance from the current position to the goal; adjust the reward accordingly
        distReward = getDistReward(state,stSpaceShpA,goal)
        rewardsCurrentState += distReward

        #Check the risk value of the position of the drone; adjust the reward accordingly using this value and distReward
        #print('risk value of current state: ', envA[state[0]-1, state[1]-1])
        riskReward = getRiskReward(envA,state)
        rewardsCurrentState += riskReward

        #Keep track of when the goal has been reached and implement a reward penalty for every step that does not reach the goal
        finishReward = getFinishReward(envA, state, goal)
        rewardsCurrentState += finishReward

        #Use the [row,col] state to create a state that is represented by a single integer
        stateNum = ((state[0]) * gridWidth) + (state[1])
        #print('state in integer form: ', stateNum)

        #Update the Q-table with all of the rewards obtained from the selected action
        qTableA[prevstateNum, action] = (qTableA[prevstateNum, action] * (1 - learningRate)) + (learningRate * (rewardsCurrentState + (discountRate * np.max(qTableA[stateNum, :]))))

        #^Use a 3D array or a dictionary to be able to use the [row,col] state variable

        #print('total reward obtained from this action: ', rewardsCurrentState)

        #Add the rewards from the selected action to the rewards of the entire episode
        rewardsCurrentEpisode += rewardsCurrentState

        previousAction = action

        #Determine whether the goal has been reached or not
        done, goalReached = episodeEnd(state,done,goal,goalReached)
        
        #End the episode if the action makes the drone reach the goal
        if done == True:
            break

    #When the episode ends, use the exponential decay formula to decrease the exploration rate
    explorationRate = minExplorationRate + ((maxExplorationRate - minExplorationRate) * np.exp(-explorationDecayRate * episode))

    #Store the rewards obtained from each episode in the previously defined array
    rewardsAllEpisodes.append(rewardsCurrentEpisode)

#Calculate the total reward obtained from all episodes
totalReward = sum(rewardsAllEpisodes)
print("\n\nTotal Reward: {}".format(totalReward))

#Split the rewards for each episode into subarrays
#Each subarray contains the rewards obtained every 100 episodes
#(e.g. subarray 1 is for the first 100 episodes, subarray 2 is for the next 100 episodes, and so on)
rewardsPerHundredEpisodes = np.split(np.array(rewardsAllEpisodes), numEpisodes/100)
count = 100

#For each subarray, display which set of 100 episodes it corresponds to, as well as the average reward during those episodes
print("\n\nAverage reward per hundred episodes\n")
for r in rewardsPerHundredEpisodes:
    print(count, ':', str(sum(r/100)))
    count += 100

#Display the final Q-Table
print("\n\nQ-Table A\n")
print(qTableA)

#Display how many times the goal was reached
print('\n\nGoal was reached {} times for environment A'.format(goalReached))

##############################TRAINING LOOP B##############################

explorationRate = 1

#Initialize an array that will contain and separate the rewards gained in each episode
rewardsAllEpisodes = []

#Create a variable to keep track of how many times the goal is reached
goalReached = 0

stSpaceShpB = stSpaceShp(envB)
stSpaceSizeB = stSpaceSize(stSpaceShpB)
rowsB = stSpaceShpB[0]
colsB = stSpaceShpB[1]
qTableB = qTable(stSpaceSizeB, actionSpaceSize)

for episode in range(numEpisodes):

    state = [stSpaceShpB[0]-1,0] #set the initial state
    gridWidth = stSpaceShpB[1]
    stateNum = ((state[0]) * gridWidth) + (state[1])
    #print('initial state: ', state)
    #print('initial state in integer form: ', stateNum)
    steps = 0

    done = False
    rewardsCurrentEpisode = 0

    #Dictate how the program selects an action and use the Bellman equation and a weighted sum of the previous Q value to update the Q-Table after an action is taken
    #This loop is executed until the maximum number of steps is reached or until the 'agent' reaches the goal
    for step in range(maxStepsPerEpisode):

        steps += 1

        rewardsCurrentState = 0

        #Generate a random number used to decide if the drone will explore or exploit the environment
        explorationRateThreshold = random.uniform(0, 1)

        #Take an action according to the Q-table to exploit the environment
        if explorationRateThreshold > explorationRate:
            action = np.argmax(qTableB[stateNum, :])

        #Take a random action to explore the environment
        else:
            action = random.choice(range(0,actionSpaceSize))
        
        #print('action: ', action)

        #Generate variables that represent the state prior to implementing the selected action; these will be used to update the Q-Table
        prevState = state
        prevstateNum = ((state[0]) * gridWidth) + (state[1])

        #Change the position of the state according to the action selected; adjust the reward accordingly
        state, posReward = updatePosReward(action,state,stSpaceShpB)
        #print('state due to action: ', state)
        rewardsCurrentState += posReward

        #Check the distance from the current position to the goal; adjust the reward accordingly
        distReward = getDistReward(state,stSpaceShpA,goal)
        rewardsCurrentState += distReward

        #Check the risk value of the position of the drone; adjust the reward accordingly using this value and distReward
        #print('risk value of current state: ', envB[state[0]-1, state[1]-1])
        riskReward = getRiskReward(envB,state)
        rewardsCurrentState += riskReward

        #Keep track of when the goal has been reached and implement a reward penalty for every step that does not reach the goal
        finishReward = getFinishReward(envB, state, goal)
        rewardsCurrentState += finishReward

        #Use the [row,col] state to create a state that is represented by a single integer
        stateNum = ((state[0]) * gridWidth) + (state[1])
        #print('state in integer form: ', stateNum)

        #Update the Q-table with all of the rewards obtained from the selected action
        qTableB[prevstateNum, action] = (qTableB[prevstateNum, action] * (1 - learningRate)) + (learningRate * (rewardsCurrentState + (discountRate * np.max(qTableB[stateNum, :]))))

        #^Use a 3D array or a dictionary to be able to use the [row,col] state variable

        #print('total reward obtained from this action: ', rewardsCurrentState)

        #Add the rewards from the selected action to the rewards of the entire episode
        rewardsCurrentEpisode += rewardsCurrentState

        previousAction = action

        #Determine whether the goal has been reached or not
        done, goalReached = episodeEnd(state,done,goal,goalReached)
        
        #End the episode if the action makes the drone reach the goal
        if done == True:
            break

    #When the episode ends, use the exponential decay formula to decrease the exploration rate
    explorationRate = minExplorationRate + ((maxExplorationRate - minExplorationRate) * np.exp(-explorationDecayRate * episode))

    #Store the rewards obtained from each episode in the previously defined array
    rewardsAllEpisodes.append(rewardsCurrentEpisode)

#Calculate the total reward obtained from all episodes
totalReward = sum(rewardsAllEpisodes)
print("\n\nTotal Reward: {}".format(totalReward))

#Split the rewards for each episode into subarrays
#Each subarray contains the rewards obtained every 100 episodes
#(e.g. subarray 1 is for the first 100 episodes, subarray 2 is for the next 100 episodes, and so on)
rewardsPerHundredEpisodes = np.split(np.array(rewardsAllEpisodes), numEpisodes/100)
count = 100

#For each subarray, display which set of 100 episodes it corresponds to, as well as the average reward during those episodes
print("\n\nAverage reward per hundred episodes\n")
for r in rewardsPerHundredEpisodes:
    print(count, ':', str(sum(r/100)))
    count += 100

#Display the final Q-Table
print("\n\nQ-Table B\n")
print(qTableB)

#Display how many times the goal was reached
print('\n\nGoal was reached {} times for environment B'.format(goalReached))

##############################TRAINING LOOP C##############################

explorationRate = 1

#Initialize an array that will contain and separate the rewards gained in each episode
rewardsAllEpisodes = []

#Create a variable to keep track of how many times the goal is reached
goalReached = 0

stSpaceShpC = stSpaceShp(envC)
stSpaceSizeC = stSpaceSize(stSpaceShpC)
rowsC = stSpaceShpC[0]
colsC = stSpaceShpC[1]
qTableC = qTable(stSpaceSizeC, actionSpaceSize)

for episode in range(numEpisodes):

    state = [stSpaceShpC[0]-1,0] #set the initial state
    gridWidth = stSpaceShpC[1]
    stateNum = ((state[0]) * gridWidth) + (state[1])
    #print('initial state: ', state)
    #print('initial state in integer form: ', stateNum)
    steps = 0

    done = False
    rewardsCurrentEpisode = 0

    #Dictate how the program selects an action and use the Bellman equation and a weighted sum of the previous Q value to update the Q-Table after an action is taken
    #This loop is executed until the maximum number of steps is reached or until the 'agent' reaches the goal
    for step in range(maxStepsPerEpisode):

        steps += 1

        rewardsCurrentState = 0

        #Generate a random number used to decide if the drone will explore or exploit the environment
        explorationRateThreshold = random.uniform(0, 1)

        #Take an action according to the Q-table to exploit the environment
        if explorationRateThreshold > explorationRate:
            action = np.argmax(qTableC[stateNum, :])

        #Take a random action to explore the environment
        else:
            action = random.choice(range(0,actionSpaceSize))
        
        #print('action: ', action)

        #Generate variables that represent the state prior to implementing the selected action; these will be used to update the Q-Table
        prevState = state
        prevstateNum = ((state[0]) * gridWidth) + (state[1])

        #Change the position of the state according to the action selected; adjust the reward accordingly
        state, posReward = updatePosReward(action,state,stSpaceShpC)
        #print('state due to action: ', state)
        rewardsCurrentState += posReward

        #Check the distance from the current position to the goal; adjust the reward accordingly
        distReward = getDistReward(state,stSpaceShpA,goal)
        rewardsCurrentState += distReward

        #Check the risk value of the position of the drone; adjust the reward accordingly using this value and distReward
        #print('risk value of current state: ', envB[state[0]-1, state[1]-1])
        riskReward = getRiskReward(envC,state)
        rewardsCurrentState += riskReward

        #Keep track of when the goal has been reached and implement a reward penalty for every step that does not reach the goal
        finishReward = getFinishReward(envC, state, goal)
        rewardsCurrentState += finishReward

        #Use the [row,col] state to create a state that is represented by a single integer
        stateNum = ((state[0]) * gridWidth) + (state[1])
        #print('state in integer form: ', stateNum)

        #Update the Q-table with all of the rewards obtained from the selected action
        qTableC[prevstateNum, action] = (qTableC[prevstateNum, action] * (1 - learningRate)) + (learningRate * (rewardsCurrentState + (discountRate * np.max(qTableC[stateNum, :]))))

        #^Use a 3D array or a dictionary to be able to use the [row,col] state variable

        #print('total reward obtained from this action: ', rewardsCurrentState)

        #Add the rewards from the selected action to the rewards of the entire episode
        rewardsCurrentEpisode += rewardsCurrentState

        previousAction = action

        #Determine whether the goal has been reached or not
        done, goalReached = episodeEnd(state,done,goal,goalReached)
        
        #End the episode if the action makes the drone reach the goal
        if done == True:
            break

    #When the episode ends, use the exponential decay formula to decrease the exploration rate
    explorationRate = minExplorationRate + ((maxExplorationRate - minExplorationRate) * np.exp(-explorationDecayRate * episode))

    #Store the rewards obtained from each episode in the previously defined array
    rewardsAllEpisodes.append(rewardsCurrentEpisode)

#Calculate the total reward obtained from all episodes
totalReward = sum(rewardsAllEpisodes)
print("\n\nTotal Reward: {}".format(totalReward))

#Split the rewards for each episode into subarrays
#Each subarray contains the rewards obtained every 100 episodes
#(e.g. subarray 1 is for the first 100 episodes, subarray 2 is for the next 100 episodes, and so on)
rewardsPerHundredEpisodes = np.split(np.array(rewardsAllEpisodes), numEpisodes/100)
count = 100

#For each subarray, display which set of 100 episodes it corresponds to, as well as the average reward during those episodes
print("\n\nAverage reward per hundred episodes\n")
for r in rewardsPerHundredEpisodes:
    print(count, ':', str(sum(r/100)))
    count += 100

#Display the final Q-Table
print("\n\nQ-Table C\n")
print(qTableC)

#Display how many times the goal was reached
print('\n\nGoal was reached {} times for environment C'.format(goalReached))

##############################TRAINING LOOP D##############################

explorationRate = 1

#Initialize an array that will contain and separate the rewards gained in each episode
rewardsAllEpisodes = []

#Create a variable to keep track of how many times the goal is reached
goalReached = 0

stSpaceShpD = stSpaceShp(envD)
stSpaceSizeD = stSpaceSize(stSpaceShpD)
rowsD = stSpaceShpD[0]
colsD = stSpaceShpD[1]
qTableD = qTable(stSpaceSizeD, actionSpaceSize)

for episode in range(numEpisodes):

    state = [stSpaceShpD[0]-1,0] #set the initial state
    gridWidth = stSpaceShpD[1]
    stateNum = ((state[0]) * gridWidth) + (state[1])
    #print('initial state: ', state)
    #print('initial state in integer form: ', stateNum)
    steps = 0

    done = False
    rewardsCurrentEpisode = 0

    #Dictate how the program selects an action and use the Bellman equation and a weighted sum of the previous Q value to update the Q-Table after an action is taken
    #This loop is executed until the maximum number of steps is reached or until the 'agent' reaches the goal
    for step in range(maxStepsPerEpisode):

        steps += 1

        rewardsCurrentState = 0

        #Generate a random number used to decide if the drone will explore or exploit the environment
        explorationRateThreshold = random.uniform(0, 1)

        #Take an action according to the Q-table to exploit the environment
        if explorationRateThreshold > explorationRate:
            action = np.argmax(qTableD[stateNum, :])

        #Take a random action to explore the environment
        else:
            action = random.choice(range(0,actionSpaceSize))
        
        #print('action: ', action)

        #Generate variables that represent the state prior to implementing the selected action; these will be used to update the Q-Table
        prevState = state
        prevstateNum = ((state[0]) * gridWidth) + (state[1])

        #Change the position of the state according to the action selected; adjust the reward accordingly
        state, posReward = updatePosReward(action,state,stSpaceShpD)
        #print('state due to action: ', state)
        rewardsCurrentState += posReward

        #Check the distance from the current position to the goal; adjust the reward accordingly
        distReward = getDistReward(state,stSpaceShpD,goal)
        rewardsCurrentState += distReward

        #Check the risk value of the position of the drone; adjust the reward accordingly using this value and distReward
        #print('risk value of current state: ', envA[state[0]-1, state[1]-1])
        riskReward = getRiskReward(envD,state)
        rewardsCurrentState += riskReward

        #Keep track of when the goal has been reached and implement a reward penalty for every step that does not reach the goal
        finishReward = getFinishReward(envD, state, goal)
        rewardsCurrentState += finishReward

        #Use the [row,col] state to create a state that is represented by a single integer
        stateNum = ((state[0]) * gridWidth) + (state[1])
        #print('state in integer form: ', stateNum)

        #Update the Q-table with all of the rewards obtained from the selected action
        qTableD[prevstateNum, action] = (qTableD[prevstateNum, action] * (1 - learningRate)) + (learningRate * (rewardsCurrentState + (discountRate * np.max(qTableD[stateNum, :]))))

        #^Use a 3D array or a dictionary to be able to use the [row,col] state variable

        #print('total reward obtained from this action: ', rewardsCurrentState)

        #Add the rewards from the selected action to the rewards of the entire episode
        rewardsCurrentEpisode += rewardsCurrentState

        previousAction = action

        #Determine whether the goal has been reached or not
        done, goalReached = episodeEnd(state,done,goal,goalReached)
        
        #End the episode if the action makes the drone reach the goal
        if done == True:
            break

    #When the episode ends, use the exponential decay formula to decrease the exploration rate
    explorationRate = minExplorationRate + ((maxExplorationRate - minExplorationRate) * np.exp(-explorationDecayRate * episode))

    #Store the rewards obtained from each episode in the previously defined array
    rewardsAllEpisodes.append(rewardsCurrentEpisode)

#Calculate the total reward obtained from all episodes
totalReward = sum(rewardsAllEpisodes)
print("\n\nTotal Reward: {}".format(totalReward))

#Split the rewards for each episode into subarrays
#Each subarray contains the rewards obtained every 100 episodes
#(e.g. subarray 1 is for the first 100 episodes, subarray 2 is for the next 100 episodes, and so on)
rewardsPerHundredEpisodes = np.split(np.array(rewardsAllEpisodes), numEpisodes/100)
count = 100

#For each subarray, display which set of 100 episodes it corresponds to, as well as the average reward during those episodes
print("\n\nAverage reward per hundred episodes\n")
for r in rewardsPerHundredEpisodes:
    print(count, ':', str(sum(r/100)))
    count += 100

#Display the final Q-Table
print("\n\nQ-Table D\n")
print(qTableD)

#Display how many times the goal was reached
print('\n\nGoal was reached {} times for environment D'.format(goalReached))

##############################COMBINED Q-TABLE & SIMULATION LOOP##############################

#Assign a weight to each risk and create a combined Q-Table
weightA, weightB, weightC, weightD = getWeights(envA,envB,envC,envD,rowsA,colsA)
print('\n\nweightA ', weightA, 'weightB: ', weightB, 'weightC: ', weightC, 'weightD: ', weightD)
qTable = weightedTable(qTableA,qTableB,qTableC,qTableD,weightA,weightB,weightC,stSpaceSizeA,actionSpaceSize)
print("\n\nWeighted Q-Table\n")
print(qTable)

#Create screen parameters to be used for the simulation
height = 600 #amount of pixels of height
width = 600 #amount of pixels of width
rows, cols = stSpaceShpA
gridLocationSize = width // cols

#Create a screen to be used for the simulation
screen = pygame.display.set_mode((height,width)) #create the screen to be used for the simulation
pygame.display.set_caption('Drone Path Planner') #title of simulation

Padding = 10 #define the amount of pixels between the edge of the circle that represents the drone and the edges of its current location
Outline = 2 #define how thick the circle's outline is
radius = gridLocationSize // 2 - Padding #define the radius of the circle that will represent the drone in Pygame

droneColor = black #set the color of the drone

clock = pygame.time.Clock() #set the clock that will determine how fast the simulation progresses

state = [stSpaceShpA[0]-1,0] #set the initial state of the simulation
stateNum = ((state[0]) * gridWidth) + (state[1])
stateHistory = []
stateHistory.append(stateNum)

step = 0 #create a variable that counts the steps taken inside the simulation

done = False

stepsPerSimulation = 30

#Begin the simulation
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    step += 1
    screen.fill(black) #create a black screen to 'delete' the images from the previous step

    createEnv(envA,envB,envC,envD,rows,cols,screen,gridLocationSize) #create the Pygame array based on the 2D numpy array used for training

    #Determine the drone's action based on its state
    if step == 1: #if the drone is taking its first step, use the Q-Table as normal
        action = np.argmax(qTable[stateNum, :])

    else: #for subsequent steps, if back-and-forth movement occurs, use the 2nd highest Q-Value at the current state to find an alternate path
        if stateHistory[step - 1] == stateHistory[step - 3]:
            action = altPath(stateNum, qTable)
        
        else:
            action = np.argmax(qTable[stateNum, :])

    updatePos(action,state,stSpaceShpA) #update the drone's position by using the determined action
    stateNum = ((state[0]) * gridWidth) + (state[1])
    stateHistory.append(stateNum)      

    x = (state[1]+1) * gridLocationSize - gridLocationSize // 2 #assign the new x coordinate position to the drone
    y = (state[0]+1) * gridLocationSize - gridLocationSize // 2 #assign the new y coordinate position to the drone

    #Draw the circles that represent the drone's updated location
    pygame.draw.circle(screen, black, (x, y), radius + Outline)
    pygame.draw.circle(screen, droneColor, (x, y), radius)

    done, goalReached = episodeEnd(state,done,goal,goalReached) #determine if the drone has reached the goal

    #If the maximum amount of steps are performed, exit the simulation
    if step == stepsPerSimulation:
        done = True
    
    #Update the Pygame screen based on the Frames Per Second (FPS) specified
    pygame.display.update()
    clock.tick(0.5) #set the FPS; 1/FPS will give the duration of each frame
