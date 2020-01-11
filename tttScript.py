# -*- coding: utf-8 -*-

import os.path
import pickle
import time
import numpy as np
import torch
from tttPlayer import tttPlayer
from tttTable import tttTable
from tttTrainer import tttTrainer

# Hyperparameters:
learning_rate = 0.01
howManyHidden0 = 63
howManyHidden1 = 63
decayFactor = 1
sheepConfidenceFactor = 0.3
lionConfidenceFactor = 0.9
sheepProbPicker = 1
lionProbPicker = 0
numGamesVsRandPerIter = 100
numGamesVsSelfPerIter = 100
numGamesVsLionPerIter = 400
numTrainIters = 1000
numPlayTrainCycles = 30
useGPU = 1

numTestRandGames = 500
numTestOptGames = 500

theDevice = myDevice = torch.device("cuda:0")

# check to see if the network is already in the namespace (maybe unneeded now?)
storedNetworks = 0
if os.path.isfile("sheepNetMaybe.pickle"):
    storedNetworks = 1
    
# Generate data from randomly played games
randTable = tttTable()
randTrainer = tttTrainer(0,decayFactor,sheepConfidenceFactor)
    
winPctRandVsRand = 0
for ii in range(1000):
    randTable.playMatch()
    if ii%100 == 0:
        print(ii/100)
    randTrainer.addToDatabase(randTable)

# Figuring stupid tables out
boardArr = np.array(list(randTrainer.boards.values()))
evalArr = np.array(list(randTrainer.evals.values()))
numBoards = boardArr.shape[0]
boardTensFlat = torch.tensor(boardArr.reshape(numBoards,9), dtype = torch.float32)
evalTensFlat = torch.tensor(evalArr.reshape(numBoards,9), dtype = torch.float32)
numBoards = boardArr.shape[0]
if useGPU == 1:
    boardTensFlat = boardTensFlat.cuda()
    evalTensFlat = evalTensFlat.cuda()

if storedNetworks == 0:
    # Create two neural Networks
    sheepNet = torch.nn.Sequential(
        torch.nn.Linear(9, howManyHidden0),
        torch.nn.Sigmoid(),
        torch.nn.Linear(howManyHidden0, howManyHidden1),
        torch.nn.Sigmoid(),    
        torch.nn.Linear(howManyHidden1, 9),
        torch.nn.Sigmoid(),    
    )
    if useGPU == 1:
        sheepNet = sheepNet.cuda()
    origSheepEval = sheepNet(boardTensFlat)
    
    lionNet = torch.nn.Sequential(
        torch.nn.Linear(9, howManyHidden0),
        torch.nn.Sigmoid(),
        torch.nn.Linear(howManyHidden0, howManyHidden1),
        torch.nn.Sigmoid(),    
        torch.nn.Linear(howManyHidden1, 9),
        torch.nn.Sigmoid(),    
    )
    if useGPU == 1:
        lionNet = lionNet.cuda()
    
    # Do initial training of networks
    loss_fn = torch.nn.MSELoss(reduction='sum')
    sheepTimizer = torch.optim.Adam(sheepNet.parameters(), lr=learning_rate)
    lionIzer = torch.optim.Adam(lionNet.parameters(), lr=learning_rate)
    for t in range(numTrainIters):
        sheepEval = sheepNet(boardTensFlat)
        theSheepLoss = loss_fn(sheepEval, evalTensFlat)
        lionEval = lionNet(boardTensFlat)
        theLionLoss = loss_fn(lionEval, evalTensFlat)
        
        if t % 100 == 99:
            print(t, theSheepLoss.item(), theLionLoss.item())
        sheepTimizer.zero_grad()
        theSheepLoss.backward()
        sheepTimizer.step()
        lionIzer.zero_grad()
        theLionLoss.backward()
        lionIzer.step()
else:
    with open('sheepNetMaybe.pickle', 'rb') as handle:
        sheepNet = pickle.load(handle)
    with open('lionNetMaybe.pickle', 'rb') as handle:
        lionNet = pickle.load(handle)
    if useGPU == 1:
        sheepNet = sheepNet.cuda()
        lionNet = lionNet.cuda()            

sheepPlayer = tttPlayer("network", sheepNet, sheepProbPicker)
lionPlayer = tttPlayer("network", lionNet, lionProbPicker)
loss_fn = torch.nn.MSELoss(reduction='sum')
sheepTimizer = torch.optim.Adam(sheepNet.parameters(), lr=learning_rate)
lionIzer = torch.optim.Adam(lionNet.parameters(), lr=learning_rate)

# Initialize Tables for Sheep and Lion to play and respective Trainers
sheepVsRandTable = tttTable("network","random",sheepPlayer,0)
sheepVsSelfTable = tttTable("network","network",sheepPlayer,sheepPlayer)
sheepVsLionTable = tttTable("network","network",sheepPlayer,lionPlayer)
sheepTrainer = tttTrainer(0, decayFactor, sheepConfidenceFactor)
lionTrainer = tttTrainer(1, decayFactor, lionConfidenceFactor,0.02)



# Play games and train
for jj in range(numPlayTrainCycles):
    print("Play Train Cycle", jj,"\n")
    theStartTime = time.time()
    if useGPU == 1:
        sheepPlayer.net = sheepPlayer.net.cpu()
        cow = 2
        lionPlayer.net = lionPlayer.net.cpu()
    sheepTrainer.clearDatabase()
    lionTrainer.clearDatabase()
    winPctVsRand = 0
    for ii in range(numGamesVsRandPerIter):
        sheepVsRandTable.playMatch()
        sheepTrainer.addToDatabase(sheepVsRandTable)
        if sheepVsRandTable.theBoard.winState == 0:
            winPctVsRand = winPctVsRand + 0.5
        else:
            if ((ii%2 == 0)&(sheepVsRandTable.theBoard.winState == 1))|((ii%2 == 1)&(sheepVsRandTable.theBoard.winState == -1)):
                winPctVsRand = winPctVsRand + 1                
        sheepVsRandTable.clearBoard()
        sheepVsRandTable.changeFirstPlayer()
    winPctVsRand /= numGamesVsRandPerIter
    
    winPctVsSelf = 0
    for ii in range(numGamesVsSelfPerIter):
        sheepVsSelfTable.playMatch()
        sheepTrainer.addToDatabase(sheepVsSelfTable)
        if sheepVsSelfTable.theBoard.winState == 0:
            winPctVsSelf = winPctVsSelf + 0.5
        else:
            if ((ii%2 == 0)&(sheepVsSelfTable.theBoard.winState == 1))|((ii%2 == 1)&(sheepVsSelfTable.theBoard.winState == -1)):
                winPctVsSelf = winPctVsSelf + 1
        sheepVsSelfTable.clearBoard()
        sheepVsSelfTable.changeFirstPlayer()
    winPctVsSelf /= numGamesVsSelfPerIter
    
    winPctVsLion = 0
    for ii in range(numGamesVsLionPerIter):
        sheepVsLionTable.playMatch()
        sheepTrainer.addToDatabase(sheepVsLionTable)
        lionTrainer.addToDatabase(sheepVsLionTable)
        if sheepVsLionTable.theBoard.winState == 0:
            winPctVsLion = winPctVsLion + 0.5
        else:
            if ((ii%2 == 0)&(sheepVsLionTable.theBoard.winState == 1))|((ii%2 == 1)&(sheepVsLionTable.theBoard.winState == -1)):
                winPctVsLion = winPctVsLion + 1
        sheepVsLionTable.clearBoard()
        sheepVsLionTable.changeFirstPlayer()
    winPctVsLion /= numGamesVsLionPerIter
    if winPctVsLion > 0.5:
        sheepVsLionTable.player1.probPickerOn = 1
    else:
        sheepVsLionTable.player1.probPickerOn = 0
    
    winPctRandVsRand = 0
    for ii in range(200):
        randTable.playMatch()
        if randTable.theBoard.winState == 0:
            winPctRandVsRand = winPctRandVsRand + 0.5
        else:
            if ((ii%2 == 0)&(randTable.theBoard.winState == 1))|((ii%2 == 1)&(randTable.theBoard.winState == -1)):
                winPctRandVsRand = winPctRandVsRand + 1
        randTable.clearBoard()
        randTable.changeFirstPlayer()
    winPctRandVsRand /= 200
    
    theEndPlayTime = time.time()
    print([winPctVsRand,winPctVsSelf,winPctVsLion,winPctRandVsRand])
    print("\n")
    
    
    sheepBoardArr = np.array(list(sheepTrainer.boards.values()))
    sheepEvalArr = np.array(list(sheepTrainer.evals.values()))
    numSheepBoards = sheepBoardArr.shape[0]
    sheepBoardTensFlat = torch.tensor(sheepBoardArr.reshape(numSheepBoards,9), dtype = torch.float32)
    sheepEvalTensFlat = torch.tensor(sheepEvalArr.reshape(numSheepBoards,9), dtype = torch.float32)
    if useGPU == 1:
        sheepBoardTensFlat = sheepBoardTensFlat.cuda()
        sheepEvalTensFlat = sheepEvalTensFlat.cuda()
        sheepPlayer.net = sheepPlayer.net.cuda()
        sheepTimizer = torch.optim.Adam(sheepNet.parameters(), lr=learning_rate)
    
    lionBoardArr = np.array(list(lionTrainer.boards.values()))
    lionEvalArr = np.array(list(lionTrainer.evals.values()))
    numLionBoards = lionBoardArr.shape[0]
    lionBoardTensFlat = torch.tensor(lionBoardArr.reshape(numLionBoards,9), dtype = torch.float32)
    lionEvalTensFlat = torch.tensor(lionEvalArr.reshape(numLionBoards,9), dtype = torch.float32)
    if useGPU == 1:
        lionBoardTensFlat = lionBoardTensFlat.cuda()
        lionEvalTensFlat = lionEvalTensFlat.cuda()
        lionPlayer.net = lionPlayer.net.cuda()
        lionIzer = torch.optim.Adam(lionNet.parameters(), lr=learning_rate)
    
    theStartTrainTime = time.time()
    for t in range(numTrainIters):
        sheepEval = sheepNet(sheepBoardTensFlat)
        theSheepLoss = loss_fn(sheepEval, sheepEvalTensFlat)
        lionEval = lionNet(lionBoardTensFlat)
        theLionLoss = loss_fn(lionEval, lionEvalTensFlat)
        
        if t % 100 == 99:
            print(t, theSheepLoss.item(), theLionLoss.item())
        sheepTimizer.zero_grad()
        theSheepLoss.backward()
        sheepTimizer.step()
        lionIzer.zero_grad()
        theLionLoss.backward()
        lionIzer.step()
    theEndTime = time.time()
    print('Play Time: ', theEndPlayTime - theStartTime)
    print('Train Time: ', theEndTime - theStartTrainTime)
    print('Total Time: ', theEndTime - theStartTime,'\n')
# This is where playing games and training ends

# Store the trained network and evaluate it
if useGPU == 1:
    sheepPlayer.net = sheepPlayer.net.cpu()
    lionPlayer.net = lionPlayer.net.cpu()

with open('sheepNetMaybe.pickle', 'wb') as handle:
    pickle.dump(sheepNet, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open('lionNetMaybe.pickle', 'wb') as handle:
    pickle.dump(lionNet, handle, protocol=pickle.HIGHEST_PROTOCOL)

winPctVsRand = 0
sheepVsRandTable.player0.probPickerOn = 0
for ii in range(numTestRandGames):
    sheepVsRandTable.playMatch()
    if sheepVsRandTable.theBoard.winState == 0:
        winPctVsRand = winPctVsRand + 0.5
    else:
        if ((ii%2 == 0)&(sheepVsRandTable.theBoard.winState == 1))|((ii%2 == 1)&(sheepVsRandTable.theBoard.winState == -1)):
            winPctVsRand = winPctVsRand + 1
        else:
            print("\nLosing Game: ")
            for jj in range(sheepVsRandTable.numMovesPlayed):
                print(sheepVsRandTable.boardHist[jj,:,:])
                print("\n")
    sheepVsRandTable.clearBoard()
    sheepVsRandTable.changeFirstPlayer()
winPctVsRand /= numTestRandGames
print(winPctVsRand)
sheepVsRandTable.player0.probPickerOn = 1


sheepVsOptTable = tttTable("network","optimal",sheepPlayer)
sheepVsOptTable.player0.probPickerOn = 0
winPctVsOpt = 0
for ii in range(numTestOptGames):
    sheepVsOptTable.playMatch()
    if sheepVsOptTable.theBoard.winState == 0:
        winPctVsOpt += 0.5
    else:
        if ((ii%2 == 0)&(sheepVsOptTable.theBoard.winState == 1))|((ii%2 == 1)&(sheepVsOptTable.theBoard.winState == -1)):
            winPctVsOpt = winPctVsOpt + 1
        else:
            print("\nLosing Game: ")
            for jj in range(sheepVsOptTable.numMovesPlayed):
                print(sheepVsOptTable.boardHist[jj,:,:])
                print("\n")
    sheepVsOptTable.clearBoard()
    sheepVsOptTable.changeFirstPlayer()
winPctVsOpt /= numTestOptGames
print(winPctVsOpt)

sheepVsOptTable.player0.probPickerOn = 1
if useGPU == 1:
    sheepPlayer.net = sheepPlayer.net.cuda()
    sheepTimizer = torch.optim.Adam(sheepNet.parameters(), lr=learning_rate)
    lionPlayer.net = lionPlayer.net.cuda()
    lionIzer = torch.optim.Adam(lionNet.parameters(), lr=learning_rate)