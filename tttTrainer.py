# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 22:41:07 2019

@author: GAMEZ
"""

import numpy as np
from utilityFunctions import numToIndices 

class tttTrainer:
    
    def __init__(self, myPlayerNumber, decayFactor = 1, confFactor = 0.3, drawValue = 0.5):
        self.boards = {}
        self.evals = {}
        self.myPlayerNumber = myPlayerNumber
        self.drawValue = drawValue
        self.decayFactor = decayFactor
        self.confFactor = confFactor
        
    def clearDatabase(self):
        self.boards = {}
        self.evals = {}
    
    def addToDatabase(self,itsAtttTable):
        boardHist = itsAtttTable.boardHist
        numMovesPlayed = itsAtttTable.numMovesPlayed
        whoWon = itsAtttTable.theBoard.winState
        boardKeys = list()
        for ii in range(numMovesPlayed):
            boardKeys.append(boardHist[ii].tobytes())
        if self.myPlayerNumber == 0:
            gameEvals = itsAtttTable.p0Evals
        else:
            gameEvals = itsAtttTable.p1Evals
        moveList = itsAtttTable.theBoard.moveList
        
        for ii in range(numMovesPlayed):
            moveNum = numMovesPlayed - ii - 1
            if boardKeys[moveNum] in self.boards:
                theEval = self.evals[boardKeys[moveNum]]
            else:
                self.boards[boardKeys[moveNum]] = boardHist[moveNum]
                theEval = gameEvals[moveNum]
                
    
    
#            print("Eval Time!\n")
#            print(boardHist[moveNum])
#            print("\n")
#            print(theEval)
#            print("\n")
            
            if ii == 0:
                movedIndices = numToIndices(moveList[moveNum])
                movedRow = movedIndices[0]
                movedCol = movedIndices[1]
                if whoWon != 0:
                    theEval[movedRow,movedCol] = 1
                else:
                    theEval[movedRow,movedCol] = self.drawValue
            else:
                movedIndices = numToIndices(moveList[moveNum])
                movedRow = movedIndices[0]
                movedCol = movedIndices[1]
                if oppNextBestScore == 1:
                    theEval[movedRow,movedCol] = 0
                else:
                    theEval[movedRow,movedCol] = theEval[movedRow,movedCol] + self.confFactor*(self.decayFactor*(1-oppNextBestScore) - theEval[movedRow,movedCol])            
#            print(theEval)
#            print("\n")
            oppNextBestScore = max(theEval[boardHist[moveNum]==0])
            self.boards[boardKeys[moveNum]] = boardHist[moveNum]
            self.evals[boardKeys[moveNum]] = theEval
            
        
        