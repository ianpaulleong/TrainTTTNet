from tttPlayer import tttPlayer
from tttBoard import tttBoard
import numpy as np

class tttTable:
    # This class represents a table on which there sits a Tic-Tac-Toe board and
    # at which there sit two players. It has several functions:
    #
    # init: initialize the table. This involves initializing a tttBoard and two
    #       tttPlayers. These players need to be provided as an input if they 
    #       are Neural-Network based. Otherwise, they can be created from 
    #       scratch
    #
    # changeFirstPlayer: Change whether player 0 or player 1 goes first
    #
    # playMatch: have the players play a tic-tac-toe match.
    #
    # clearBoard: reset the match to the initial state
    # 
    # togglePrintMode: change whether or not the game state gets printed after
    #                  each move
    
    
    def __init__(self, player0Mode = "random", player1Mode = "random", p0 = 0, p1 = 0):
        if (player0Mode == "random") | (player0Mode == "optimal"):
            self.player0 = tttPlayer(player0Mode)
        elif player0Mode == "network":
            self.player0 = p0
        
        if (player1Mode == "random") | (player1Mode == "optimal"):
            self.player1 = tttPlayer(player1Mode)
        elif player1Mode == "network":
            self.player1 = p1
        
        # I don't know if this particular property ever gets used
        self.areTheyNets = np.array([0,0],int)
        if player0Mode == "network":
            self.areTheyNets[0] = 1
        if player1Mode == "network":
            self.areTheyNets[1] = 1
        
        # Set the first player
        self.whoGoesFirst = 0
        
        # Create a tic-tac-toe board
        self.theBoard = tttBoard()
        
        # Preallocate some space for data about the game to reside in
        self.p0Evals = np.zeros([9,3,3])
        self.p1Evals = np.zeros([9,3,3])
        self.boardHist = np.zeros([9,3,3],int)
        self.numMovesPlayed = 0
        
        
    def changeFirstPlayer(self):
        # if it's a 1, make it a 0, and vice versa
        self.whoGoesFirst = 1 - self.whoGoesFirst

        
    def playMatch(self, whoGoesFirst = 0):
        # First, if the board isn't empty, clear the board
        if self.numMovesPlayed != 0:
            self.clearBoard()
        
        # Next, make up to 9 moves!
        for ii in range(9):
            # Check to see if it's an odd or even turn
            if ii%2 == 0:
                # Store the current board state
                self.boardHist[ii,:,:] = self.theBoard.board
                
                # Have both players evaluate the board. This can used for 
                # NN training data later
                self.p0Evals[ii,:,:] = self.player0.evaluateBoard(self.theBoard.board.copy())
                self.p1Evals[ii,:,:] = self.player1.evaluateBoard(self.theBoard.board.copy())
                
                # Depending on who got to go first, have one of the players 
                # make a move
                if self.whoGoesFirst == 0:
                    self.theBoard.makeMove(self.player0.chooseMove(self.theBoard.board.copy()))
                else:
                    self.theBoard.makeMove(self.player1.chooseMove(self.theBoard.board.copy()))                    
            else:
                # Create a opposite version of the board so that the current 
                # player's past moves are marked with a 1 and their opponent's 
                # past moves are marked with a -1
                negBoard = -self.theBoard.board.copy()
                
                # Store the current board state
                self.boardHist[ii,:,:] = negBoard
                
                # Have both players evaluate the board. This can used for 
                # NN training data later
                self.p0Evals[ii,:,:] = self.player0.evaluateBoard(negBoard.copy())
                self.p1Evals[ii,:,:] = self.player1.evaluateBoard(negBoard.copy())
                    
                # Depending on who got to go first, have one of the players 
                # make a move
                if self.whoGoesFirst == 1:
                    self.theBoard.makeMove(self.player0.chooseMove(negBoard.copy()))
                else:
                    self.theBoard.makeMove(self.player1.chooseMove(negBoard.copy()))
                    
            # Store the move number
            self.numMovesPlayed = self.numMovesPlayed + 1
            
            # Check to see if anyone has won yet. If so, trim any trailing 
            # zeros (arising from the preallocation) from the stored match data
            # and then stop playing
            if self.theBoard.winState != 0:
                self.boardHist = self.boardHist[0:self.numMovesPlayed]
                self.p0Evals = self.p0Evals[0:self.numMovesPlayed]
                self.p1Evals = self.p1Evals[0:self.numMovesPlayed]
                break
    
    def clearBoard(self):
        # Clear the board and any recorded data
        self.p0Evals = np.zeros([9,3,3])
        self.p1Evals = np.zeros([9,3,3])
        self.boardHist = np.zeros([9,3,3],int)
        self.numMovesPlayed = 0
        self.theBoard.clearBoard()
        
    def togglePrintMode(self):
        # Turn the feature where the board state is printed each move on or off
        self.theBoard.togglePrintMode()
    
        