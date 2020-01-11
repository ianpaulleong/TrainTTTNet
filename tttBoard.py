import numpy as np

class tttBoard:
    # This is a tic-tac-toe board class.
    
    # Here are the properties:
    def __init__(self, printMode = 0):
        
        # This is the board state, a 3x3 array. Valid values are -1 for 'o', 0 for empty, and 1 for 'x'
        self.board = np.zeros((3,3),int)        
        
        # This is the turn number. It starts at 1 and can go as high as 9
        self.turnNum = 1
        
        # This is whose turn it is; it can be either 1 or -1. This can be deduced from other properties, but I find it 
        # useful to have as a property by itself
        self.whoseTurn = 1
        
        # This is the list of moves played this game. The moves are recorded as a number from 1 to 9, corresponding
        # to a keyboard number pad. Like so:
        # 7|8|9
        # 4|5|6
        # 1|2|3
        # Thus, if the first player took the center with the first move, and his opponent took the top left, the move
        # list would contain [5, 7]
        self.moveList = []
        
        # This indicates whether or not someone has won. It can be -1, 1, or 0. Obviously, it spends most of its time at
        # 0.
        self.winState = 0;
                
        # This indicates whether the game should be printed for the player to see. When playing lots of games, it should be off.
        # It can have values of 0 or 1
        self.printMode = printMode
        
    def printBoard(self):
        # This is a fairly kludgey means of printing the current board state. Eventually, I want to replace this with an
        # actual GUI. BUT IT IS NOT THIS DAY!!
        print("+-+-+-+")
        for xx in self.board:
            print("|", end = "") 
            cow = 0
            for yy in xx:
                if yy == 0:
                    print(" ", end = "")
                elif yy == 1:
                    print("x", end = "")
                else:
                    print("o", end = "")                    
                cow = cow + 1
                if cow == 3:
                    print("|\n+-+-+-+")
                else:
                    print("|", end = "") 
        print("\n")
    
    def makeMove(self, theMove):
        # This method is how you make moves in the game. You merely specify the location you wish to make your move; as
        # stated above in the properties, the move number corresponds to your keyboard number pad
        
        # I really should make this section into its own function. This is where I convert the move number into row and
        # column numbers. 
        # Later Note: I did make it into its own function but I haven't 
        # implemented it here yet :P
        theMove = theMove - 1
        theRowList = [2,2,2,1,1,1,0,0,0]
        theColList = [0,1,2,0,1,2,0,1,2]
        theRow = theRowList[theMove]
        theCol = theColList[theMove]
        
        # First, we check to make sure the move is valid, i.e., that the chosen location is not occupied. If not, insult
        # the player.
        if self.board[theRow][theCol] != 0:
            print("Make a valid move, dummy")
        else:
            # If it's a valid move, do the following:
            # Modify the board state
            self.board[theRow][theCol] = self.whoseTurn
            
            # Increment turn number
            self.turnNum = self.turnNum + 1
            
            # Change whose turn it is
            self.whoseTurn = -self.whoseTurn
            
            # Add the chosen move to the move list
            self.moveList.append(theMove + 1)
            
            # Display the board state, if allowed
            if self.printMode:
                self.printBoard()
                
            # Check to see if anyone has won
            self.checkWin()
            
    def undoMove(self):
        # Sometimes, you want to not make that move you just made.
        theMove = self.moveList.pop(len(self.moveList)-1)-1
        theRowList = [2,2,2,1,1,1,0,0,0]
        theColList = [0,1,2,0,1,2,0,1,2]
        theRow = theRowList[theMove]
        theCol = theColList[theMove]
        self.board[theRow][theCol] = 0
        self.turnNum = self.turnNum - 1
        self.whoseTurn = -self.whoseTurn
        self.winState = 0
        if self.printMode:
            self.printBoard()
        
    def checkWin(self):
        linBoard = [];
        for xx in self.board:
            linBoard.extend(xx)
        winTypes = [[1,2,3],[4,5,6],[7,8,9],[1,4,7],[2,5,8],[3,6,9],[1,5,9],[7,5,3]]
        for xx in winTypes:
            theSum = 0;
            for yy in xx:
                theSum = theSum + linBoard[yy-1]
            if abs(theSum) == 3:
                self.winState = theSum/3
                break
        # Print if allowed
        if self.printMode:
            if self.winState == -1:
                print("O wins!")
            elif self.winState == 1:
                print("X wins!")
        
    def clearBoard(self):
        self.board[:] = 0
        self.moveList = []
        self.turnNum = 1
        self.whoseTurn = 1
        self.winState = 0
        
    def togglePrintMode(self):
        self.printMode = 1 - self.printMode
