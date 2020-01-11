# TrainTTTNet
 Train a Tic-Tac-Toe-Playing Neural Network

In this project, I teach a Neural Network how to play Tic-Tac-Toe optimally, which in my case was defined as 'never loses.' https://medium.com/@carsten.friedrich/teaching-a-computer-to-play-tic-tac-toe-88feb838b5e3 was a heavy inspiration for concepts of reinforcement learning; however, no code was copied. 

A second goal of this project was to get practice with objects and classes, as I was not comfortable with object-oriented design. As a result, basically the entire framework of playing Tic-Tac-Toe, from boards to players to performance evaluators were given their own class and methods.

One cool aspect of the training was the existence of two neural networks simultaneously learning. The first, SheepNet, was the network being groomed for glory -- it's goal was to learn to play optimal Tic-Tac-Toe. LionNet, on the other hand, existed only to hunt SheepNet; it trained to find weaknesses in SheepNet's play and POUND THEM RUTHLESSLY until SheepNet learned to defend itself.
