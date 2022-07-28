# multiAgents.py
# --------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from util import manhattanDistance
from game import Directions
import random, util
import numpy as np

from game import Agent

class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.
    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.
        getAction chooses among the best options according to the evaluation function.
        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {NORTH, SOUTH, WEST, EAST, STOP}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.
        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.
        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.
        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        #print("new_food:", newFood)
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()#list
        ghostPos = []
        for G in newGhostStates:
            ghostPos_ = G.getPosition()[0], G. getPosition()[1]
            ghostPos.append(ghostPos_)
        #ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = newScaredTimes[0] > 0
        # if not new ScaredTimes new state is ghost: return lowest value
        if not scared and (newPos in ghostPos):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda gDist: util.manhattanDistance(gDist, newPos))

        fd = lambda fDis: util.manhattanDistance(fDis, newPos)

        gd = lambda gDis: util.manhattanDistance(gDis, newPos)

        return 1 / fd(closestFoodDist[0]) - 1/gd(closestGhostDist[0])
        "*** YOUR CODE HERE ***"

def scoreEvaluationFunction(currentGameState):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.
    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.
    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.
    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
    Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.
        Here are some method calls that might be useful when implementing minimax.
        gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1
        gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action
        gameState.getNumAgents():
        Returns the total number of agents in the game
        gameState.isWin():
        Returns whether or not the game state is a winning state
        gameState.isLose():
        Returns whether or not the game state is a losing state
        """
        "*** YOUR CODE HERE ***"
        PacmanValue = -100000.0
        PacmanAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)

            if nextValue > PacmanValue:
                PacmanValue = nextValue
                PacmanAction = action
        return PacmanAction



    def getValue(self, gameState, currentDepth, agentIndex):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            value = self.evaluationFunction(gameState)
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:
            return self.PacmanValue(gameState,currentDepth)

        else:
            return self.GhostValue(gameState,currentDepth,agentIndex)



    def PacmanValue(self, gameState, currentDepth):
        PacmanValue = -100000.0
        for action in gameState.getLegalActions(0):
            PacmanValue = max(PacmanValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return PacmanValue
    def GhostValue(self, gameState, currentDepth, agentIndex):
        GhostValue = 100000.0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents()-1:
                GhostValue = min(GhostValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0))
            else:
                GhostValue = min(GhostValue, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1))
        return GhostValue


class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value=-100000.0
        alpha=-100000.0
        beta=100000.0
        PacmanAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            value = max(value,self.getValue(nextState, 0, 1,alpha,beta))

            if value > alpha:
                alpha = value
                PacmanAction = action

        return PacmanAction



    def getValue(self, gameState, currentDepth, agentIndex,alpha,beta):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:
            return self.PacmanValue(gameState,currentDepth,alpha,beta)

        else:
            return self.GhostValue(gameState,currentDepth,agentIndex,alpha,beta)



    def PacmanValue(self, gameState, currentDepth,alpha,beta):
        value=-100000.0
        for action in gameState.getLegalActions(0):
            value=max(value,self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1,alpha,beta))
            if value>beta:
                return value
            alpha=max(value,alpha)
        return value

    def GhostValue(self, gameState, currentDepth, agentIndex,alpha,beta):
        value=100000.0
        if agentIndex == gameState.getNumAgents()-1:
            depth=currentDepth + 1
            index=0
        else:
            depth=currentDepth
            index=agentIndex+1
        for action in gameState.getLegalActions(agentIndex):
            value=min(value,self.getValue(gameState.generateSuccessor(agentIndex, action), depth, index,alpha,beta))
            if value<alpha:
                return value
            beta=min(value,beta)
        return value

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction
        All ghosts should be modeled as choosing uniformly at random from their
        legal moves.
        """
        "*** YOUR CODE HERE ***"
        PacmanValue = -100000.0
        PacmanAction = Directions.STOP

        for action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, action)
            nextValue = self.getValue(nextState, 0, 1)

            if nextValue > PacmanValue:
                PacmanValue = nextValue
                PacmanAction = action
        return PacmanAction



    def getValue(self, gameState, currentDepth, agentIndex):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            value = self.evaluationFunction(gameState)
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:
            return self.PacmanValue(gameState,currentDepth)

        else:
            return self.GhostValue(gameState,currentDepth,agentIndex)



    def PacmanValue(self, gameState, currentDepth):
        PacmanValue = -100000.0
        for action in gameState.getLegalActions(0):
            PacmanValue = max(PacmanValue, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return PacmanValue
    def GhostValue(self, gameState, currentDepth, agentIndex):
        GhostValue = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents()-1:
                GhostValue += self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0)
            else:
                GhostValue += self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1)
        return GhostValue/len(gameState.getLegalActions(agentIndex))

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).
    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"

    # Useful information you can extract from a GameState (pacman.py)
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    #print("new_food:", newFood)
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    "*** YOUR CODE HERE ***"
    newFood = newFood.asList()#list

    weightOfGhost=0
    for G in newGhostStates:
        ghostPos = G.getPosition()[0], G. getPosition()[1]
        distance=manhattanDistance(newPos,ghostPos)
        if distance>0:
            if G.scaredTimer>0:
                weightOfGhost+=1/distance
            else:
                weightOfGhost-=1/distance
    
    closestFoodDist = sorted(newFood, key=lambda fDist: util.manhattanDistance(fDist, newPos))
    weightOfFood=0
    fd = lambda fDis: util.manhattanDistance(fDis, newPos)
    

    if len(closestFoodDist)>0:
        if fd(closestFoodDist[0])>0:
            weightOfFood=1/fd(closestFoodDist[0])
        else:
            weightOfFood=0

    return currentGameState.getScore()+weightOfGhost+weightOfFood


# Abbreviation
better = betterEvaluationFunction