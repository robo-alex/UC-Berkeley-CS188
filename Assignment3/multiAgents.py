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
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        newFood = newFood.asList()
        ghostPos = [(G.getPosition()[0], G.getPosition()[1]) for G in newGhostStates]
        scared = min(newScaredTimes) > 0

        if not scared and (newPos in ghostPos):
            return -1.0

        if newPos in currentGameState.getFood().asList():
            return 1

        closestFoodDist = sorted(newFood, key=lambda F_dis: util.manhattanDistance(F_dis, newPos))
        closestGhostDist = sorted(ghostPos, key=lambda G_dis: util.manhattanDistance(G_dis, newPos))

        F_Dist = lambda F_D: util.manhattanDistance(F_D, newPos)
        G_Dist = lambda G_D: util.manhattanDistance(G_D, newPos)

        return 1.0 / F_Dist(closestFoodDist[0]) - 1.0 / G_Dist(closestGhostDist[0])
        # return successorGameState.getScore()

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
        Pac_val = -999999.0
        Pac_state = Directions.STOP

        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            next_val = self.getValue(next_state, 0, 1)

            if next_val > Pac_val:
                Pac_val = next_val
                Pac_state = action
        return Pac_state

    def getValue(self, gameState, currentDepth, agentIndex):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            value = self.evaluationFunction(gameState)
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:
            return self.Pac_val(gameState,currentDepth)
        else:
            return self.Ghost_val(gameState,currentDepth,agentIndex)

    def Pac_val(self, gameState, currentDepth):
        Pac_val = -999999.0
        for action in gameState.getLegalActions(0):
            Pac_val = max(Pac_val, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return Pac_val

    def Ghost_val(self, gameState, currentDepth, agentIndex):
        Ghost_val = 999999.0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                Ghost_val = min(Ghost_val, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0))
            else:
                Ghost_val = min(Ghost_val, self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex+1))
        return Ghost_val

        util.raiseNotDefined()

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"
        value = -999999.0
        alpha = -999999.0
        beta = 999999.0
        Pac_state = Directions.STOP

        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            value = max(value,self.getValue(next_state, 0, 1, alpha,beta))

            if value > alpha:
                alpha = value
                Pac_state = action

        return Pac_state

    def getValue(self, gameState, currentDepth, agentIndex,alpha,beta):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:
            return self.Pac_val(gameState,currentDepth,alpha,beta)
        else:
            return self.Ghost_val(gameState,currentDepth,agentIndex,alpha,beta)

    def Pac_val(self, gameState, currentDepth,alpha,beta):
        value=-999999.0
        for action in gameState.getLegalActions(0):
            value=max(value,self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1, alpha, beta))
            if value>beta:
                return value
            alpha=max(value,alpha)
        return value

    def Ghost_val(self, gameState, currentDepth, agentIndex,alpha,beta):
        value=999999.0
        if agentIndex == gameState.getNumAgents()-1:
            depth = currentDepth + 1
            index = 0
        else:
            depth = currentDepth
            index = agentIndex + 1
        for action in gameState.getLegalActions(agentIndex):
            value = min(value,self.getValue(gameState.generateSuccessor(agentIndex, action), depth, index,alpha,beta))
            if value < alpha:
                return value
            beta = min(value,beta)
        return value

        util.raiseNotDefined()

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
        Pac_val = -999999.0
        Pac_state = Directions.STOP

        for action in gameState.getLegalActions(0):
            next_state = gameState.generateSuccessor(0, action)
            next_val = self.getValue(next_state, 0, 1)

            if next_val > Pac_val:
                Pac_val = next_val
                Pac_state = action
        return Pac_state

    def getValue(self, gameState, currentDepth, agentIndex):

        if currentDepth == self.depth or gameState.isWin() or gameState.isLose():
            value = self.evaluationFunction(gameState)
            return self.evaluationFunction(gameState)

        elif agentIndex == 0:
            return self.Pac_val(gameState,currentDepth)

        else:
            return self.Ghost_val(gameState,currentDepth,agentIndex)


    def Pac_val(self, gameState, currentDepth):
        Pac_val = -999999.0
        for action in gameState.getLegalActions(0):
            Pac_val = max(Pac_val, self.getValue(gameState.generateSuccessor(0, action), currentDepth, 1))
        return Pac_val

    def Ghost_val(self, gameState, currentDepth, agentIndex):
        Ghost_val = 0
        for action in gameState.getLegalActions(agentIndex):
            if agentIndex == gameState.getNumAgents() - 1:
                Ghost_val += self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth + 1, 0)
            else:
                Ghost_val += self.getValue(gameState.generateSuccessor(agentIndex, action), currentDepth, agentIndex + 1)
        return Ghost_val / len(gameState.getLegalActions(agentIndex))

        util.raiseNotDefined()

def betterEvaluationFunction(currentGameState):
    """
    Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
    evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    newPos = currentGameState.getPacmanPosition()
    newFood = currentGameState.getFood()
    newGhostStates = currentGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    foodL = newFood.asList()
    nearestP = [manhattanDistance(newPos, x) for x in foodL]
    nearestG = min([manhattanDistance(newPos, x.getPosition()) for x in newGhostStates])
    Pdis = 0
    if len(nearestP) != 0:
        Pdis = min(nearestP)

    
    "*** YOUR CODE HERE ***"
    currentPos = currentGameState.getPacmanPosition()
    Dis_food = [+float('inf')]
    for foodPos in currentGameState.getFood().asList():
        Dis_food.append(util.manhattanDistance(currentPos, foodPos)) 

    Dis_G = [1]
    for ghostPos in currentGameState.getGhostPositions():
        Dis_G.append(util.manhattanDistance(currentPos, ghostPos))

    return 0.5 * currentGameState.getScore() + 1.5 / min(Dis_food) - 1.0 / max(Dis_G) + 2.0 / (currentGameState.getNumFood() + 1) + 10.0/(len(currentGameState.getCapsules()) + 1)
    
    util.raiseNotDefined()

# Abbreviation
better = betterEvaluationFunction
