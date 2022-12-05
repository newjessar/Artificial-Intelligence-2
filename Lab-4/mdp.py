### code for representing/solving an MDP

import random as rd
import numpy as np
import numpy
import time
numpy.warnings.filterwarnings('ignore')
np.warnings.filterwarnings('ignore')
from problem_utils import *

class State :

    def __init__(self) :
        self.utility = 0.0
        self.reward = 0.0
        ### an action maps to a list of probability/state pairs
        self.transitions = {}
        self.actions = []
        self.policy = None
        self.coords = 0
        self.isGoal = False
        self.isWall = False
        self.id = 0
    # compute the expected utility of an action
    def computeEU(self, action) :
        return sum([trans[0] * trans[1].utility \
            for trans in self.transitions[action]])
        
    # to find the best action
    def selectBestAction(self) :
        best = max([(self.computeEU(a), a) for a in self.actions])
        return best[1]

class Map :
    def __init__(self) :
        self.states = {}
        self.stop_crit = 0.01
        self.gamma = 0.8 # discount
        self.n_rows = 0
        self.n_cols = 0
    
    class PrintType :
        ACTIONS = 0
        VALUES = 1


    ### you write this method
    def valueIteration(self) :
        start = time.time()
        ## reset every epoch after updating the actual utilities
        epoch = 0

        ### 1. initialize utilities to 0
        for s in self.states.values() :
            ## Exclude the goal state
            if not (s.isGoal == True):
                s.utility = 0.0
            
        while(True):
            ## count the process
            epoch +=1
            # monitor the change in the utilities
            max_utility_diff, old_util = 0.0, 0.0
            ## Bellmen utility equation
            for s in self.states.values():
                if s.isGoal:
                    pass
                else:
                    old_util = s.utility
                    stat_action_Util = s.computeEU(s.selectBestAction())
                    s.utility = s.reward + self.gamma * stat_action_Util
                    
                    ## Get the difference for all states
                    max_utility_diff += abs(s.utility - old_util)
                    
            ## When the states are less than the stop crit, break
            if self.stop_crit > max_utility_diff:
                print("Epochs:", epoch)
                end = time.time()
                print("Time", end - start)
                break

                    

    ### you write this method
    def policyIteration(self) :
        ## reset every epoch after updating the actual utilities
        epoch = 0
        start = time.time()
        ### 1. initialize random policy
        for s in self.states.values() :
            ## Initiate a random policy
            if not (s.isGoal == True):
                s.policy = rd.choice(s.actions)
                 
        ### 2 repeat policy iteration loop until policy is stable        
        while(True):
            ## count the process
            epoch +=1
            
            ## Calculate the utility linearly of the policy regarding each state 
            self.calculateUtilitiesLinear()
            
            ## monitor the change in the policy
            policy_differ = False
            
            ## Bellmen utility equation
            for s in self.states.values():
                current_a = 0.0
                if s.isGoal:
                    pass
                else:
                    current_a = 0.0
                    ## Calculate the the current action: U*T of s
                    for p_s_trans in s.transitions[s.policy]:
                        current_a += (p_s_trans[0] * p_s_trans[1].utility)
                        
                    ## Update the actions
                    for a in s.actions:
                        ## possible actions
                        pro_a = 0
                        ## Calculate the the possible action: U*T of s'
                        for p_s_trans in s.transitions[a]:
                            pro_a += (p_s_trans[0] * p_s_trans[1].utility)
                        ## Chose the best action
                        if current_a < pro_a:
                            current_a = pro_a
                            s.policy = a
                            policy_differ = True
                            
            ## Break the iteration after optioning the new policy
            if policy_differ == False:
                print("Epochs:", epoch)
                end = time.time()
                print("Time", end - start)
                break        

    
    def calculateUtilitiesLinear(self) :
        n_states = len(self.states)
        coeffs = numpy.zeros((n_states, n_states))
        ordinate = numpy.zeros((n_states, 1))
        for s in self.states.values() :
            row = s.id
            ordinate[row, 0] = s.reward
            coeffs[row,row] += 1.0
            if not s.isGoal :
                probs = s.transitions[s.policy]
                for p in probs :
                    col = p[1].id
                    coeffs[row,col] += -self.gamma * p[0]
        solution, _, _, _ = numpy.linalg.lstsq(coeffs, ordinate)
        for s in self.states.values() :
            if not s.isGoal :
                s.utility = solution[s.id, 0]
    
    def printActions(self) :
        self.printMaze(self.PrintType.ACTIONS)

    def printValues(self) :
        self.printMaze(self.PrintType.VALUES)
    
    def printMaze(self, print_type) :
        to_print = ":"
        for c in range(self.n_cols) :
            to_print = to_print + "--------:"
        to_print = to_print + '\n'
        for r in range(self.n_rows) :
            to_print = to_print + "|"
            for c in range(self.n_cols) :
                if self.states[(c,r)].isWall :
                    to_print = to_print + "        "
                else:
                    to_print = to_print + ' '
                    if self.states[(c,r)].isGoal :
                        to_print = to_print + \
                        "  {0: d}  ".format(int(self.states[(c,r)].utility))
                    else :
                        if print_type == self.PrintType.VALUES :
                            to_print = to_print + \
                                "{0: .3f}".format(self.states[(c,r)].utility)
                        elif print_type == self.PrintType.ACTIONS :
                            a = self.states[(c,r)].selectBestAction()
                            to_print = to_print + "  "
                            if a == 'left' :
                                to_print = to_print + "<<"
                            elif a == 'right' :
                                to_print = to_print + ">>"
                            elif a == 'up' :
                                to_print = to_print + "/\\"
                            elif a == 'down' :
                                to_print = to_print + "\\/"
                            to_print = to_print + "  "
                    to_print = to_print + ' '
                to_print = to_print + "|"
            to_print = to_print + '\n'
            to_print = to_print + ":"
            for c in range(self.n_cols) :
                to_print = to_print + "--------:"
            to_print = to_print + '\n'
        print(to_print)

def makeRNProblem() :
    """
    Creates the maze defined in Russell & Norvig. Utilizes functions defined
    in the problem_utils module.
    """


    walls = [(1,1)]
    actions = ['left', 'right','up','down']
    cols = 4
    rows = 3

    def filterState(oldState, newState) :
        if (newState[0] < 0 or newState[1] < 0 or newState[0] > cols - 1  or 
            newState[1] > rows - 1 or newState in walls) :
            return oldState
        else :
            return newState

    m = Map()
    m.n_cols = cols;
    m.n_rows = rows;
    for i in range(m.n_cols) :
        for j in range(m.n_rows) :
            m.states[(i,j)] = State()
            m.states[(i,j)].coords = (i,j)
            m.states[(i,j)].isGoal = False
            m.states[(i,j)].actions = actions
            m.states[(i,j)].id = j * m.n_cols + i
            m.states[(i,j)].reward = -0.04
                    
    m.states[(3,0)].isGoal = True
    m.states[(3,1)].isGoal = True
    m.states[(3,0)].utility = 1.0
    m.states[(3,1)].utility = -1.0

    m.states[(3,0)].reward = 1.0
    m.states[(3,1)].reward = -1.0

    for t in walls :
        m.states[t].isGoal = True
        m.states[t].isWall = True
        m.states[t].reward = 0.0
        m.states[t].utility = 0.0

    for s in m.states.items() :
        for a in actions :
            s[1].transitions[a] = [\
            (0.8, m.states[filterState(s[0], getSuccessor(s[0],a))]),  
            (0.1, m.states[filterState(s[0],getSuccessor(s[0], left(a)))]),
            (0.1, m.states[filterState(s[0], getSuccessor(s[0], right(a)))])]
    return m

def make2DProblem() :
    """
    Creates the larger maze described in the exercise. Utilizes functions 
    defined in the problem_utils module.
    """

    walls = [(1,1), (4,1), (5,1), (6,1),(7,1),(1,2), (7,2), (1,3), (5,3),
             (7,3), (1,4), (5,4), (7,4), (1,5), (5,5), (7,5), (1,6), (5,6), 
             (7,6), (1,7), (5,7), (7,7), (1,8), (3,8), (4,8), (5,8), 
             (7,8), (1,9)]
    actions = ['left', 'right','up','down']

    def filterState(oldState, newState) :
        if (newState[0] < 0 or newState[1] < 0 or newState[0] > 9 or 
            newState[1] > 9 or newState in walls) :
            return oldState
        else :
            return newState

    m = Map()
    m.n_cols = 10;
    m.n_rows = 10;
    for i in range(m.n_cols) :
        for j in range(m.n_rows) :
            m.states[(i,j)] = State()
            m.states[(i,j)].coords = (i,j)
            m.states[(i,j)].isGoal = False
            m.states[(i,j)].actions = actions
            m.states[(i,j)].id = j * 10 + i
            m.states[(i,j)].reward = -0.04
                    
    m.states[(0,9)].isGoal = True
    m.states[(9,9)].isGoal = True
    m.states[(9,0)].isGoal = True

    m.states[(0,9)].utility = 1.0
    m.states[(9,9)].utility = -1.0
    m.states[(9,0)].utility = 1.0

    m.states[(0,9)].reward = 1.0
    m.states[(9,9)].reward = -1.0
    m.states[(9,0)].reward = 1.0

    for t in walls :
        m.states[t].isGoal = True
        m.states[t].isWall = True
        m.states[t].utility = 0.0
        m.states[t].reward = 0.0


    for s in m.states.items() :
        for a in actions :
            s[1].transitions[a] = [\
            (0.7, m.states[filterState(s[0], getSuccessor(s[0],a))]),  
            (0.1, m.states[filterState(s[0], getSuccessor(s[0], opposite(a)))]), 
            (0.1, m.states[filterState(s[0],getSuccessor(s[0], left(a)))]),
            (0.1, m.states[filterState(s[0], getSuccessor(s[0], right(a)))])]

    return m
