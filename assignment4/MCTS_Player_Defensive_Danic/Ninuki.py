#!/usr/bin/python3
# Set the path to your python3 above

"""
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller
"""
from gtp_connection import GtpConnection, format_point
from board_base import DEFAULT_SIZE
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine
import time
import copy
import random
import signal
import numpy as np
import pickle
import math
from sys import getsizeof
from board_base import (
    BLACK, WHITE, EMPTY, BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    point_to_coord,
    opponent
)


class A4SubmissionPlayer(GoEngine):
    def __init__(self) -> None:
        """
        Starter code for assignment 4
        """
        GoEngine.__init__(self, "Go0", 1.0)
        self.time_limit = 1
        signal.signal(signal.SIGALRM, handler)
        self.dict = {}

    def get_move(self, board: GoBoard, colour: GO_COLOR, debug=False) -> GO_POINT:
        """
        Returns an availble move for the chosen colour according to a simulation player.
        ("debug" only used for debugging purposes)
        """
        if debug:
            self.activate_alarm()
            player = SimulationPlayer()
            temp_board = copy.deepcopy(board)
            move, self.dict = player.genmove(temp_board, colour, self.dict)
            self.deactivate_alarm()
            return move
        
        else:
            try:
                self.activate_alarm()
                player = SimulationPlayer()
                temp_board = copy.deepcopy(board)
                move, self.dict = player.genmove(temp_board, colour, self.dict)
                self.deactivate_alarm()
                return move
            
            except:
                self.deactivate_alarm()
                return GoBoardUtil.generate_random_move(board, colour, False)
    
    def get_moves(self, board: GoBoard) -> GO_POINT:
        """
        Added to keep backwards compatibility with assignment3
        """
        player = SimulationPlayer()
        temp_board = copy.deepcopy(board)
        return player.get_rule_based_moves(temp_board)

    def set_time_limit(self, time_limit):
        """
        Sets the maximum time limit the player has to choose a move.
        """
        self.time_limit = time_limit
    
    def activate_alarm(self):
        """
        Activates an alarm according to the time limit.
        """
        signal.setitimer(signal.ITIMER_REAL, self.time_limit - 0.05)
    
    def deactivate_alarm(self):
        """
        Deactivates the alarm.
        """
        signal.setitimer(signal.ITIMER_REAL, 0)

class SimulationPlayer(object):
    """
    A simulation player that uses MCTS and UCT to choose a move.
    """
    def __init__(self):
        self.UCT_dict:dict = {}
    
    def UCT(self, move:GO_POINT, state:GoBoard, colour:GO_COLOR, key_parent:str):
        """
        Implements the UCT algorithm to choose the next node to explore within the tree.
        """
        temp_key = key_parent + "_" + str(move)

        self.boards.setdefault(temp_key, [])
        if len(self.boards[temp_key]) == 0:
            self.boards[temp_key] = state.get_board_after_move(move, colour)
        board = self.boards[temp_key]

        self.keys.setdefault(temp_key, None)
        if self.keys[temp_key] == None:
            key = str(int.from_bytes(board.tobytes(), byteorder='big')) + str(state.score(BLACK))+ "_" + str(state.score(WHITE))+ "_" + str(colour)
            self.keys[temp_key] = key
        key = self.keys[temp_key]
        self.UCT_dict[key_parent][4].add(key)

        # [num_wins, num_explored, heuristic_mat, moves, children]
        self.UCT_dict.setdefault(key, [0, 0, [], [], set()])
        if len(self.UCT_dict[key_parent][2]) == 0:
            self.UCT_dict[key_parent][2] = state.board_state_heuristics(colour)

        coord = point_to_coord(move, state.size)
        exploitation = 1
        knowledge = math.sqrt(1 / ((self.UCT_dict[key][1] + 1))) * self.UCT_dict[key_parent][2][coord[0], coord[1]]
        exploration = 1
        C = 1
        if self.UCT_dict[key][1] > 0 and self.UCT_dict[key_parent][1] > 0:
            exploitation = self.UCT_dict[key][0] / self.UCT_dict[key][1]
            exploration = C * math.sqrt(math.log(self.UCT_dict[key_parent][1]) / self.UCT_dict[key][1])

        return exploitation + knowledge + exploration

    def select_move(self, move, key_parent):
        """
        A heuristic function to select a move once simulations have stopped.
        """
        key = self.keys[key_parent + "_" + str(move)]
        wins = self.UCT_dict[key][0]
        sims = self.UCT_dict[key][1]
        coord = point_to_coord(move, self.size)
        knowledge = math.sqrt(1 / ((self.UCT_dict[key][1] + 1))) * self.UCT_dict[key_parent][2][coord[0], coord[1]]
        if sims > 0:
            rate = wins / sims
        else:
            rate = 0
        return rate + knowledge
    
    def remove_entries(self, parent_key):
        """
        Removes all entries, including children, starting from the parent.
        """
        try:
            #for key in self.children[parent_key]:
            for key in self.UCT_dict[parent_key][4]:
                self.remove_entries(key)
            
            self.UCT_dict.pop(parent_key)
        except:
            pass
    
    def make_new_dict(self, parent_key, new_dict):
        """
        Makes a new dictionary/graph/tree that contains all nodes and children nodes from the provided parent node.
        """
        new_dict[parent_key] = self.UCT_dict[parent_key]

        for key in self.UCT_dict[parent_key][4]:
            try:
                self.make_new_dict(key)
            except:
                pass

    def genmove(self, state:GoBoard, colour:GO_COLOR, dictionary):
        """
        Generates a move according to the simulation player.
        """
        self.colour = colour
        self.size = state.size
        game_end, _ = state.terminal
        assert not game_end

        self.UCT_dict:dict = dictionary
        self.boards = {}
        #moves_init = self.get_rule_based_moves(state)
        moves_init = self.get_moves_prevent_win(state)
        #moves_init = state.get_empty_points()
        
        self.keys = {}
        key_parent = str(int.from_bytes(state.board2d.tobytes(), byteorder='big')) + str(state.score(BLACK))+ "_" + str(state.score(WHITE))+ "_" + str(colour)

        # [num_wins, num_explored, heuristic matrix, possible moves, children]
        self.UCT_dict.setdefault(key_parent, [0, 0, [], moves_init.copy(), set()])
        self.UCT_dict[key_parent][2] = state.board_state_heuristics(colour)
        new_dict = {}
        self.make_new_dict(key_parent, new_dict)
        self.UCT_dict = new_dict

        depth = state.moveNumber()

        score = 0
        try:
            while True:
                # create/use the UCT Tree according to the UCT sorting heuristic function.
                parents = [key_parent]
                key_curr = key_parent
                move = None

                while (self.UCT_dict[key_curr][1] > 0) and not state.terminal[0]:
                    # play move for player
                    moves = self.UCT_dict[key_curr][3]
                    random.shuffle(moves)
                    moves = sorted(moves, key=lambda x: self.UCT(x, state, state.current_player, key_curr), reverse=True)
                    move = moves[0]
                    state.play_move(move, state.current_player)
                    if state.terminal[0]:
                        break
                    
                    # play move for opponent
                    temp_key = key_curr + "_" + str(move)
                    self.keys.setdefault(temp_key, None)
                    if self.keys[temp_key] == None:
                        key = str(int.from_bytes(state.board2d.tobytes(), byteorder='big')) + str(state.score(BLACK))+ "_" + str(state.score(WHITE))+ "_" + str(colour)
                        self.keys[temp_key] = key
                    key_curr = self.keys[temp_key]
                    self.UCT_dict.setdefault(key_curr, [0, 0, [], [], set()])
                    if len(self.UCT_dict[key_curr][3]) == 0:
                        self.UCT_dict[key_curr][3] = self.get_rule_based_moves(state)  
                    move = random.choice(self.UCT_dict[key_curr][3])
                    state.play_move(move, state.current_player)
                    parents.append(key_curr)

                    # choose move for player
                    temp_key = key_curr + "_" + str(move)
                    self.keys.setdefault(temp_key, None)
                    if self.keys[temp_key] == None:
                        key = str(int.from_bytes(state.board2d.tobytes(), byteorder='big')) + str(state.score(BLACK))+ "_" + str(state.score(WHITE))+ "_" + str(colour)
                        self.keys[temp_key] = key
                    key_curr = self.keys[temp_key]
                    self.UCT_dict.setdefault(key_curr, [0, 0, [], [], set()])
                    if len(self.UCT_dict[key_curr][3]) == 0:
                        #self.UCT_dict[key_curr][3] = self.get_rule_based_moves(state)
                        #self.UCT_dict[key_curr][3] = state.get_empty_points()
                        self.UCT_dict[key_curr][3] = self.get_moves_prevent_win(state)
                    parents.append(key_curr)

                # run simulations for chosen move
                if not state.terminal[0]:
                    moves = self.UCT_dict[key_curr][3]
                    random.shuffle(moves)
                    moves = sorted(moves, key=lambda x: self.UCT(x, state, state.current_player, key_curr), reverse=True)
                    move = moves[0]
                    state.play_move(move, state.current_player)

                    temp_key = key_curr + "_" + str(move)
                    self.keys.setdefault(temp_key, None)
                    if self.keys[temp_key] == None:
                        key = str(int.from_bytes(state.board2d.tobytes(), byteorder='big')) + str(state.score(BLACK))+ "_" + str(state.score(WHITE))+ "_" + str(colour)
                        self.keys[temp_key] = key
                    key_prev = key_curr
                    key_curr = self.keys[temp_key]
                    parents.append(key_curr)
                    self.UCT_dict.setdefault(key_curr, [0, 0, [], [], set()])
                    

                    for _ in range(35):
                        score = self.simulate(state, move, key_prev)
                        for parent in parents:
                            self.UCT_dict[parent][0] += score
                            self.UCT_dict[parent][1] += 1
                else:
                    # if the terminal state has already been explored, increment score instead of doing simulations
                    if state.terminal[1] == self.colour:
                        score = 1
                    elif state.terminal[1] == EMPTY:
                        score = 0.5
                    else:
                        score = 0
                    for _ in range(3):
                        for parent in parents:
                            self.UCT_dict[parent][0] += score
                            self.UCT_dict[parent][1] += 1
                
                # reset board
                while state.depth > depth:
                    state.undo()
                state.update_board2d()

        # Once time limit has been reached, choose move
        except OSError:
            random.shuffle(moves_init)
            moves_init = sorted(moves_init, key=lambda x: self.select_move(x, key_parent), reverse=True)
            best = moves_init[0]
        
        # Make sure move is in available moves
        try:
            assert best in moves_init
        except:
            best = random.choice(moves_init)
        
        # Return the chosen move, and the dictionary/tree to store for next move.
        return best, self.UCT_dict

    def simulate(self, state:GoBoard, move, key_prev):
        """
        Simulate a game using MCTS and a random move policy
        """
        depth = 5
        game_end, col = state.terminal
        init_depth = state.moveNumber()
        
        i = 0
        while not ((game_end) or (i >= depth)):
            i += 1
            colour = state.current_player
            move = random.choice(state.get_empty_points())
            state.play_move_no_board2d(move, colour)
            
            game_end, col = state.terminal
        
        # if terminal state reached, add score accordingly
        if game_end:
            if col == self.colour:
                score = 1
            elif col == EMPTY:
                score = 0.5
            else:
                score = 0
        # otherwise, use heuristic value
        else:
            score = state.heuristic_eval(self.colour)
        
        # reset board
        while state.depth > init_depth:
            state.undo()

        return score
    
    def get_rule_based_moves(self, state:GoBoard):
        """
        Returns moves according to a rule-based policy
        """
        colour = state.current_player

        points = state.immediate_win(colour)
        if len(points) != 0:
            return points
        points = state.block_win(colour)
        if len(points) != 0:
            return points
        points = state.open_four(colour)
        points += state.block_open_four(colour)
        if len(points) != 0:
            # remove duplicate points
            points = list(dict.fromkeys(points))
            return points
        points = state.captures(colour)
        if len(points) != 0:
            return points

        empty_points = state.get_empty_points()
        return empty_points

    def get_moves_prevent_win(self, state:GoBoard):
            """
            Returns moves according to a rule-based policy,
            but tries to only limit the moveset in cases where a win in imminent.
            """
            colour = state.current_player

            points = state.immediate_win(colour)
            if len(points) != 0:
                return points
            points += state.block_win(colour)
            if len(points) != 0:
                return points

            empty_points = state.get_empty_points()
            return empty_points

def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(A4SubmissionPlayer(), board)
    con.start_connection()

def handler(signum, frame):
    """
    Raises an error code when the timer is up.
    """
    raise OSError("Used too much time")

if __name__ == "__main__":
    run()
