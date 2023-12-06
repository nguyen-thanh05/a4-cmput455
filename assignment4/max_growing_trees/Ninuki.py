#!/usr/bin/python3
# Set the path to your python3 above

"""
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller
"""
from gtp_connection import GtpConnection, format_point, point_to_coord
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR, NO_POINT
from board import GoBoard
from board_util import GoBoardUtil
from engine import GoEngine
import time
import random
from board_base import (
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    GO_COLOR, GO_POINT,
    PASS,
    MAXSIZE,
    coord_to_point,
    opponent
)
import numpy as np
import TreeNode
import Tree

SIMULATION_COUNT = 400  # number of simulations to run for each leaf node
TIME_TO_SIMULATE = 55  # number of seconds to simulate for
MID_MOVE = 36  # board square to play in the middle of the board

MID_GAME_THRESHOLD = 48  # number of open spaces at which to switch from opener to midgame
ENDGAME_THRESHOLD = 0  # number of open spaces at which to switch from midgame to endgame


class A4SubmissionPlayer(GoEngine):
    def __init__(self) -> None:
        """
        Starter code for assignment 4
        """
        GoEngine.__init__(self, "Go0", 1.0)
        self.time_limit = 10
        self.solve_start_time = 0
        self.game_tree = None
        self.simulation_table = {}

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """
        Uses the maximum allowable time to determine the best move. Has 3 modes determined by the number of open spaces:
        1. Opener: if there are more than 45 open spaces, then choose from the dataset of opening moves
        2. Midgame: if there are between 45 and 15 open spaces, then use MCTS/A-B to determine the best move
        3. Endgame: if there are less than 15 open spaces, then use alpha-beta pruning to determine the best move
        """

        # count the number of open spaces
        open_spaces = len(board.get_empty_points())
        if open_spaces > MID_GAME_THRESHOLD:
            opener_move = self.opener(board, color)
            return opener_move
        elif open_spaces < ENDGAME_THRESHOLD:
            """end_game_move = self.end_game(board, color)
            return end_game_move"""
            midgame_move = self.mid_game(board, color)
            return midgame_move
        else:
            midgame_move = self.mid_game(board, color)
            return midgame_move

    def opener(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """ Chooses a move from the dataset of opening moves.
        """
        self.mid_game(board, color)
        best_move = MID_MOVE
        return format_point(point_to_coord(best_move, board.size)).lower()

    def mid_game(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """ Uses MCTS/A-B to determine the best move.
        """

        # leaves_to_simulate is a list of leaf nodes that have not been simulated yet
        # leaves_to_expand is an ordered list of tuples (leaf node, leaf node value), ordered by leaf node value
        leaves_to_simulate = []
        leaves_to_expand = []

        if color == "w":
            given_color = WHITE
        else:
            given_color = BLACK

        # build tree by creating root node and adding it to the leaves dictionary
        if board.last2_move == NO_POINT or self.game_tree is None:
            root = TreeNode.Node()
            self.game_tree = Tree.Tree()
            root.board = board.copy()
            root.color_to_play = opponent(given_color)
            root.name = "root"
            root.value = None
            root.parent = None
            leaves_to_expand.append(root)
        else:
            root, leaves_to_expand, leaves_to_simulate = self.game_tree.load_tree(board.copy())

        # start timer
        self.solve_start_time = time.time()
        current_time = self.solve_start_time

        # while the timer has not exceeded TIME_TO_SIMULATE:
        # if the number of leaves to simulate is greater than 0:
        # ----- simulate -----
        # pop a leaf node from leaves_to_simulate
        # if the leaf node is terminal, update the leaf node's value and continue without expanding
        # simulate the game from the leaf node's board position
        # update the leaf node's value
        # insert the leaf node into leaves_to_expand so that leaves_to_expand is ordered by leaf node value
        # repeat until the number of leaves to simulate is 0
        # ----- expand -----
        # pop the leaf node with the highest value from leaves_to_expand if it is our turn to play, else pop the leaf
        # node with the lowest value
        # expand the leaf node by adding all legal moves to the tree
        # insert the new leaf nodes into leaves_to_simulate so that leaves_to_simulate is ordered by leaf node value
        # repeat until the timer has exceeded TIME_TO_SIMULATE

        sim_count = max(100, 800-((len(board.get_empty_points())//10)*100))  # increase simulations as game progresses
        # print(f"sim_count is {sim_count}.  ")

        while current_time - self.solve_start_time < TIME_TO_SIMULATE:
            current_time = time.time()
            # print(f"start time is {self.solve_start_time} ")
            # print(f"current time is {current_time} ")
            # print(f"elapsed time is {current_time - self.solve_start_time} ")
            # print(f"leaves to expand is:", leaves_to_expand)
            # print(f"leaves to simulate is", leaves_to_simulate)
            if leaves_to_simulate:  # if there are leaves to simulate, then simulate
                leaf = leaves_to_simulate.pop()
                # print(f"now simulating from {leaf.name}")
                if leaf.board.is_terminal()[0]:
                    # print(f"leaf {leaf.name} is terminal, not adding to expand")
                    result = leaf.board.is_terminal()[1]
                    if result == WHITE:
                        leaf.value = -1
                    elif result == BLACK:
                        leaf.value = 1
                    else:
                        leaf.value = 0
                    continue

                key = leaf.board.state_to_key()
                if key in self.simulation_table:
                    sim_value = self.simulate(leaf.board, int(sim_count//4))
                    leaf.value = (self.simulation_table[key] + sim_value)/2
                    self.simulation_table[key] = leaf.value
                else:
                    leaf.value = self.simulate(leaf.board, sim_count)
                    self.simulation_table[key] = leaf.value
                # efficiently insert the leaf node into leaves_to_expand so that leaves_to_expand is ordered by value
                # print(f"now inserting {leaf.name} with value {leaf.value}into leaves_to_expand")
                self.insort_right(leaves_to_expand, leaf, key=lambda node: node.value)
            else:  # else there are no leaves to simulate, so expand
                if leaves_to_expand:  # if there are leaves to expand, then expand
                    if leaves_to_expand[0].color_to_play == given_color:  # if it is our turn to play, then expand the leaf with the highest value
                        leaf = leaves_to_expand.pop()
                    else:  # else it is our opponent's turn to play, so expand the leaf with the lowest value
                        leaf = leaves_to_expand.pop(0)
                    # print(f"no leaf to simulate, so expanding {leaf.name}")
                    moves = GoBoardUtil.generate_legal_moves(leaf.board, leaf.color_to_play)
                    # print(moves)
                    for move in moves:
                        child = leaf.add_child(move, 0)
                        child.name = str(move)
                        child.color_to_play = opponent(leaf.color_to_play)  # assign the colours of the children
                        child.board = leaf.board.copy()
                        child.board.play_move(move, child.color_to_play)
                        leaves_to_simulate.append(child)
                else:  # else there are no leaves to expand, so break
                    break

        # once the timer has exceeded TIME_TO_SIMULATE, use alpha-beta pruning to update the value of
        # nodes in the tree

        if given_color == BLACK:
            maximizing_player = True
        else:
            maximizing_player = False
        self.alpha_beta_pruning(root, float('-inf'), float('inf'), maximizing_player)
        if maximizing_player:
            best_move = int(root.get_max_child().name)
        else:
            best_move = int(root.get_min_child().name)
        # for kiddo in root.children:
        #     print(f"{root.children[kiddo].name}'s value is {root.children[kiddo].value}. ")
        # print(f"Choosing {best_move}. ")
        self.game_tree.save_tree(root, leaves_to_expand, leaves_to_simulate)
        return format_point(point_to_coord(best_move, board.size)).lower()

    def bisect_right(self, a, x, lo=0, hi=None, key=None):
        if lo < 0:
            raise ValueError('lo must be non-negative')
        if hi is None:
            hi = len(a)
        # Note, the comparison uses "<" to match the
        # __lt__() logic in list.sort() and in heapq.
        if key is None:
            while lo < hi:
                mid = (lo + hi) // 2
                if x < a[mid]:
                    hi = mid
                else:
                    lo = mid + 1
        else:
            while lo < hi:
                mid = (lo + hi) // 2
                if x < key(a[mid]):
                    hi = mid
                else:
                    lo = mid + 1
        return lo

    def insort_right(self, a, x, lo=0, hi=None, *, key=None):
        """Insert item x in list a, and keep it sorted assuming a is sorted.

        If x is already in a, insert it to the right of the rightmost x.

        Optional args lo (default 0) and hi (default len(a)) bound the
        slice of a to be searched.
        """
        if key is None:
            lo = self.bisect_right(a, x, lo, hi)
        else:
            lo = self.bisect_right(a, key(x), lo, hi, key=key)
        a.insert(lo, x)

    def end_game(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """ Uses alpha-beta pruning to determine the best move.
        """
        pass

    def alpha_beta_pruning(self, node: TreeNode, alpha, beta, maximizing_player: bool):
        """
        Uses alpha-beta pruning to determine the value of all important nodes.
        """
        # print(f"node is {node}")
        if not node.children:
            return node.value

        if maximizing_player:
            max_eval = float('-inf')
            for child in node.children:
                eval_child = self.alpha_beta_pruning(node.children[child], alpha, beta, False)
                max_eval = max(max_eval, eval_child)
                alpha = max(alpha, eval_child)
                if eval_child >= beta:
                    break
            node.value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval_child = self.alpha_beta_pruning(node.children[child], alpha, beta, True)
                min_eval = min(min_eval, eval_child)
                beta = min(beta, eval_child)
                if eval_child <= alpha:
                    break
            node.value = min_eval
            return min_eval

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit

    def simulate(self, board, simulations):
        """
        Simulates a game from the current board position and returns the value of the game
        """
        value = 0
        for i in range(simulations):
            # get the result of random play to termination
            board_copy = board.copy()
            terminal = None

            while True:
                terminal = board_copy.is_terminal()
                if terminal[0]:
                    break
                moves = terminal[2]
                move = random.choice(moves)
                board_copy.play_move(move, board_copy.current_player)
                # board_copy.current_player = opponent(board_copy.current_player)

            # board is terminal so get value: (win for black: 1), (win for white: -1), (draw: 0)
            result = terminal[1]
            # print(f" simulation {i}'s result was: {result}")

            if result == WHITE:
                update = -1
            elif result == BLACK:
                update = 1
            else:
                update = 0

            # update value
            value += update/simulations

        return value


def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(A4SubmissionPlayer(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
