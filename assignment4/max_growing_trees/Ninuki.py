#!/usr/bin/python3
# Set the path to your python3 above

"""
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller
"""
from gtp_connection import GtpConnection, format_point, point_to_coord
from board_base import DEFAULT_SIZE, GO_POINT, GO_COLOR
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
import bisect

SIMULATION_COUNT = 100
TIME_TO_SIMULATE = 5


class A4SubmissionPlayer(GoEngine):
    def __init__(self) -> None:
        """
        Starter code for assignment 4
        """
        GoEngine.__init__(self, "Go0", 1.0)
        self.time_limit = 10
        self.solve_start_time = 0

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """
        Uses the maximum allowable time to determine the best move by simulating and then using alpha-beta pruning.
        """

        # leaves_to_simulate is a list of leaf nodes that have not been simulated yet
        # leaves_to_expand is an ordered list of tuples (leaf node, leaf node value), ordered by leaf node value
        leaves_to_simulate = []
        leaves_to_expand = []

        # build tree by creating root node and adding it to the leaves dictionary
        root = TreeNode.Node()
        root.board = board.copy()
        root.color_to_play = color
        root.name = "root"
        root.value = None
        root.parent = None
        leaves_to_simulate.append(root)

        # start timer
        self.solve_start_time = time.time()

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

        while time.time() - self.solve_start_time < TIME_TO_SIMULATE:
            if leaves_to_simulate:  # if there are leaves to simulate, then simulate
                leaf = leaves_to_simulate.pop()
                if leaf.board.get_empty_points().size == 0:
                    leaf.value = leaf.board.score()[0]
                    continue
                leaf.value = self.simulate(leaf.board, SIMULATION_COUNT)
                # efficiently insert the leaf node into leaves_to_expand so that leaves_to_expand is ordered by value
                bisect.insort(leaves_to_expand, (leaf, leaf.value), key=lambda x: x[1])
            else:  # else there are no leaves to simulate, so expand
                if leaves_to_expand:  # if there are leaves to expand, then expand
                    if leaves_to_expand[0][0].color_to_play == color:  # if it is our turn to play, then expand the leaf with the highest value
                        leaf = leaves_to_expand.pop()[0]
                    else:  # else it is our opponent's turn to play, so expand the leaf with the lowest value
                        leaf = leaves_to_expand.pop(0)[0]
                    moves = GoBoardUtil.generate_legal_moves(leaf.board, leaf.color_to_play)
                    for move in moves:
                        child = leaf.add_child(move, None)
                        child.board = leaf.board.copy()
                        child.board.play_move(move, child.color_to_play)
                        child.color_to_play = opponent(child.color_to_play)
                        bisect.insort(leaves_to_simulate, child, key=lambda x: x.value)
                else:  # else there are no leaves to expand, so break
                    break

        # once the timer has exceeded TIME_TO_SIMULATE, use alpha-beta pruning to update the value of
        # nodes in the tree

        self.alpha_beta_pruning(root, float('-inf'), float('inf'), True)
        best_move = root.get_max_child().name
        return best_move

    def alpha_beta_pruning(self, node, alpha, beta, maximizing_player):
        """
        Uses alpha-beta pruning to determine the value of all important nodes.
        """
        if not node.children:
            return node.value

        if maximizing_player:
            max_eval = float('-inf')
            for child in node.children:
                eval_child = self.alpha_beta_pruning(child, alpha, beta, False)
                max_eval = max(max_eval, eval_child)
                alpha = max(alpha, eval_child)
                if beta <= alpha:
                    break
            node.value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval_child = self.alpha_beta_pruning(child, alpha, beta, True)
                min_eval = min(min_eval, eval_child)
                beta = min(beta, eval_child)
                if beta <= alpha:
                    break
            node.value = min_eval
            return min_eval

    def set_time_limit(self, time_limit):
        self.time_limit = time_limit

    def simulate(self, board, simulations):
        """
        Simulates a game from the current board position and returns the value of the game.
        """
        for i in range(simulations):
            self.board = board.copy()
            while not self.board.is_terminal()[0]:
                moves = GoBoardUtil.generate_legal_moves(self.board, self.board.current_player)
                move = random.choice(moves)
                self.board.play_move(move, self.board.current_player)
                self.board.current_player = opponent(self.board.current_player)
        return self.board.score()[0]


def run() -> None:
    """
    start the gtp connection and wait for commands.
    """
    board: GoBoard = GoBoard(DEFAULT_SIZE)
    con: GtpConnection = GtpConnection(A4SubmissionPlayer(), board)
    con.start_connection()


if __name__ == "__main__":
    run()
