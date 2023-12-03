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
import bisect
import Tree

SIMULATION_COUNT = 5
TIME_TO_SIMULATE = 10


class A4SubmissionPlayer(GoEngine):
    def __init__(self) -> None:
        """
        Starter code for assignment 4
        """
        GoEngine.__init__(self, "Go0", 1.0)
        self.time_limit = 10
        self.solve_start_time = 0
        self.game_tree = None

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """
        Uses the maximum allowable time to determine the best move by simulating and then using alpha-beta pruning.
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
        if board.last2_move == NO_POINT:
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
                leaf.value = self.simulate(leaf.board, SIMULATION_COUNT)
                # efficiently insert the leaf node into leaves_to_expand so that leaves_to_expand is ordered by value
                # print(f"now inserting {leaf.name} with value {leaf.value}into leaves_to_expand")
                bisect.insort(leaves_to_expand, leaf, key=lambda node: node.value)
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
        for kiddo in root.children:
            print(f"{root.children[kiddo].name}'s value is {root.children[kiddo].value}. ")
        print(f"Choosing {best_move}. ")
        self.game_tree.save_tree(root, leaves_to_expand, leaves_to_simulate)
        return format_point(point_to_coord(best_move, board.size)).lower()

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
                if beta <= alpha:
                    break
            node.value = max_eval
            return max_eval
        else:
            min_eval = float('inf')
            for child in node.children:
                eval_child = self.alpha_beta_pruning(node.children[child], alpha, beta, True)
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
        Simulates a game from the current board position and returns the value of the game
        """
        value = 0
        for i in range(simulations):
            # get the result of random play to termination
            board_copy = board.copy()
            while not board_copy.is_terminal()[0]:
                moves = GoBoardUtil.generate_legal_moves(board_copy, board_copy.current_player)
                move = random.choice(moves)
                board_copy.play_move(move, board_copy.current_player)
                # board_copy.current_player = opponent(board_copy.current_player)

            # board is terminal so get value: (win for black: 1), (win for white: -1), (draw: 0)
            result = board_copy.is_terminal()[1]
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
