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


import MCTS
import MCTSNode

SIMULATION_COUNT = 300
TIME_TO_SIMULATE = 58

MID_GAME_THRESHOLD = 49  # number of open spaces at which to switch from opener to midgame
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
        self.mcts = None

    def get_move(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """
        Uses the maximum allowable time to determine the best move. Has 3 modes determined by the number of open spaces:
        1. Opener: if there are more than 45 open spaces, then choose from the dataset of opening moves
        2. Midgame: if there are between 45 and 15 open spaces, then use MCTS/A-B to determine the best move
        3. Endgame: if there are less than 15 open spaces, then use alpha-beta pruning to determine the best move
        """

        # count the number of open spaces
        """open_spaces = len(board.get_empty_points())
        if open_spaces > MID_GAME_THRESHOLD:
            opener_move = self.opener(board, color)
            return opener_move
        elif open_spaces < ENDGAME_THRESHOLD:
            end_game_move = self.end_game(board, color)
            return end_game_move
        else:
            midgame_move = self.mid_game(board, color)
            return midgame_move
"""
        return self.mid_game(board, color)
    def opener(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """ Chooses a move from the dataset of opening moves.
        """
        pass

    def mid_game(self, board: GoBoard, color: GO_COLOR) -> GO_POINT:
        """MCTS"""


        if color == "b":
            color = BLACK
        else:
            color = WHITE

        if board.last_move == NO_POINT or board.last2_move == NO_POINT:
            root = MCTSNode.Node(None, None)
            root.board = board
            root.colour = opponent(color)
            self.mcts = MCTS.MCTS(root)
        else:
            self.mcts.tree_reuse(board.last_move, board.last2_move)
        current_time = time.time()
        while time.time() - current_time < TIME_TO_SIMULATE:
            self.mcts.run()

        return format_point(point_to_coord(self.mcts.get_root().get_max_child().move, board.size)).lower()

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
