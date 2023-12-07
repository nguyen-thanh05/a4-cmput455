"""
board.py
Cmput 455 sample code
Written by Cmput 455 TA and Martin Mueller

Implements a basic Go board with functions to:
- initialize to a given board size
- check if a move is legal
- play a move

The board uses a 1-dimensional representation with padding
"""

import numpy as np
from typing import List, Tuple

from numpy import ndarray

from board_base import (
    board_array_size,
    coord_to_point,
    is_black_white,
    is_black_white_empty,
    opponent,
    where1d,
    BLACK,
    WHITE,
    EMPTY,
    BORDER,
    MAXSIZE,
    NO_POINT,
    PASS,
    GO_COLOR,
    GO_POINT,
)
from collections import deque

"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""

class AhoCorasickNode:
    def __init__(self):
        self.children = {}
        self.fail = None
        self.output = []


def build_ac_trie(patterns):
    root = AhoCorasickNode()

    for pattern in patterns:
        node = root
        for num in pattern:
            if num not in node.children:
                node.children[num] = AhoCorasickNode()
            node = node.children[num]
        node.output.append(pattern)

    queue = deque()
    for child in root.children.values():
        queue.append(child)
        child.fail = root

    while queue:
        current_node = queue.popleft()
        for num, child in current_node.children.items():
            queue.append(child)
            fail_node = current_node.fail
            while fail_node and num not in fail_node.children:
                fail_node = fail_node.fail
            if fail_node:
                child.fail = fail_node.children.get(num, root)
            else:
                child.fail = root
            child.output.extend(child.fail.output)

    return root


def aho_corasick_search(pos_array, ac_trie, patterns):
    current_node = ac_trie
    i = 0
    output = []

    for num in pos_array:
        while num not in current_node.children and current_node.fail:
            current_node = current_node.fail
        if num in current_node.children:
            current_node = current_node.children[num]
        for j in current_node.output:
            pattern_index = patterns.index(j)
            start_pos = i - len(j) + 1
            output.append([pattern_index, start_pos])

        i += 1
    if len(output) == 0:
        return [[-1, -1]]
    else:
        return output


IMMEDIATE_WIN_WHITE = [[WHITE, WHITE, WHITE, WHITE, EMPTY], [WHITE, WHITE, WHITE, EMPTY, WHITE],
                       [WHITE, WHITE, EMPTY, WHITE, WHITE], [WHITE, EMPTY, WHITE, WHITE, WHITE],
                       [EMPTY, WHITE, WHITE, WHITE, WHITE]]
IMMEDIATE_WIN_WHITE_EMPTY_OFFSET = [[4], [3], [2], [1], [0]]

IMMEDIATE_WIN_BLACK = [[BLACK, BLACK, BLACK, BLACK, EMPTY], [BLACK, BLACK, BLACK, EMPTY, BLACK],
                       [BLACK, BLACK, EMPTY, BLACK, BLACK], [BLACK, EMPTY, BLACK, BLACK, BLACK],
                       [EMPTY, BLACK, BLACK, BLACK, BLACK]]
IMMEDIATE_WIN_BLACK_EMPTY_OFFSET = [[4], [3], [2], [1], [0]]

WHITE_CAPTURE = [[WHITE, BLACK, BLACK, EMPTY],
                 [EMPTY, BLACK, BLACK, WHITE]]
WHITE_CAPTURE_EMPTY_OFFSET = [[3], [0]]

BLACK_CAPTURE = [[BLACK, WHITE, WHITE, EMPTY],
                 [EMPTY, WHITE, WHITE, BLACK]]
BLACK_CAPTURE_EMPTY_OFFSET = [[3], [0]]

OPEN_FOUR_WHITE = [[EMPTY, WHITE, WHITE, WHITE, EMPTY, EMPTY],
                   [EMPTY, WHITE, WHITE, EMPTY, WHITE, EMPTY],
                   [EMPTY, WHITE, EMPTY, WHITE, WHITE, EMPTY],
                   [EMPTY, EMPTY, WHITE, WHITE, WHITE, EMPTY]]
OPEN_FOUR_WHITE_EMPTY_OFFSET = [[4], [3], [2], [1]]

OPEN_FOUR_BLACK = [[EMPTY, BLACK, BLACK, BLACK, EMPTY, EMPTY],
                   [EMPTY, BLACK, BLACK, EMPTY, BLACK, EMPTY],
                   [EMPTY, BLACK, EMPTY, BLACK, BLACK, EMPTY],
                   [EMPTY, EMPTY, BLACK, BLACK, BLACK, EMPTY]]
OPEN_FOUR_BLACK_EMPTY_OFFSET = [[4], [3], [2], [1]]

immediate_win_white_trie = build_ac_trie(IMMEDIATE_WIN_WHITE)
immediate_win_black_trie = build_ac_trie(IMMEDIATE_WIN_BLACK)
white_capture_trie = build_ac_trie(WHITE_CAPTURE)
black_capture_trie = build_ac_trie(BLACK_CAPTURE)
open_four_white_trie = build_ac_trie(OPEN_FOUR_WHITE)
open_four_black_trie = build_ac_trie(OPEN_FOUR_BLACK)


class GoBoard(object):
    def __init__(self, size: int) -> None:
        """
        Creates a Go board of given size
        """
        assert 2 <= size <= MAXSIZE
        self.reset(size)
        self.black_captures = 0
        self.white_captures = 0
        self.depth = 0
        self.black_capture_history = []
        self.white_capture_history = []
        self.move_history = []
        self.rows = [[9, 10, 11, 12, 13, 14, 15], [17, 18, 19, 20, 21, 22, 23], [25, 26, 27, 28, 29, 30, 31], [33, 34, 35, 36, 37, 38, 39], [41, 42, 43, 44, 45, 46, 47], [49, 50, 51, 52, 53, 54, 55], [57, 58, 59, 60, 61, 62, 63]]
        self.cols = [[9, 17, 25, 33, 41, 49, 57], [10, 18, 26, 34, 42, 50, 58], [11, 19, 27, 35, 43, 51, 59], [12, 20, 28, 36, 44, 52, 60], [13, 21, 29, 37, 45, 53, 61], [14, 22, 30, 38, 46, 54, 62], [15, 23, 31, 39, 47, 55, 63]]
        self.diags = [[9, 18, 27, 36, 45, 54, 63], [10, 19, 28, 37, 46, 55], [11, 20, 29, 38, 47], [12, 21, 30, 39], [17, 26, 35, 44, 53, 62], [25, 34, 43, 52, 61], [33, 42, 51, 60], [33, 26, 19, 12], [41, 34, 27, 20, 13], [49, 42, 35, 28, 21, 14], [57, 50, 43, 36, 29, 22, 15], [58, 51, 44, 37, 30, 23], [59, 52, 45, 38, 31], [60, 53, 46, 39]]

    def add_two_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            self.black_captures += 2
        elif color == WHITE:
            self.white_captures += 2
    
    def get_captures(self, color: GO_COLOR) -> None:
        if color == BLACK:
            return self.black_captures
        elif color == WHITE:
            return self.white_captures
    
    def reset(self, size: int) -> None:
        """
        Creates a start state, an empty board with given size.
        """
        self.size: int = size
        self.NS: int = size + 1
        self.WE: int = 1
        self.last_move: GO_POINT = NO_POINT
        self.last2_move: GO_POINT = NO_POINT
        self.current_player: GO_COLOR = BLACK
        self.maxpoint: int = board_array_size(size)
        self.board: np.ndarray[GO_POINT] = np.full(self.maxpoint, BORDER, dtype=GO_POINT)
        self._initialize_empty_points(self.board)
        self.black_captures = 0
        self.white_captures = 0
        self.depth = 0
        self.black_capture_history = []
        self.white_capture_history = []
        self.move_history = []

    def copy(self) -> 'GoBoard':
        b = GoBoard(self.size)
        assert b.NS == self.NS
        assert b.WE == self.WE
        b.last_move = self.last_move
        b.last2_move = self.last2_move
        b.current_player = self.current_player
        assert b.maxpoint == self.maxpoint
        b.board = np.copy(self.board)
        b.black_captures = self.black_captures
        b.white_captures = self.white_captures
        b.depth = self.depth
        b.black_capture_history = self.black_capture_history.copy()
        b.white_capture_history = self.white_capture_history.copy()
        b.move_history = self.move_history.copy()
        return b

    def get_color(self, point: GO_POINT):
        return self.board[point]

    def pt(self, row: int, col: int):
        return coord_to_point(row, col, self.size)

    def is_legal(self, point: GO_POINT, color: GO_COLOR):
        """
        Check whether it is legal for color to play on point
        This method tries to play the move on a temporary copy of the board.
        This prevents the board from being modified by the move
        """
        if point == PASS:
            return True
        #board_copy: GoBoard = self.copy()
        #can_play_move = board_copy.play_move(point, color)
        #return can_play_move
        return self.board[point] == EMPTY

    def end_of_game(self):
        empty_points = self.get_empty_points()
        return empty_points.size == 0 or (self.last_move == PASS and self.last2_move == PASS), empty_points
           
    def get_empty_points(self):
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray):
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def play_move(self, point: GO_POINT, color: GO_COLOR):
        """
        Tries to play a move of color on the point.
        Returns whether or not the point was empty.
        """
        if self.board[point] != EMPTY:
            return False
        self.board[point] = color
        self.current_player = opponent(color)
        self.last2_move = self.last_move
        self.last_move = point
        O = opponent(color)
        offsets = [1, -1, self.NS, -self.NS, self.NS+1, -(self.NS+1), self.NS-1, -self.NS+1]
        bcs = []
        wcs = []
        for offset in offsets:
            if self.board[point+offset] == O and self.board[point+(offset*2)] == O and self.board[point+(offset*3)] == color:
                self.board[point+offset] = EMPTY
                self.board[point+(offset*2)] = EMPTY
                if color == BLACK:
                    self.black_captures += 2
                    bcs.append(point+offset)
                    bcs.append(point+(offset*2))
                else:
                    self.white_captures += 2
                    wcs.append(point+offset)
                    wcs.append(point+(offset*2))
        self.depth += 1
        self.black_capture_history.append(bcs)
        self.white_capture_history.append(wcs)
        self.move_history.append(point)
        return True
    
    def undo(self):
        self.board[self.move_history.pop()] = EMPTY
        self.current_player = opponent(self.current_player)
        self.depth -= 1
        bcs = self.black_capture_history.pop()
        for point in bcs:
            self.board[point] = WHITE
            self.black_captures -= 1
        wcs = self.white_capture_history.pop()
        for point in wcs:
            self.board[point] = BLACK
            self.white_captures -= 1
        if len(self.move_history) > 0:
            self.last_move = self.move_history[-1]
        if len(self.move_history) > 1:
            self.last2_move = self.move_history[-2]

    def neighbors_of_color(self, point: GO_POINT, color: GO_COLOR):
        """ List of neighbors of point of given color """
        nbc: List[GO_POINT] = []
        for nb in self._neighbors(point):
            if self.get_color(nb) == color:
                nbc.append(nb)
        return nbc

    def _neighbors(self, point: GO_POINT):
        """ List of all four neighbors of the point """
        return [point - 1, point + 1, point - self.NS, point + self.NS]

    def _diag_neighbors(self, point: GO_POINT) -> List:
        """ List of all four diagonal neighbors of point """
        return [point - self.NS - 1,
                point - self.NS + 1,
                point + self.NS - 1,
                point + self.NS + 1]

    def last_board_moves(self):
        """
        Get the list of last_move and second last move.
        Only include moves on the board (not NO_POINT, not PASS).
        """
        board_moves: List[GO_POINT] = []
        if self.last_move != NO_POINT and self.last_move != PASS:
            board_moves.append(self.last_move)
        if self.last2_move != NO_POINT and self.last2_move != PASS:
            board_moves.append(self.last2_move)
        return board_moves

    def full_board_detect_five_in_a_row(self):
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        Checks the entire board.
        """
        for point in range(self.maxpoint):
            c = self.board[point]
            if c != BLACK and c != WHITE:
                continue
            for offset in [(1, 0), (0, 1), (1, 1), (1, -1)]:
                i = 1
                num_found = 1
                while self.board[point + i * offset[0] * self.NS + i * offset[1]] == c:
                    i += 1
                    num_found += 1
                i = -1
                while self.board[point + i * offset[0] * self.NS + i * offset[1]] == c:
                    i -= 1
                    num_found += 1
                if num_found >= 5:
                    return c
        
        return EMPTY
    
    def detect_five_in_a_row(self):
        """
        Returns BLACK or WHITE if any five in a row is detected for the color
        EMPTY otherwise.
        Only checks around the last move for efficiency.
        """
        if self.last_move == NO_POINT or self.last_move == PASS:
            return EMPTY
        c = self.board[self.last_move]
        for offset in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            i = 1
            num_found = 1
            while self.board[self.last_move + i * offset[0] * self.NS + i * offset[1]] == c:
                i += 1
                num_found += 1
            i = -1
            while self.board[self.last_move + i * offset[0] * self.NS + i * offset[1]] == c:
                i -= 1
                num_found += 1
            if num_found >= 5:
                return c
        
        return EMPTY

    def is_terminal(self):
        """
        Returns: is_terminal, winner, open_points
        If the result is a draw, winner = EMPTY
        """
        winner = self.detect_five_in_a_row()
        if winner != EMPTY:
            # print("5 in a row detected")
            return True, winner, None
        elif self.get_captures(BLACK) >= 10:
            # print("10 captures detected")
            return True, BLACK, None
        elif self.get_captures(WHITE) >= 10:
            # print("10 captures detected")
            return True, WHITE, None
        else:
            e_o_g = self.end_of_game()
            if e_o_g[0]:
                # print("end of game")
                return True, EMPTY, None
            else:
                return False, EMPTY, e_o_g[1]

    def heuristic_eval(self):
        """
        Returns: a very basic heuristic value of the board
        Currently only considers captures
        """
        if self.current_player == BLACK:
            return (self.black_captures - self.white_captures) / 10
        else:
            return (self.white_captures - self.black_captures) / 10

    def state_to_str(self):
        state = np.array2string(self.board, separator='')
        state += str(self.current_player)
        state += str(self.black_captures)
        state += str(self.white_captures)
        return state

    def state_to_key(self):
        state = self.board.tobytes()
        return state, self.current_player, self.black_captures, self.white_captures


    def pattern_search(self, patterns, offsets, trie):
        moves = []
        for position_array in self.rows + self.cols + self.diags:
            search_output = aho_corasick_search(self.board[position_array], trie, patterns)
            for item in search_output:
                if item[0] != -1:  # A match was found
                    for index in offsets[item[0]]:
                        moves.append(position_array[item[1] + index])
        return moves

    def immediate_win_search(self, colour):
        immediate_win_moves = []
        if colour == WHITE:
            immediate_win_moves += self.pattern_search(IMMEDIATE_WIN_WHITE, IMMEDIATE_WIN_WHITE_EMPTY_OFFSET,
                                                       immediate_win_white_trie)
            if self.white_captures >= 8:
                immediate_win_moves += self.pattern_search(WHITE_CAPTURE, WHITE_CAPTURE_EMPTY_OFFSET,
                                                           white_capture_trie)
        elif colour == BLACK:
            immediate_win_moves += self.pattern_search(IMMEDIATE_WIN_BLACK, IMMEDIATE_WIN_BLACK_EMPTY_OFFSET,
                                                       immediate_win_black_trie)
            if self.black_captures >= 8:
                immediate_win_moves += self.pattern_search(BLACK_CAPTURE, BLACK_CAPTURE_EMPTY_OFFSET,
                                                           black_capture_trie)
        return immediate_win_moves

    def block_opponent_win_search(self, colour):
        opponent_colour = opponent(colour)
        opponent_win_moves = self.immediate_win_search(opponent_colour)
        block_moves = []

        # Block 4 in a row by capturing stones
        capturing_moves = self.capture_search(colour)

        for capture in capturing_moves:
            self.play_move(capture, colour)
            new_opponent_win = self.immediate_win_search(opponent_colour)
            if len(opponent_win_moves) > len(new_opponent_win):
                block_moves.append(capture)
            self.undo()

        # Prevent opponent from capturing 10 stones
        if colour == WHITE:
            opp_captures = self.black_captures
        elif colour == BLACK:
            opp_captures = self.white_captures
        if opp_captures >= 8:
            opp_capture_moves = self.capture_search(opponent_colour)
            if len(opp_capture_moves) > 0:
                block_moves += opp_capture_moves

        return list(set(block_moves + opponent_win_moves))

    def open_four_search(self, colour):
        open_four_moves = []
        if colour == WHITE:
            open_four_moves += self.pattern_search(OPEN_FOUR_WHITE, OPEN_FOUR_WHITE_EMPTY_OFFSET,
                                                   open_four_white_trie)
        elif colour == BLACK:
            open_four_moves += self.pattern_search(OPEN_FOUR_BLACK, OPEN_FOUR_BLACK_EMPTY_OFFSET,
                                                   open_four_black_trie)
        return open_four_moves

    def capture_search(self, colour):
        capture_moves = []
        if colour == WHITE:
            return self.pattern_search(WHITE_CAPTURE, WHITE_CAPTURE_EMPTY_OFFSET, white_capture_trie)
        elif colour == BLACK:
            return self.pattern_search(BLACK_CAPTURE, WHITE_CAPTURE_EMPTY_OFFSET, black_capture_trie)
        return capture_moves

    def heuristic_move_search(self, colour):
        moves = []
        immediate_win = self.immediate_win_search(colour)
        block_opponent_win = self.block_opponent_win_search(colour)
        open_four_ours = self.open_four_search(colour)
        open_four_theirs = self.open_four_search(opponent(colour))
        moves += immediate_win
        moves += block_opponent_win
        moves += open_four_ours
        moves += open_four_theirs
        return list(set(moves))