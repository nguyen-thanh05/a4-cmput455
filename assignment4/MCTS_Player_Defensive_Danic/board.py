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


"""
The GoBoard class implements a board and basic functions to play
moves, check the end of the game, and count the acore at the end.
The class also contains basic utility functions for writing a Go player.
For many more utility functions, see the GoBoardUtil class in board_util.py.

The board is stored as a one-dimensional array of GO_POINT in self.board.
See coord_to_point for explanations of the array encoding.
"""
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
        self.move_dict = {}

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
        self.board2d: np.ndarray[GO_POINT] = np.zeros((size, size), dtype=GO_POINT)
        self.zeros: np.ndarray[int] = np.zeros((size, size))
        self.ones: np.ndarray[int] = np.ones((size, size))
        self.possible_captures_bk: list[GO_POINT] = None
        self.possible_captures_wt: list[GO_POINT] = None
        self.mat_bk: np.ndarray[GO_POINT] = np.zeros((size, size))
        self.mat_wt: np.ndarray[GO_POINT] = np.zeros((size, size))
        self.mat_empt: np.ndarray[GO_POINT] = np.ones((size, size))
        self.filter_square = np.array([[1, 0, 0, 0, 1],
                                        [0, 1, 0, 1, 0],
                                        [0, 0, 1, 0, 0],
                                        [0, 1, 0, 1, 0],
                                        [1, 0, 0, 0, 1]])
        self.filter_square_add = np.array([[0, 0, 0, 0, 0],
                                            [0, 1, 0, 1, 0],
                                            [0, 0, 1, 0, 0],
                                            [0, 1, 0, 1, 0],
                                            [0, 0, 0, 0, 0]])
        self.filter_square_padded_1 = np.array([[1, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 0, 0, 0, 1]])
        self.filter_square_padded_2 = np.array([[0, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [1, 0, 0, 0, 0, 0, 0]])
        self.filter_diamond = np.array([[0, 0, 1, 0, 0],
                                        [0, 0, 1, 0, 0],
                                        [1, 1, 1, 1, 1],
                                        [0, 0, 1, 0, 0],
                                        [0, 0, 1, 0, 0]])
        self.filter_diamond_add = np.array([[0, 0, 0, 0, 0],
                                            [0, 0, 1, 0, 0],
                                            [0, 1, 1, 1, 0],
                                            [0, 0, 1, 0, 0],
                                            [0, 0, 0, 0, 0]])
        self.filter_diamond_padded_1 = np.array([[0, 0, 0, 1, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 1, 0, 0, 0]])
        self.filter_diamond_padded_2 = np.array([[0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [0, 0, 0, 0, 0, 0, 0],
                                                [1, 0, 0, 0, 0, 0, 1],
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 0, 0, 0, 0], 
                                                [0, 0, 0, 0, 0, 0, 0]])
        self.terminal = (False, EMPTY)
        self.terminal_last = [self.terminal]

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

    def get_color(self, point: GO_POINT) -> GO_COLOR:
        return self.board[point]

    def pt(self, row: int, col: int) -> GO_POINT:
        return coord_to_point(row, col, self.size)

    def is_legal(self, point: GO_POINT, color: GO_COLOR) -> bool:
        """
        Check whether it is legal for color to play on point
        """
        return (point == PASS) or (self.board[point] == EMPTY)
           
    def get_empty_points(self) -> np.ndarray:
        """
        Return:
            The empty points on the board
        """
        return where1d(self.board == EMPTY)

    def row_start(self, row: int) -> int:
        assert row >= 1
        assert row <= self.size
        return row * self.NS + 1

    def _initialize_empty_points(self, board_array: np.ndarray) -> None:
        """
        Fills points on the board with EMPTY
        Argument
        ---------
        board: numpy array, filled with BORDER
        """
        for row in range(1, self.size + 1):
            start: int = self.row_start(row)
            board_array[start : start + self.size] = EMPTY

    def play_move(self, point: GO_POINT, color: GO_COLOR) -> bool:
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
        self.update_board2d()
        self.terminal = self.is_terminal()
        self.terminal_last.append(self.terminal)
        return True
    
    def play_move_no_board2d(self, point: GO_POINT, color: GO_COLOR) -> bool:
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
        self.terminal = self.is_terminal()
        self.terminal_last.append(self.terminal)
        return True


    def score(self, colour):
        if colour == BLACK:
            return self.black_captures
        else:
            return self.white_captures

    def update_board2d(self):
        board2d: np.ndarray[GO_POINT] = np.zeros((self.size, self.size), dtype=GO_POINT)
        for row in range(self.size):
            start: int = self.row_start(row + 1)
            board2d[row, :] = self.board[start : start + self.size]
        board2d = np.flipud(board2d)

        self.board2d = board2d
        self.mat_bk = self.board2d == BLACK
        self.mat_wt = self.board2d == WHITE
        self.mat_empt = self.board2d == EMPTY
        self.possible_captures_bk: list[GO_POINT] = None
        self.possible_captures_wt: list[GO_POINT] = None

    def get_board_after_move(self, point: GO_POINT, color: GO_COLOR):
        """
        Tries to play a move of color on the point.
        Returns the 2d board after the stone was played.
        """
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
        
        board2d: np.ndarray[GO_POINT] = np.zeros((self.size, self.size), dtype=GO_POINT)
        for row in range(self.size):
            start: int = self.row_start(row + 1)
            board2d[row, :] = self.board[start : start + self.size]
        board2d = np.flipud(board2d)

        self.undo(False)
        return board2d
    
    def undo(self, update_terminal=True):
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
        if update_terminal:
            self.terminal = self.terminal_last.pop()
    
    def moveNumber(self):
        return self.depth

    def last_board_moves(self) -> List:
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

    def full_board_detect_five_in_a_row(self) -> GO_COLOR:
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
    
    def detect_five_in_a_row(self) -> GO_COLOR:
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
        Returns: is_terminal, winner
        If the result is a draw, winner = EMPTY
        """
        winner = self.detect_five_in_a_row()
        if winner != EMPTY:
            return True, winner
        elif self.get_captures(BLACK) >= 10:
            return True, BLACK
        elif self.get_captures(WHITE) >= 10:
            return True, WHITE
        elif (self.get_empty_points().size == 0) or (self.last_move == PASS and self.last2_move == PASS):
            return True, EMPTY
        else:
            return False, EMPTY

    def heuristic_eval(self, colour):
        """
        Returns: a very basic heuristic value of the board
        Currently only considers captures
        """
        if colour == BLACK:
            return (10 + self.black_captures - self.white_captures) / 20
        else:
            return (10 + self.white_captures - self.black_captures) / 20

    def state_to_str(self):
        state = np.array2string(self.board, separator='')
        state += str(self.current_player)
        state += str(self.black_captures)
        state += str(self.white_captures)
        return state

    def board2d_to_point(self, board) -> list:
        points = []
        zipped = list(zip(*np.where(np.flipud(board) != 0)))
        for (x, y) in zipped:
            points.append(coord_to_point(x+1, y+1, self.size))
        
        return points


    def num_out_of_tot(self, colour:GO_COLOR, num1:int, total:int, get_filter=False):
        """
        Returns empty points where "num1" amount of stones of the chosen colour are present in a total of "total" 
        amount of either empty or stones of the same colour are present, in a row, anywhere on the board.
        Ex:
        -x--xx would count as a num1 = 3, total = 6, while
        -xo-xx wouldn't count because of the interference.
        """
        num2 = total - num1
        zeros = self.zeros
        ones = self.ones
        if colour == BLACK:
            mat_col = self.mat_bk
        else:
            mat_col = self.mat_wt
        mat_empt = self.mat_empt

        filter = zeros.copy()
        for i in range(self.size):
            for convo in range(self.size - (total-1)):
                filter[i, convo : convo+total] += (mat_col[i, convo : convo+total].sum() == num1) and\
                                                      (mat_empt[i, convo : convo+total].sum() == num2)
                
                filter[convo : convo+total, i] += (mat_col[convo : convo+total, i].sum() == num1) and\
                                                      (mat_empt[convo : convo+total, i].sum() == num2)
            
        for diag_col_1, diag_col_2, diag_empt_1, diag_empt_2, offset \
            in [(np.diag(mat_col, k), np.diag(np.fliplr(mat_col), k), \
                 np.diag(mat_empt, k), np.diag(np.fliplr(mat_empt), k), k) \
                    for k in range(total-self.size, self.size-total+1)]:
            
            for convo in range(diag_col_1.size - (total-1)):
                temp = np.zeros(diag_col_1.shape)
                temp[convo : convo+total] = ((diag_col_1[convo : convo+total].sum() == num1) and\
                                               (diag_empt_1[convo : convo+total].sum() == num2))
                filter += np.diag(temp, offset)

                temp[convo : convo+total] = ((diag_col_2[convo : convo+total].sum() == num1) and\
                                                         (diag_empt_2[convo : convo+total].sum() == num2))
                filter += np.fliplr(np.diag(temp, offset))
        
        new_filter = ones * (filter > 0)
        mat_points = mat_empt * new_filter
        
        if get_filter:
            return self.board2d_to_point(mat_points), new_filter, mat_empt * filter
    
        return self.board2d_to_point(mat_points)

    def immediate_win(self, colour:GO_COLOR) -> list:
        """
        Returns points where the player can win imediately, including 5 in a rows, 
        or captures (counting multiple captures with the same stone as well).
        """
        points = []
        sum_wt = self.mat_wt.sum()
        sum_bk = self.mat_bk.sum()

        if colour == BLACK:
            if ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0):
                # Calculate a matrix of points where the number of captured stones + the current score would be a win
                _, mat, __, captures_filter_unchanged, ___, ____ = self.captures(colour, True)
                point_mat = mat * (((captures_filter_unchanged * 2) + self.score(colour)) >= 10)
                points += self.board2d_to_point(point_mat)
            if sum_bk >= 4:
                # If the player can win by a 5 in a row, add those points
                points += self.num_out_of_tot(colour, 4, 5)

        else:
            if ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0):
                # Calculate a matrix of points where the number of captured stones + the current score would be a win
                _, mat, __, captures_filter_unchanged, ___, ____ = self.captures(colour, True)
                point_mat = mat * (((captures_filter_unchanged * 2) + self.score(colour)) >= 10)
                points += self.board2d_to_point(point_mat)

            if sum_wt >= 4:
                # If the player can win by a 5 in a row, add those points
                points += self.num_out_of_tot(colour, 4, 5)
        
        # remove duplicate points
        points = list(dict.fromkeys(points))
        return points

    def block_win(self, colour:GO_COLOR, get_mat=False) -> list:
        """
        Returns points where the player can block the opponent's immediate win, either through captures,
        or by blocking a 5 in a row.
        """
        points = []
        
        # Get points where the opponent can win with a 5 in a row.
        # The filter is a matrix containing all points in the 5 that could cause a win for the opponent, for all cases on the board.
        temp, filter1, _ = self.num_out_of_tot(opponent(colour), 4, 5, True)
        points += temp

        # Get a matrix containing all empty points that allow captures for the current player.
        # The filter is a matrix containing all points within the possible captures (to find points that would be captured as well).
        _, captures_mat, filter2, __, filter2_3d, filter2_captures = self.captures(colour, True)

        # Calculate a matrix of points where the number of captured stones + the current score would be a win for the opponent
        _, opp_win_mat, __, captures_filter_unchanged, ___, ____ = self.captures(opponent(colour), True)
        point_win_filter = (((captures_filter_unchanged * 2) + self.score(opponent(colour))) >= 10)
        point_mat = opp_win_mat * point_win_filter
        points += self.board2d_to_point(point_mat)

        # Calculate points where a catpture for the current player would enable a win-block.
        # points where a capture would intercept a 5 in a row
        mat_5_in_a_row = np.repeat(filter1[:, :, np.newaxis], filter2_3d.shape[2], axis=2) * filter2_3d
        mat_5_in_a_row = (filter2_3d * (mat_5_in_a_row.sum(axis=0).sum(axis=0) > 0))
        mat_5_in_a_row *= np.repeat(captures_mat[:, :, np.newaxis], filter2_3d.shape[2], axis=2)
        mat_5_in_a_row = self.ones * (mat_5_in_a_row.sum(axis=2) != 0)
        points += self.board2d_to_point(mat_5_in_a_row)

        # points where a capture would prevent a capture win
        mat_capture_win = np.repeat(point_win_filter[:, :, np.newaxis], filter2_captures.shape[2], axis=2) * filter2_captures
        mat_capture_win = (filter2_captures * (mat_capture_win.sum(axis=0).sum(axis=0) > 0))
        mat_capture_win *= np.repeat(captures_mat[:, :, np.newaxis], filter2_captures.shape[2], axis=2)
        mat_capture_win = self.ones * (mat_capture_win.sum(axis=2) != 0)
        points += self.board2d_to_point(mat_capture_win)

        # remove duplicate points
        points = list(dict.fromkeys(points))
        return points

    def open_four(self, colour:GO_COLOR, get_mat=False) -> list:
        """
        Returns points where the player can succeed in creating an open-four.
        """
        num1, num2, total = 3, 3, 6 
        zeros = self.zeros
        if colour == BLACK:
            mat_col = self.mat_bk
        else:
            mat_col = self.mat_wt
        mat_empt = self.mat_empt

        filter = zeros.copy()
        for i in range(self.size):
            for convo in range(self.size - (total-1)):
                filter[i, convo+1 : convo+total-1] += (mat_col[i, convo+1 : convo+total-1].sum() == num1) and\
                                                      (mat_empt[i, convo : convo+total].sum() == num2)
                
                filter[convo+1 : convo+total-1, i] += (mat_col[convo+1 : convo+total-1, i].sum() == num1) and\
                                                      (mat_empt[convo : convo+total, i].sum() == num2)
            
        for diag_col_1, diag_col_2, diag_empt_1, diag_empt_2, offset \
            in [(np.diag(mat_col, k), np.diag(np.fliplr(mat_col), k), \
                 np.diag(mat_empt, k), np.diag(np.fliplr(mat_empt), k), k) \
                    for k in range(total-self.size, self.size-total+1)]:
            
            for convo in range(diag_col_1.size - (total-1)):
                temp = np.zeros(diag_col_1.shape)
                temp[convo+1 : convo+total-1] = ((diag_col_1[convo+1 : convo+total-1].sum() == num1) and\
                                             (diag_empt_1[convo : convo+total].sum() == num2))
                filter += np.diag(temp, offset)

                temp[convo+1 : convo+total-1] = ((diag_col_2[convo+1 : convo+total-1].sum() == num1) and\
                                             (diag_empt_2[convo : convo+total].sum() == num2))
                filter += np.fliplr(np.diag(temp, offset))
        
        mat_points = mat_empt * (filter > 0) 
        
        if get_mat:
            return self.board2d_to_point(mat_points), mat_empt * filter
        return self.board2d_to_point(mat_points)
    
    def captures(self, colour:GO_COLOR, get_filter=False):
        """
        Returns points that allow the player to capture stones.
        """
        if (colour == BLACK) and (self.possible_captures_bk != None):
            if get_filter:
                return self.possible_captures_bk, self.captures_mat_bk, self.captures_filter_bk, self.captures_filter_unchanged_bk, self.captures_filter_3d_bk, self.captures_filter_captured_bk
            return self.possible_captures_bk
        elif (colour == WHITE) and (self.possible_captures_wt != None):
            if get_filter:
                return self.possible_captures_wt, self.captures_mat_wt, self.captures_filter_wt, self.captures_filter_unchanged_wt, self.captures_filter_3d_wt, self.captures_filter_captured_wt
            return self.possible_captures_wt
        
        num1, num2, num3 = 1, 1, 2
        total = num1 + num2 + num3
        zeros = self.zeros
        ones = self.ones
        if colour == BLACK:
            mat_col = self.mat_bk
            mat_opp = self.mat_wt
        else:
            mat_col = self.mat_wt
            mat_opp = self.mat_bk
        mat_empt = self.mat_empt

        filter_3d = np.zeros((self.size, self.size, 1))
        filter_captured = np.zeros((self.size, self.size, 1))
        for i in range(self.size):
            for convo in range(self.size - ((total)-1)):
                temp = zeros.copy()
                cond = (mat_opp[i, convo+1 : convo+total-1].sum() == num3) and\
                       (mat_empt[i, convo : convo+total].sum() == num2) and\
                       (mat_col[i, convo : convo+total].sum() == num1)
                
                temp[i, convo+1 : convo+total-1] = cond
                filter_captured = np.dstack((filter_captured, temp))
                temp[i, convo : convo+total] = cond
                filter_3d = np.dstack((filter_3d, temp))


                temp = zeros.copy()
                cond = (mat_opp[convo+1 : convo+total-1, i].sum() == num3) and\
                       (mat_empt[convo : convo+total, i].sum() == num2) and\
                       (mat_col[convo : convo+total, i].sum() == num1)
                
                temp[convo+1 : convo+total-1, i] = cond
                filter_captured = np.dstack((filter_captured, temp))
                temp[convo : convo+total, i] = cond
                filter_3d = np.dstack((filter_3d, temp))
            
        for diag_col_1, diag_col_2, diag_empt_1, diag_empt_2, diag_opp_1, diag_opp_2, offset \
            in [(np.diag(mat_col, k), np.diag(np.fliplr(mat_col), k),\
                 np.diag(mat_empt, k), np.diag(np.fliplr(mat_empt), k),\
                 np.diag(mat_opp, k), np.diag(np.fliplr(mat_opp), k), k) \
                    for k in range(total-self.size, self.size-total+1)]:
            
            for convo in range(diag_col_1.size - (total-1)):
                temp = np.zeros(diag_col_1.shape)
                cond = ((diag_opp_1[convo+1 : convo+total-1].sum() == num3) and\
                        (diag_empt_1[convo : convo+total].sum() == num2) and\
                        (diag_col_1[convo : convo+total].sum() == num1))
                
                temp[convo+1 : convo+total-1] = cond
                filter_captured = np.dstack((filter_captured, np.diag(temp, offset)))
                temp[convo : convo+total] = cond
                filter_3d = np.dstack((filter_3d, np.diag(temp, offset)))


                temp = np.zeros(diag_col_1.shape)
                cond = ((diag_opp_2[convo+1 : convo+total-1].sum() == num3) and\
                        (diag_empt_2[convo : convo+total].sum() == num2) and\
                        (diag_col_2[convo : convo+total].sum() == num1))
                
                temp[convo+1 : convo+total-1] = cond
                filter_captured = np.dstack((filter_captured, np.fliplr(np.diag(temp, offset))))
                temp[convo : convo+total] = cond
                filter_3d = np.dstack((filter_3d, np.fliplr(np.diag(temp, offset))))
        
        filter = filter_3d.sum(axis=2)
        new_filter = ones * (filter > 0)
        mat_points = mat_empt * new_filter
        points = self.board2d_to_point(mat_points)

        if colour == BLACK:
            self.possible_captures_bk = points
            self.captures_mat_bk = mat_points
            self.captures_filter_bk = new_filter
            self.captures_filter_unchanged_bk = filter
            self.captures_filter_3d_bk = filter_3d
            self.captures_filter_captured_bk = filter_captured
        elif colour == WHITE:
            self.possible_captures_wt = points
            self.captures_mat_wt = mat_points
            self.captures_filter_wt = new_filter
            self.captures_filter_unchanged_wt = filter
            self.captures_filter_3d_wt = filter_3d
            self.captures_filter_captured_wt = filter_captured

        if get_filter:
            return points, mat_points, new_filter, filter, filter_3d, filter_captured
        
        return points
    
    def block_open_four(self, col:GO_COLOR, get_mat=False):
        """
        Returns points that would allow the player to block the opponent from creating an open-four.
        """
        colour = opponent(col)
        num1, num2, total = 3, 3, 6 
        zeros = self.zeros
        #ones = self.ones
        mat_empt = self.mat_empt
        if colour == BLACK:
            mat_col = self.mat_bk
            #mat_opp = self.mat_wt
        else:
            mat_col = self.mat_wt
            #mat_opp = self.mat_bk

        filter = zeros.copy()
        for i in range(self.size):
            for convo in range(self.size - (total-1)):
                filter[i, convo : convo+total] += (mat_col[i, convo+1 : convo+total-1].sum() == num1) and\
                                                      (mat_empt[i, convo : convo+total].sum() == num2)
                
                filter[convo : convo+total, i] += (mat_col[convo+1 : convo+total-1, i].sum() == num1) and\
                                                      (mat_empt[convo : convo+total, i].sum() == num2)
            
        for diag_col_1, diag_col_2, diag_empt_1, diag_empt_2, offset \
            in [(np.diag(mat_col, k), np.diag(np.fliplr(mat_col), k), \
                 np.diag(mat_empt, k), np.diag(np.fliplr(mat_empt), k), k) \
                    for k in range(total-self.size, self.size-total+1)]:
            
            for convo in range(diag_col_1.size - (total-1)):
                temp = np.zeros(diag_col_1.shape)
                temp[convo : convo+total] = ((diag_col_1[convo+1 : convo+total-1].sum() == num1) and\
                                             (diag_empt_1[convo : convo+total].sum() == num2))
                filter += np.diag(temp, offset)

                temp[convo : convo+total] = ((diag_col_2[convo+1 : convo+total-1].sum() == num1) and\
                                             (diag_empt_2[convo : convo+total].sum() == num2))
                filter += np.fliplr(np.diag(temp, offset))
        
        mat_points = mat_empt * filter

        _, captures_mat, filter2, __, filter2_3d, filter2_captures = self.captures(col, True)

        mat_block_four = np.repeat(filter[:, :, np.newaxis], filter2_3d.shape[2], axis=2) * filter2_3d
        mat_block_four = filter2_3d * (mat_block_four.sum(axis=0).sum(axis=0))
        mat_block_four *= np.repeat(captures_mat[:, :, np.newaxis], filter2_3d.shape[2], axis=2)
        mat_block_four = mat_block_four.sum(axis=2)

        max_block_four = mat_block_four.max()
        if max_block_four > 0:
            mat_block_four = mat_block_four == max_block_four
        max_mat_points = mat_points.max()
        if max_mat_points > 0:
            mat_points = mat_points == max_mat_points

        if max_block_four > max_mat_points:
            mat = mat_block_four
        elif max_block_four < max_mat_points:
            mat = mat_points
        else:
            mat = mat_block_four + mat_points
        
        if get_mat:
            return self.board2d_to_point(mat), (self.ones * mat_points) + mat_block_four
        return self.board2d_to_point(mat)
    
    def play_pattern(self, colour:GO_COLOR, get_mat=False):
        """
        Returns points within two specific pattern types promoting wining conditions
        - - - - - - -
        - - - x - - -
        - - x x x - -
        - - - x - - -
        - - - - - - -
             or
        - - - - - - -
        - - x - x - -
        - - - x - - -
        - - x - x - -
        - - - - - - -
        """
        if colour == BLACK:
            mat_col = self.mat_bk
        else:
            mat_col = self.mat_wt
        mat_empt = self.mat_empt
        mat_col_empt = mat_col + mat_empt
        mat_empt_padded = np.pad(mat_col_empt, 1, 'constant', constant_values=0)

        filter_sqr = self.zeros.copy()
        filter_dia = self.zeros.copy()
        for conv_x in range(self.size + 1 - 5):
            for conv_y in range(self.size + 1 - 5):
                cond1 = ((mat_col_empt[conv_x : conv_x+5, conv_y : conv_y+5] * self.filter_square).sum() == 9) and\
                        ((mat_empt_padded[conv_x : conv_x+7, conv_y : conv_y+7] * self.filter_square_padded_1).sum() >= 1) and\
                        ((mat_empt_padded[conv_x : conv_x+7, conv_y : conv_y+7] * self.filter_square_padded_2).sum() >= 1)
                
                cond2 = ((mat_col_empt[conv_x : conv_x+5, conv_y : conv_y+5] * self.filter_diamond).sum() == 9) and\
                        ((mat_empt_padded[conv_x : conv_x+7, conv_y : conv_y+7] * self.filter_diamond_padded_1).sum() >= 1) and\
                        ((mat_empt_padded[conv_x : conv_x+7, conv_y : conv_y+7] * self.filter_diamond_padded_2).sum() >= 1)
                
                filter_sqr[conv_x : conv_x+5, conv_y : conv_y+5] += self.filter_square_add * cond1
                filter_dia[conv_x : conv_x+5, conv_y : conv_y+5] += self.filter_diamond_add * cond2

                sum1 = (mat_col[conv_x : conv_x+5, conv_y : conv_y+5] * self.filter_square_add).sum()
                sum2 = (mat_col[conv_x : conv_x+5, conv_y : conv_y+5] * self.filter_diamond_add).sum()
                cond3 = cond1 and (sum1 > 0)
                cond4 = cond2 and (sum2 > 0)
                filter_sqr[conv_x : conv_x+5, conv_y : conv_y+5] += self.filter_square_add * cond3 * (sum1 + 1)
                filter_dia[conv_x : conv_x+5, conv_y : conv_y+5] += self.filter_diamond_add * cond4 * (sum2 + 1)
        
        move_filter = abs((filter_dia + filter_sqr) * abs(mat_col - 1))

        points = []
        zipped = list(zip(*np.where(np.flipud(move_filter) != 0)))
        zipped.sort(key=lambda x: move_filter[x[0], x[1]])
        for (x, y) in zipped:
            points.append(coord_to_point(x+1, y+1, self.size))
        
        if get_mat:
            return points, move_filter
        return points
    
    def board_state_heuristics(self, colour:GO_COLOR):
        """
        Returns a 2D matrix containing heuristics for every availble move on the board, as normalized values.
        """
        heur_mat = self.zeros.copy()
        sum_wt = self.mat_wt.sum()
        sum_bk = self.mat_bk.sum()

        # heuristic for direct wins
        temp = self.zeros.copy()
        if colour == BLACK:
            if ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0):
                # Calculate a matrix of points where the number of captured stones + the current score would be a win
                _, mat, __, captures_filter_unchanged, ___, ____ = self.captures(colour, True)
                temp += mat * (((captures_filter_unchanged * 2) + self.score(colour)) >= 10)
                
            if sum_bk >= 4:
                # If the player can win by a 5 in a row, add those points
                _, __, mat = self.num_out_of_tot(colour, 4, 5, True)
                temp += mat

        else:
            if ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0):
                # Calculate a matrix of points where the number of captured stones + the current score would be a win
                _, mat, __, captures_filter_unchanged, ___, ____ = self.captures(colour, True)
                temp = mat * (((captures_filter_unchanged * 2) + self.score(colour)) >= 10)

            if sum_wt >= 4:
                # If the player can win by a 5 in a row, add those points
                _, __, mat = self.num_out_of_tot(colour, 4, 5, True)
                temp += mat

        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 64 * temp



        # heuristic for blockwins
        temp = self.zeros.copy()
        if colour == WHITE:
            cond1 = ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0)
            cond2 = sum_bk >= 4
            cond3 = ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0)
        else:
            cond1 = ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0)
            cond2 = sum_wt >= 4
            cond3 = ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0)
        
        if cond1:
            # Calculate a matrix of points where the number of captured stones + the current score would be a win
            _, opp_win_mat, __, captures_filter_unchanged, ___, ____ = self.captures(opponent(colour), True)
            point_win_filter = (((captures_filter_unchanged * 2) + self.score(opponent(colour))) >= 10) * captures_filter_unchanged
            point_mat = opp_win_mat * point_win_filter
            temp += point_mat
        if cond2:
            # If the player can win by a 5 in a row, add those points
            _, filter1, mat = self.num_out_of_tot(opponent(colour), 4, 5, True)
            temp += mat
        if cond1 and cond2 and cond3:
            # Get a matrix containing all empty points that allow captures for the current player.
            # The filter is a matrix containing all points within the possible captures (to find points that would be captured as well).
            _, captures_mat, __, ___, filter2_3d, filter2_captures = self.captures(colour, True)

            # Calculate points where a catpture for the current player would enable a win-block.
            # points where a capture would intercept a 5 in a row
            mat_5_in_a_row = np.repeat(filter1[:, :, np.newaxis], filter2_3d.shape[2], axis=2) * filter2_3d
            mat_5_in_a_row = (filter2_3d * (mat_5_in_a_row.sum(axis=0).sum(axis=0) > 0))
            mat_5_in_a_row *= np.repeat(captures_mat[:, :, np.newaxis], filter2_3d.shape[2], axis=2)
            mat_5_in_a_row = mat_5_in_a_row.sum(axis=2)
            temp += mat

            # points where a capture would prevent a capture win
            mat_capture_win = np.repeat(point_win_filter[:, :, np.newaxis], filter2_captures.shape[2], axis=2) * filter2_captures
            mat_capture_win = (filter2_captures * (mat_capture_win.sum(axis=0).sum(axis=0) > 0))
            mat_capture_win *= np.repeat(captures_mat[:, :, np.newaxis], filter2_captures.shape[2], axis=2)
            mat_capture_win = mat_capture_win.sum(axis=2)
            temp += mat_capture_win
                
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 32 * temp
        


        # heuristic for open_four
        _, temp = self.open_four(colour, True)
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 16 * temp



        # heuristic for block_open_four
        _, temp = self.block_open_four(colour, True)
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 8 * temp
        


        # heuristic for captures
        temp = self.zeros.copy()
        if colour == BLACK:
            if (sum_bk >= 1) and (sum_wt >= 2):
                _, __, ___, mat, ____, _____ = self.captures(colour, True)
                temp += mat
        else:
            if (sum_bk >= 1) and (sum_wt >= 2):
                _, __, ___, mat, ____, _____ = self.captures(colour, True)
                temp += mat

        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 4 * temp
        

        # heuristic for play_pattern
        _, temp = self.play_pattern(colour, True)
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 2 * temp
        


        # calculate final heuristic matrix
        heur_mat += self.mat_empt
        heur_mat *= self.mat_empt
        heur_max = heur_mat.max()
        if heur_max > 0:
            heur_mat /= heur_max
        return heur_mat
    
    def play_heuristic(self, colour:GO_COLOR):
        """
        Returns a 2D matrix containing heuristics for every availble move on the board, as a percentage rate.
        """
        heur_mat = self.zeros.copy()
        sum_wt = self.mat_wt.sum()
        sum_bk = self.mat_bk.sum()

        # heuristic for direct wins
        temp = self.zeros.copy()
        if colour == BLACK:
            if ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0):
                # Calculate a matrix of points where the number of captured stones + the current score would be a win
                _, mat, __, captures_filter_unchanged, ___, ____ = self.captures(colour, True)
                temp += mat * (((captures_filter_unchanged * 2) + self.score(colour)) >= 10)
                
            if sum_bk >= 4:
                # If the player can win by a 5 in a row, add those points
                _, __, mat = self.num_out_of_tot(colour, 4, 5, True)
                temp += mat

        else:
            if ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0):
                # Calculate a matrix of points where the number of captured stones + the current score would be a win
                _, mat, __, captures_filter_unchanged, ___, ____ = self.captures(colour, True)
                temp = mat * (((captures_filter_unchanged * 2) + self.score(colour)) >= 10)

            if sum_wt >= 4:
                # If the player can win by a 5 in a row, add those points
                _, __, mat = self.num_out_of_tot(colour, 4, 5, True)
                temp += mat

        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 4 * temp



        # heuristic for blockwins
        temp = self.zeros.copy()
        if colour == WHITE:
            cond1 = ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0)
            cond2 = sum_bk >= 4
            cond3 = ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0)
        else:
            cond1 = ((sum_bk + self.score(WHITE)) >= 10) and (sum_wt > 0)
            cond2 = sum_wt >= 4
            cond3 = ((sum_wt + self.score(BLACK)) >= 10) and (sum_bk > 0)
        
        if cond1:
            # Calculate a matrix of points where the number of captured stones + the current score would be a win
            _, opp_win_mat, __, captures_filter_unchanged, ___, ____ = self.captures(opponent(colour), True)
            point_win_filter = (((captures_filter_unchanged * 2) + self.score(opponent(colour))) >= 10) * captures_filter_unchanged
            point_mat = opp_win_mat * point_win_filter
            temp += point_mat
        if cond2:
            # If the player can win by a 5 in a row, add those points
            _, filter1, mat = self.num_out_of_tot(opponent(colour), 4, 5, True)
            temp += mat
        if cond1 and cond2 and cond3:
            # Get a matrix containing all empty points that allow captures for the current player.
            # The filter is a matrix containing all points within the possible captures (to find points that would be captured as well).
            _, captures_mat, __, ___, filter2_3d, filter2_captures = self.captures(colour, True)

            # Calculate points where a catpture for the current player would enable a win-block.
            # points where a capture would intercept a 5 in a row
            mat_5_in_a_row = np.repeat(filter1[:, :, np.newaxis], filter2_3d.shape[2], axis=2) * filter2_3d
            mat_5_in_a_row = (filter2_3d * (mat_5_in_a_row.sum(axis=0).sum(axis=0) > 0))
            mat_5_in_a_row *= np.repeat(captures_mat[:, :, np.newaxis], filter2_3d.shape[2], axis=2)
            mat_5_in_a_row = mat_5_in_a_row.sum(axis=2)
            temp += mat

            # points where a capture would prevent a capture win
            mat_capture_win = np.repeat(point_win_filter[:, :, np.newaxis], filter2_captures.shape[2], axis=2) * filter2_captures
            mat_capture_win = (filter2_captures * (mat_capture_win.sum(axis=0).sum(axis=0) > 0))
            mat_capture_win *= np.repeat(captures_mat[:, :, np.newaxis], filter2_captures.shape[2], axis=2)
            mat_capture_win = mat_capture_win.sum(axis=2)
            temp += mat_capture_win
                
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 4 * temp
        


        # heuristic for open_four
        _, temp = self.open_four(colour, True)
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 3 * temp



        # heuristic for block_open_four
        _, temp = self.block_open_four(colour, True)
        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 3 * temp
        


        # heuristic for captures
        temp = self.zeros.copy()
        if colour == BLACK:
            if (sum_bk >= 1) and (sum_wt >= 2):
                _, __, ___, mat, ____, _____ = self.captures(colour, True)
                temp += mat
        else:
            if (sum_bk >= 1) and (sum_wt >= 2):
                _, __, ___, mat, ____, _____ = self.captures(colour, True)
                temp += mat

        max_temp = temp.max()
        if max_temp > 0:
            temp /= max_temp
            heur_mat += 2 * temp
        


        # calculate final heuristic matrix
        heur_mat += self.mat_empt
        heur_mat *= self.mat_empt
        heur_sum = heur_mat.sum()
        if heur_sum > 0:
            heur_mat /= heur_sum
        return heur_mat
