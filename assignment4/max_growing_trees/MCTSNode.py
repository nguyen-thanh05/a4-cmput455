import math


EXPLORATION_CONSTANT = 1.4


class Node:
    def __init__(self, move, parent):
        self.parent = parent
        self.move = move
        self.Q = 0
        self.N = 0
        self.board = None
        self.colour = None
        self.outcome = None
        self.children = {}

    def __str__(self):
        return "Node: " + str(self.move) + " Q: " + str(self.Q) + " N: " + str(self.N)

    def add_child(self, move):
        child = Node(move, self)
        child.board = self.board.copy()
        child.colour = 3 - self.colour

        child.board.play_move(move, child.colour)
        self.children[move] = child

    def add_children(self, moves: list):
        for move in moves:
            self.add_child(move)

    def value(self):
        if self.N == 0:
            return 1.8  # Prioritize unexplored nodes
        return self.Q / self.N + EXPLORATION_CONSTANT * math.sqrt(math.log(self.parent.N) / self.N)

    def get_max_child(self):
        max_child = max(self.children.values(), key=lambda node: node.value())
        return max_child




