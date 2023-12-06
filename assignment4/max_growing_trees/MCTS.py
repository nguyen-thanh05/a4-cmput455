from MCTSNode import Node
import random


class MCTS:
    def __init__(self, root):
        self.root = root
        self.root.add_children(self.root.board.get_empty_points())

    def tree_reuse(self, last_move, last2_move):
        self.root = self.root.children[last2_move].children[last_move]
        self.root.parent = None

        valid_points = self.root.board.get_empty_points()
        for point in valid_points:
            if point not in self.root.children:
                self.root.add_child(point)

    def get_root(self):
        return self.root

    def selection(self, root):
        """ Implement the selection phase of the MCTS algorithm."""
        current = root
        while len(current.children) != 0:
            current = current.get_max_child()
        current = current.parent
        valid_moves = current.board.get_empty_points()

        # Explore if there are unexplored moves, else choose max
        if len(valid_moves) != len(current.children):
            for move in valid_moves:
                if move not in current.children:
                    current.add_child(move)

        return current.get_max_child()

    def backpropagation(self, node, outcome):
        """ Implement the backpropagation phase of the MCTS algorithm."""
        # if node is root, return
        if node.parent is None:
            node.Q += outcome
            node.N += 1
            return
        else:
            # update node's value
            node.Q += outcome
            node.N += 1
            # recurse on parent
            self.backpropagation(node.parent, outcome)

    def expansion(self, node):
        """ Implement the expansion phase of the MCTS algorithm."""
        # If node is terminal
        if node.outcome is not None:
            return node

        legal_moves = node.board.get_empty_points()
        move = random.choice(legal_moves)
        node.add_child(move)
        return node.children[move]

    def simulation(self, node):
        """ Implement the simulation phase of the MCTS algorithm."""
        # If node is terminal
        terminal = None
        copy_board = node.board.copy()
        colour = copy_board.current_player

        while True:
            terminal = copy_board.is_terminal()
            if terminal[0]:
                break

            legal_moves = terminal[2]
            move = random.choice(legal_moves)

            copy_board.play_move(move, colour)
            colour = 3 - colour
        winner = terminal[1]
        if winner == node.colour:
            node.outcome = 1
        elif winner == 3 - node.colour:
            node.outcome = -1
        else:
            node.outcome = 0

    def run(self):
        # Selection
        selected_node = self.selection(self.root)
        # Expansion
        expanded_node = self.expansion(selected_node)
        # Simulation
        self.simulation(expanded_node)
        # Backpropagation
        self.backpropagation(expanded_node, expanded_node.outcome)

