
import numpy as np


class Node(object):
    """A class that represents a node of a game tree. Uses nested dictionaries to represent the tree.
    A node is a dictionary with a name as the key. Dictionary values are children nodes which are also
    dictionaries. The root node is the only node with a None parent. The root node is also the only node with a None
     name and value.
    """

    def __init__(self):
        self.parent = None
        self.children = {}
        self.name = None
        self.value = 0
        self.board = None
        self.color_to_play = None

    def __repr__(self):
        return self.name

    def add_child(self, name, value):
        """Adds a child node to the node.
        Args:
            name: The name of the child node.
            value: The value of the child node.
        """
        child = Node()
        child.parent = self
        child.name = name
        child.value = value
        self.children[name] = child
        return child

    def get_max_child(self):
        """
        Returns the child with the maximum value.
        """
        max_child = None
        max_value = -np.inf
        for child in self.children.values():
            if child.value > max_value:
                max_child = child
                max_value = child.value
        return max_child

    def get_min_child(self):
        """
        Returns the child with the minimum value.
        """
        min_child = None
        min_value = np.inf
        for child in self.children.values():
            if child.value < min_value:
                min_child = child
                min_value = child.value
        return min_child




