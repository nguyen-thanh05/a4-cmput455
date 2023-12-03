from TreeNode import Node
from board import GoBoard
import board_util
from board_base import opponent

class Tree:
    def __init__(self):
        self.root = Node()
        self.nodes_to_expand = []
        self.nodes_to_simulate = []

    def save_tree(self, root, nodes_to_expand, nodes_to_simulate):
        """
        A tree is saved before we have made our move (ie, it's the same tree we create during play).
        """
        self.root = root
        self.nodes_to_expand = nodes_to_expand
        self.nodes_to_simulate = nodes_to_simulate

    def load_tree(self, new_board):
        """
        Finds the new root, cleans the lists and returns the new node and the new lists

        Identify new root node and set its parent to None
            If new root has children, it has been expanded already, proceed (it's children are already in the right place
            For each item in each list, recursively search parents until hitting the old root or new root (hitting None)
                    If old root (can just check value), pop from list
                    If new root (can just check value), leave alone
            else: expand the root by adding its children to the simulate list
        """
        if not self._find_new_root(new_board): # never expanded new root
            return self.root, self.nodes_to_expand, self.nodes_to_simulate
        else:
            self._clean_lists()
            return self.root, self.nodes_to_expand, self.nodes_to_simulate

    def _find_new_root(self, new_board):
        """ find the node reached by the last two moves (our last move and our opponent's response) and make it
        the new root
        """
        # find the last2_move in the list of root's children
        l1_node = self.root.children.get(new_board.last2_move)  # This must exist because we first expand the root
        # find the last_move in the list of root's children's children
        l2_node = l1_node.children.get(new_board.last_move)  # again, possible this doesn't exist?
        if l2_node is None or not l2_node.children:  # we have never expanded this before, start a new tree from scratch
            new_root = Node()
            new_root.board = new_board
            new_root.name = "new_root"
            new_root.value = None
            new_root.parent = None
            new_root.color_to_play = opponent(l1_node.color_to_play)
            self.nodes_to_simulate = [new_root]
            self.nodes_to_expand = []
            self.root = new_root
            return False
        else:
            self.root = l2_node
            self.root.parent = None
        return True

    def _clean_lists(self):
        """ for each item in each list, traverse up the tree until finding NONE. Pop if it's not the new root
        """
        # clean the simulation list

        for leaf in self.nodes_to_simulate.copy():
            current = leaf
            while current.parent is not None:
                current = current.parent

            if current != self.root:
                self.nodes_to_simulate.remove(leaf)

        # Same logic to clean the simulation list
        for leaf in self.nodes_to_expand.copy():
            current = leaf
            while current.parent is not None:
                current = current.parent
            print("current:", current)
            if current != self.root:
                self.nodes_to_expand.remove(leaf)





        """
        Key problem: we do not know if leafs in either list are way down from the new node, or from some other node
        
        New approach:
        Identify new root node and set its parent to None
            If new root has children, it has been expanded already, proceed (it's children are already in the right place
            For each item in each list, recursively search parents until hitting the old root or new root (hitting None)
                    If old root (can just check value), pop from list
                    If new root (can just check value), leave alone
            else: expand the root by adding its children to the simulate list
        
        done
        """

"""
def test():
    # Build a tree:
    root = Node()
    root.name = "root"
    root.value = 0

    child1 = root.add_child("9", 1)
    child2 = root.add_child("10", 2)
    child3 = root.add_child("11", 3)

    child1_1 = child1.add_child("12", 11)
    child1_2 = child1.add_child("13", 12)
    child1_3 = child1.add_child("14", 13)


    to_expand = [child1_1, child1_2, child3, child2]
    to_simulate = [child1_3]

    board = GoBoard(7)
    root.board = board.copy()

    board.play_move(9, 1)
    child1.board = board.copy()
    board.undo()

    board.play_move(10, 1)
    child2.board = board.copy()
    board.undo()

    board.play_move(11, 1)
    child3.board = board.copy()
    board.undo()

    board.play_move(11, 1)

    board.play_move(12, 0)
    child1_1.board = board.copy()
    board.undo()

    board.play_move(13, 0)
    child1_2.board = board.copy()
    board.undo()

    board.play_move(14, 0)
    child1_3.board = board.copy()
    board.undo()

    board.play_move(13, 0)

    tree = Tree()
    tree.save_tree(root, to_expand, to_simulate)
    print(tree.root, tree.nodes_to_expand, tree.nodes_to_simulate)


    print(tree.root)
    print(tree.root.children)
    for child in tree.root.children:
        print(tree.root.children[child])
        print(tree.root.children[child].children)
    tree._find_new_root(board)
    tree._clean_lists()

    print(tree.root, tree.nodes_to_expand, tree.nodes_to_simulate)
    
    print(child2.parent)
    

if __name__ == "__main__":
    test()
"""