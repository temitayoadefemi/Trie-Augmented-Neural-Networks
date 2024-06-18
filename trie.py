from node import TrieNode

class Trie():
    def __init__(self, neural_network=None, depth=1):
        self.neural_network = neural_network
        self.depth = depth
        self.default_network = neural_network

    def build_trie(self, current_depth=None):
        if current_depth is None:
            current_depth = self.depth

        if current_depth == 0:
            return None

        root = TrieNode(self.default_network)
        root.left_child = self.build_trie(current_depth - 1)
        root.right_child = self.build_trie(current_depth - 1)
        return root
    
    def traverse_nodes(self, node):
        nodes = []
        if node is None:
            return nodes
        nodes.append(node.neural_network)
        nodes.extend(self.traverse_nodes(node.left_child))
        nodes.extend(self.traverse_nodes(node.right_child))