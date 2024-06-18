from node import TrieNode

class Trie():
    def __init__(self, neural_network=None, depth=1):
        # Initialize the Trie with a neural network and a specified depth.
        self.neural_network = neural_network
        self.depth = depth
        self.default_network = neural_network  # Default neural network used to populate the Trie.

    def build_trie(self, current_depth=None):
        # Recursively build the Trie structure up to the specified depth.
        if current_depth is None:
            current_depth = self.depth  # Start with the max depth if not specified.

        if current_depth == 0:
            return None  # Base case: no more depth to build, return None.

        root = TrieNode(self.default_network)  # Create a new Trie node with the default network.
        root.left_child = self.build_trie(current_depth - 1)  # Recursively build the left child.
        root.right_child = self.build_trie(current_depth - 1)  # Recursively build the right child.
        return root  # Return the root of this sub-trie.
    
    def traverse_nodes(self, node):
        # Traverse the Trie and collect all neural networks from the nodes.
        nodes = []  # List to store neural networks.
        if node is None:
            return nodes  # If the node is None, return the empty list.

        nodes.append(node.neural_network)  # Add the current node's network to the list.
        nodes.extend(self.traverse_nodes(node.left_child))  # Recursively traverse the left child.
        nodes.extend(self.traverse_nodes(node.right_child))  # Recursively traverse the right child.
        return nodes  # Return the list of all networks.
