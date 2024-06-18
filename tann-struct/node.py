class TrieNode():
    def __init__(self, neural_network=None, feature_value=None):
        # Constructor for the TrieNode class.
        self.left_child = None  # Initialize left_child to None, this will hold the reference to the left child node.
        self.right_child = None  # Initialize right_child to None, this will hold the reference to the right child node.
        self.neural_network = neural_network  # Assign the neural network passed to the node. This attribute stores the neural network associated with this node.
        self.feature_value = feature_value
