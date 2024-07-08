from tann_struct.node import TrieNode

class Trie:
    def __init__(self, depth=1, trie_network=None, feature_value=None):
        self.depth = depth
        self.root = self.build_trie(depth, trie_network, feature_value)
        self.network = trie_network
        self.feature_value = feature_value


    def build_trie(self, current_depth, network, feature_value):
        if current_depth == 0:
            return None
        node = TrieNode(trie_network=network, feature_value=feature_value)
        node.left_child = self.build_trie(current_depth - 1, network, feature_value)
        node.right_child = self.build_trie(current_depth - 1, network, feature_value)
        return node
    

    def traverse_nodes(self):
        return self._traverse_nodes_helper(self.root)
    

    def _traverse_nodes_helper(self, node):
        if node is None:
            return []
        nodes = [node]  # Collect the node itself, not just the network
        nodes.extend(self._traverse_nodes_helper(node.left_child))
        nodes.extend(self._traverse_nodes_helper(node.right_child))
        return nodes

