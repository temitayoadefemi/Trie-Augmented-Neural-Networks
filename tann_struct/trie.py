


class Trie:
    def __init__(self, depth=1, node = None):
        self.depth = depth
        self.root = self.build_trie(depth)
        self.node = node


    def build_trie(self, current_depth):
        if current_depth == 0:
            return None
        self.node.left_child = self.build_trie(current_depth - 1)
        self.node.right_child = self.build_trie(current_depth - 1)
        return self.node
    

    def traverse_nodes(self):
        return self._traverse_nodes_helper(self.root)
    

    def _traverse_nodes_helper(self, node):
        if node is None:
            return []
        nodes = [node]  # Collect the node itself, not just the network
        nodes.extend(self._traverse_nodes_helper(node.left_child))
        nodes.extend(self._traverse_nodes_helper(node.right_child))
        return nodes

