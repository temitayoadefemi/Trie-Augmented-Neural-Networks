from .trie import Trie
from misc.tensor import Tensor

class TANN():
    def __init__(self, trie=None):
        # Constructor method to initialize a TANN object with optional custom node and trie.
        # If no node or trie is provided, new instances of TrieNode and Trie will be created.
        self.trie = trie if trie is not None else Trie()
        self.training_criteria = None


    def train(self, train_data, epochs, lr):
        # Placeholder for the train method, which should be implemented to train the model.
        optimizers = []
        nodes = self.trie.traverse_nodes(self.trie.root)
        for node in nodes:
            optimizer = self.trie.network.wrapper.get_optimizer(node.trie_network.parameters())
            optimizers.append(optimizer)
        
        for epoch in range(epochs):
            for inputs, target in train_data:
                inputs = Tensor(inputs, backend="pytorch")
                target = Tensor([target], backend="pytorch")

            node = self.trie.root
            for i in range(len(inputs)):
                if inputs[i] == 0:
                    node = node.left_child
                else:
                    node = node.right_child
            output = node.trie_network(inputs)
            loss = self.node.trie_network.get_criterion(output, target)

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def inference(self):
        inputs = Tensor(inputs, backend=self.trie.network.library)
        node = self.trie.root
        for i in range(len(inputs)):
            if inputs[i] == 0:
                node = node.left_child
            else:
                node = node.right_child
        #Implement inference criteria
        output = node.mini_nn(inputs)
        return output.item()
