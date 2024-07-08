from .trie import Trie
from misc.tensor import Tensor
import torch.optim as optim
import torch

class TANN():
    def __init__(self, trie=None):
        # Constructor method to initialize a TANN object with optional custom node and trie.
        # If no node or trie is provided, new instances of TrieNode and Trie will be created.
        self.trie = trie if trie is not None else Trie()
        self.training_criteria = None


    def train(self, train_data, epochs, lr):
        # Placeholder for the train method, which should be implemented to train the model.
        optimizers = []
        nodes = self.trie.traverse_nodes()
        for node in nodes:
            optimizer = optim.Adam(self.trie.network.default_model().parameters(), lr)
            optimizers.append(optimizer)
        
        for epoch in range(epochs):
            for inputs, target in train_data:
                inputs = torch.tensor(inputs, dtype=torch.float32)
                target = torch.tensor([target], dtype=torch.float32)

            node = self.trie.root
            for i in range(len(inputs)):
                if inputs[i] == 0:
                    node = node.left_child
                else:
                    node = node.right_child
            output = self.trie.network.collect_inputs(inputs)
            loss = self.trie.network.wrapper.get_loss(output, target)

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
