from trie import Trie
from node import TrieNode

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
            optimizer = self.trie.neural_network.wrapper.get_optomizer(node.trie_network.parameters())
            optimizers.append(optimizer)
        
        for epoch in range(epochs):
            for inputs, target in train_data:
                inputs = tensor(inputs, dtype=torch.float32)
                target = tensor([target], dtype=torch.float32)

            node = self.trie.root

            #Implement training criteria

            output = node.trie_network(inputs)
            loss = self.node.trie_network.get_criterion(output, target)

            for optimizer in optimizers:
                optimizer.zero_grad()
            loss.backward()
            for optimizer in optimizers:
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def inference(self):
        inputs = tensor(inputs, dtype=torch.float32)
        node = self.trie.root

        #Implement inference criteria

        output = node.mini_nn(inputs)
        return output.item()

    def classification_report(self):
        # Placeholder for the classification_report method, which should generate a report 
        # summarizing the performance of the model after training.
        pass

