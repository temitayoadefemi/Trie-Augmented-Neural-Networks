from tann_struct.tann import TANN
from tann_struct.node import TrieNode
from tann_struct.trie import Trie
from neural_network_struct.neural_network import TrieNetwork


network = TrieNetwork(library="pytorch", input_size=2, hidden_size=20, output_size=2)
trie = Trie(depth=5, trie_network=network, feature_value=0.5)
tann = TANN(trie=trie)

train_data = [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)]
tann.train(train_data, 10, lr=0.2)