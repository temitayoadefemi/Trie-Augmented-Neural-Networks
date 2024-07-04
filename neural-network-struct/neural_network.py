from library_wrapper.wrapper import Wrapper

class TrieNetwork():
    def __init__(self, library, input_size, hidden_size, output_size):
        self.wrapper = Wrapper(library=library, input_size=input_size, hidden_size=hidden_size, output_size=output_size)


    def init_default_model(self):
        return self.wrapper.init_default_model()