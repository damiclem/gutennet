from torch import nn


class GutenNet(nn.Module):

    # Constructor
    def __init__(
        self, vocab_size, embedding_dim, hidden_size,
        num_layers, hidden_type=nn.LSTM, dropout=0.0
    ):
        # Parent constructor
        super().__init__()
        # Handle embedding layer (first one)
        self.emb = nn.Embedding(vocab_size, embedding_dim)
        # Set hidden layer
        self.rnn = hidden_type(
            input_size=embedding_dim,  # Set hidden layers input size
            hidden_size=hidden_size,  # Number of units in each hidden layer
            num_layers=num_layers,  # Set number of hidden layers
            dropout=dropout,  # Set dropout
            batch_first=True  # First dimension is batch size
        )
        # TODO add batch normalization
        # Define output layer
        self.out = nn.Linear(hidden_size, embedding_dim)

    # Forward step
    def forward(self, x, state=None):
        # Use embedding only during training
        if self.training:
            # Go through embedding dimension
            x = self.emb(x)
        # Multi-layer rnn
        x, state = self.rnn(x, state)
        # Linear layer
        x = self.out(x)
        # Return both linear layer and state
        return x, state
