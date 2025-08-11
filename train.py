from data import *
from model import *
import argparse

argparser = argparse.ArgumentParser()
argparser.add_argument('filename', type=str, help='Path to the text file containing training data')
argparser.add_argument('--n-epochs', type=int, default=2000, help='Number of training epochs')
argparser.add_argument('--print-every', type=int, default=100, help='Frequency of printing training progress')
argparser.add_argument('--hidden-size', type=int, default=100, help='Size of the hidden layer in the RNN')
argparser.add_argument('--n-layers', type=int, default=2, help='Number of layers in the RNN')
argparser.add_argument('--learning-rate', type=float, default=0.005, help='Learning rate for the optimizer')
argparser.add_argument('--example-length', type=int, default=200, help='Length of each training example')
argparser.add_argument('--batch-size', type=int, default=32, help='Batch size for minibatch training')
args = argparser.parse_args()

n_epochs = args.n_epochs
print_every = args.print_every
hidden_size = args.hidden_size
n_layers = args.n_layers
lr = args.learning_rate
example_length = args.example_length
batch_size = args.batch_size
n_characters = len(all_characters)

decoder = RNN(n_characters, hidden_size, n_characters, n_layers)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=lr)
criterion = nn.CrossEntropyLoss()

file = read_data(args.filename)


def train(input_batch, target_batch):
    hidden = decoder.init_hidden(batch_size)
    decoder.zero_grad()
    loss = 0

    target_indices = torch.argmax(target_batch, dim=2)
    
    output, hidden = decoder(input_batch, hidden)

    output_reshaped = output.view(-1, output.size(-1))
    target_reshaped = target_indices.view(-1)
    
    # Calculate loss
    loss = criterion(output_reshaped, target_reshaped)
    
    loss.backward()
    decoder_optimizer.step()

    return loss.item()


try:
    for epoch in range(1, n_epochs + 1):
        input_batch, target_batch = create_minibatch(file, batch_size=batch_size, subset_length=example_length)
        loss = train(input_batch, target_batch)
        if epoch % print_every == 0:
            print('[(Training progress: %d%%) loss = %.4f]' % ( epoch / n_epochs * 100, loss))

    torch.save(decoder, 'short-phrase-generation.pt')
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed with error: {str(e)}")