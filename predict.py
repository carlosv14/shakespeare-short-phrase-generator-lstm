from data import *
import sys
import argparse

decoder = torch.load('short-phrase-generation.pt')

def generate(starting_string='A', max_words=3, temperature=0.8, top_k=5):
    hidden = decoder.init_hidden()
    starting_input = text_to_tensor(starting_string)
    predicted = starting_string

    for p in range(len(starting_string) - 1):
        _, hidden = decoder(starting_input[p], hidden)
    inp = starting_input[-1]
    
    word_count = 0
    while word_count < max_words:
        output, hidden = decoder(inp, hidden)
        
        output_dist = output.data.view(-1).div(temperature)
        top_probs, top_indices = torch.topk(output_dist, top_k)
        probs = torch.softmax(top_probs, dim=0)
        top_i = top_indices[torch.multinomial(probs, 1)[0]]
        predicted_char = index_to_char(top_i)
        predicted += predicted_char
        inp = text_to_tensor(predicted_char)[0]

        if predicted_char == ' ':
            word_count += 1

    return predicted

if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('input', type=str, default='A')
    argparser.add_argument('--word-count', type=int, default=3)
    argparser.add_argument('--temperature', type=float, default=0.8)
    args = argparser.parse_args()
    print(generate(args.input, args.word_count, args.temperature))