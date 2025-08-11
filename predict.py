from data import *
import sys
import argparse

decoder = torch.load('short-phrase-generation.pt', weights_only=False)

def generate(starting_string='A', max_words=3, temperature=0.8, top_k=5):
    hidden = decoder.init_hidden(batch_size=1)
    starting_input = text_to_tensor(starting_string)
    predicted = starting_string

    for p in range(len(starting_string) - 1):
        char_input = starting_input[p:p+1].unsqueeze(0)
        _, hidden = decoder(char_input, hidden)
    
    inp = starting_input[-1:]
    
    word_count = 0
    while word_count < max_words:
        char_input = inp.unsqueeze(0)
        output, hidden = decoder(char_input, hidden)
        
        output = output[0, -1]
        
        output_dist = output.div(temperature)
        top_probs, top_indices = torch.topk(output_dist, top_k)
        probs = torch.softmax(top_probs, dim=0)
        top_i = top_indices[torch.multinomial(probs, 1)[0]]
        predicted_char = index_to_char(top_i)
        predicted += predicted_char
        inp = text_to_tensor(predicted_char)

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