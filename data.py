import unidecode
import random
import string
import torch
import torch.nn.functional as F

all_characters = string.printable

def read_data(filePath):
    text = unidecode.unidecode(open(filePath).read())
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    return text

def random_training_example(file, subset_length=100):
    start_index = random.randint(0, len(file) - subset_length - 1)
    end_index = start_index + subset_length + 1
    return file[start_index : end_index]

def text_to_tensor(example):
    indexes = [char_to_index(c) for c in example]
    one_hot_tensors = [F.one_hot(torch.tensor(i), num_classes=len(all_characters)).float() for i in indexes]
    return torch.stack(one_hot_tensors)

def char_to_index(char):
    return all_characters.index(char)

def index_to_char(index):
    return all_characters[index]

def random_training_set(file, subset_length=100):  
    example = random_training_example(file, subset_length)
    input_tensor = text_to_tensor(example[:-1])
    target_tensor = text_to_tensor(example[1:])
    return input_tensor, target_tensor

def create_minibatch(file, batch_size=32, subset_length=100):
    batch_inputs = []
    batch_targets = []
    
    for _ in range(batch_size):
        example = random_training_example(file, subset_length)
        input_tensor = text_to_tensor(example[:-1])
        target_tensor = text_to_tensor(example[1:])
        batch_inputs.append(input_tensor)
        batch_targets.append(target_tensor)
    
    batch_input = torch.stack(batch_inputs)
    batch_target = torch.stack(batch_targets)
    
    return batch_input, batch_target