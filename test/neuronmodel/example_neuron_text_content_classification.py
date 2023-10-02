import time

import numpy as np
import torch
import unicodedata
import unidecode

from himeko_neural_model.src.foundations.factory.factory_vocab import FactoryVocabularyNode
from himeko_neural_model.src.neuronmodel.char_level_lstm import CharLevelLstmMultiClassGenerator, \
    CharLevelLstmMultiClass

def softmax_to_prediction(softmax_output, classes):
    return classes[torch.argmax(softmax_output).item()]

def main():
    vocab = FactoryVocabularyNode.create_vertex_default("vocab_node0", time.time_ns())
    vocab.load_vocabulary("../../../text_datalist/vocab/vocab-char-extended.txt")
    classes = sorted(['city', 'number', 'name', 'email', 'profession', 'significant', 'education', 'gt', 'other', 'lang'])
    hyperparameters = {
        "classes": classes,
        "char_number": vocab.vocab_len,
        "num_classes": len(classes),
        "num_layers": 1,
        "hidden_size": 128
    }
    lstm_model: CharLevelLstmMultiClass = CharLevelLstmMultiClassGenerator.generate(hyperparameters)
    lstm_model.load_state_dict(torch.load("../../../data/weights_full/text_element_classification_combined_sorted_nadam_50_2023_9_27_2_19_42"))
    device = 'cpu'
    print(f'Running on {device}')
    while True:
        h0, c0 = lstm_model.init_hidden(device)
        print("Input name: ")
        line = input()
        line = unidecode.unidecode(line)
        input_tensor = vocab.line_to_one_hot_tensor(line).to(device).unsqueeze(0)
        output, next_hidden, cell_state = lstm_model(input_tensor.type(torch.FloatTensor).to(device), h0, c0)
        prediction = softmax_to_prediction(output, classes)
        print(f'Prediction: {prediction}\t output classes: {output.detach().numpy()}')


if __name__ == "__main__":
    main()
