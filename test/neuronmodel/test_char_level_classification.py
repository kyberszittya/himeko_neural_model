import time

import torch
import numpy as np

from himeko_neural_model.src.foundations.engine.engine_charlevel import CharLevelMultiClassEngine
from himeko_neural_model.src.foundations.factory.factory_vocab import FactoryVocabularyNode
from himeko_neural_model.src.neuronmodel.char_level_lstm import CharLevelLstmMultiClass, \
    CharLevelLstmMultiClassGenerator

CNT_TEST_CHAR_NUMBER = 76
CNT_TEST_NUM_CLASSES = 6
CNT_TEST_NUM_LAYERS = 1
CNT_TEST_MULTI_NUM_LAYERS = 4
CNT_TEST_HIDDEN_SIZE = 128

CLASSES = sorted(['city', 'number', 'name', 'email', 'profession', 'significant', 'education'])

VOCABULARY_PATH = "../../../text_datalist/vocab/vocab-char.txt"

def test_text_element_classification_tensor():
    vocab = FactoryVocabularyNode.create_vertex_default("vocab_node0", time.time_ns())
    vocab.load_vocabulary("../../../text_datalist/vocab/vocab-char.txt")
    #
    assert ' ' in vocab.vocab
    assert 'A' in vocab.vocab
    assert 'y' in vocab.vocab
    assert 'l' in vocab.vocab
    assert '_' in vocab.vocab
    assert 'ü' not in vocab.vocab
    assert 'ő' not in vocab.vocab
    assert ',' in vocab.vocab
    assert 'Ő' not in vocab.vocab
    assert 'Ű' not in vocab.vocab
    #
    classes = CLASSES
    hyperparameters = {
        "classes": classes,
        "char_number": vocab.vocab_len,
        "num_classes": len(classes),
        "num_layers": CNT_TEST_NUM_LAYERS,
        "hidden_size": CNT_TEST_HIDDEN_SIZE
    }
    assert hyperparameters['char_number'] == CNT_TEST_CHAR_NUMBER
    hot_tensor = vocab.line_to_one_hot_tensor("slim shady")
    assert hot_tensor.shape[0] == 10
    assert hot_tensor.shape[1] == CNT_TEST_CHAR_NUMBER


def softmax_to_prediction(softmax_output, classes):
    return classes[torch.argmax(softmax_output).item()]


def test_text_element_classification_classification():
    vocab = FactoryVocabularyNode.create_vertex_default("vocab_node0", time.time_ns())
    vocab.load_vocabulary(VOCABULARY_PATH)
    classes = CLASSES
    hyperparameters = {
        "classes": classes,
        "char_number": vocab.vocab_len,
        "num_classes": len(classes),
        "num_layers": CNT_TEST_NUM_LAYERS,
        "hidden_size": CNT_TEST_HIDDEN_SIZE
    }
    assert hyperparameters['char_number'] == CNT_TEST_CHAR_NUMBER
    hot_tensor = vocab.line_to_one_hot_tensor("slim shady")
    assert hot_tensor.shape[0] == 10
    assert hot_tensor.shape[1] == CNT_TEST_CHAR_NUMBER
    lstm_model: CharLevelLstmMultiClass = CharLevelLstmMultiClassGenerator.generate(hyperparameters)
    lstm_model.load_state_dict(torch.load("../../../data/weights/text_element_classification/text_element_classification_combined_sorted_nadam_2023_9_20_21_16_44"))
    device = 'cpu'
    print(f'Running on {device}')
    h0, c0 = lstm_model.init_hidden(device)
    input_tensor = vocab.line_to_one_hot_tensor('1567').to(device).unsqueeze(0)
    output, next_hidden, cell_state = lstm_model(input_tensor.type(torch.FloatTensor).to(device), h0, c0)
    prediction = softmax_to_prediction(output, classes)
    print(f'\nPrediction: {prediction}')
    input_tensor = vocab.line_to_one_hot_tensor('Zsolt').to(device).unsqueeze(0)
    output, next_hidden, cell_state = lstm_model(input_tensor.type(torch.FloatTensor).to(device), h0, c0)
    prediction = softmax_to_prediction(output, classes)
    print(f'\nPrediction: {prediction}')
    input_tensor = vocab.line_to_one_hot_tensor('Nemesrembehollos').to(device).unsqueeze(0)
    output, next_hidden, cell_state = lstm_model(input_tensor.type(torch.FloatTensor).to(device), h0, c0)
    prediction = softmax_to_prediction(output, classes)
    print(f'\nPrediction: {prediction}')
    input_tensor = vocab.line_to_one_hot_tensor('Budakalasz').to(device).unsqueeze(0)
    output, next_hidden, cell_state = lstm_model(input_tensor.type(torch.FloatTensor).to(device), h0, c0)
    prediction = softmax_to_prediction(output, classes)
    print(f'\nPrediction: {prediction}')
    input_tensor = vocab.line_to_one_hot_tensor('farago.szabolcs@gomail.hu').to(device).unsqueeze(0)
    output, next_hidden, cell_state = lstm_model(input_tensor.type(torch.FloatTensor).to(device), h0, c0)
    prediction = softmax_to_prediction(output, classes)
    print(f'\nPrediction: {prediction}')


def test_text_element_classification_node():
    engine = CharLevelMultiClassEngine()
    vocab = FactoryVocabularyNode.create_vertex_default("vocab_node0", time.time_ns())
    vocab.load_vocabulary(VOCABULARY_PATH)

