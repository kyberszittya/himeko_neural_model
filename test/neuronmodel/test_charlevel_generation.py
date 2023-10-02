from himeko_neural_model.src.neuronmodel.char_level_lstm import CharLevelLstmMultiClassGenerator

CNT_TEST_CHAR_NUMBER = 65
CNT_TEST_NUM_CLASSES = 6
CNT_TEST_NUM_LAYERS = 1
CNT_TEST_MULTI_NUM_LAYERS = 4
CNT_TEST_HIDDEN_SIZE = 128


def test_charlevel_multiclass_generator():
    hyperparameters = {
        "char_number": CNT_TEST_CHAR_NUMBER,
        "num_classes": CNT_TEST_NUM_CLASSES,
        "num_layers": CNT_TEST_NUM_LAYERS,
        "hidden_size": CNT_TEST_HIDDEN_SIZE
    }
    factory = CharLevelLstmMultiClassGenerator()
    model = factory.generate(hyperparameters)
    print(model)
    assert model.rnn.input_size == CNT_TEST_CHAR_NUMBER
    assert model.rnn.hidden_size == CNT_TEST_HIDDEN_SIZE
    assert model.fc.out_features == CNT_TEST_NUM_CLASSES


def test_charlevel_multiclass_generator_multilayer():
    hyperparameters = {
        "char_number": CNT_TEST_CHAR_NUMBER,
        "num_classes": CNT_TEST_NUM_CLASSES,
        "num_layers": CNT_TEST_MULTI_NUM_LAYERS,
        "hidden_size": CNT_TEST_HIDDEN_SIZE
    }
    factory = CharLevelLstmMultiClassGenerator()
    model = factory.generate(hyperparameters)
    print(model)
    assert model.rnn.input_size == CNT_TEST_CHAR_NUMBER
    assert model.rnn.hidden_size == CNT_TEST_HIDDEN_SIZE
    assert model.rnn.num_layers == CNT_TEST_MULTI_NUM_LAYERS
    assert model.fc.out_features == CNT_TEST_NUM_CLASSES
