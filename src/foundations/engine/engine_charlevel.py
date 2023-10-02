import torch.nn as nn

import typing

import torch.optim

from himeko_neural_model.src.foundations.engine.engine import AbstractInferenceNeuralEngine
from himeko_neural_model.src.neuronmodel.char_level_lstm import CharLevelLstmMultiClassGenerator, \
    CharLevelLstmMultiClass


class CharLevelMultiClassEngine(AbstractInferenceNeuralEngine):


    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str,
                  parent: typing.Optional = None, neural_model: typing.Optional[nn.Module] = None,
                 hyperparameters: typing.Optional[typing.Dict] = None):
        generator: CharLevelLstmMultiClassGenerator = CharLevelLstmMultiClassGenerator()
        super().__init__(name, timestamp, serial, guid, suid, label, generator, parent)
        self._hyperparameters = hyperparameters
        if neural_model is None:
            if hyperparameters is None:
                raise ValueError("Hyperparameters are not set, unable to create model")
            self._check_hyperparameters(hyperparameters)
            self._neural_model: CharLevelLstmMultiClass = generator.generate(hyperparameters)
            self._neural_model.load_state_dict(torch.load(hyperparameters["weights_path"]))
        else:
            self._neural_model = neural_model
        # Setup learning
        if "online_lr" in hyperparameters and "offline_lr" in hyperparameters:
            self.online_optimizer = torch.optim.SGD(self._neural_model.parameters(), lr=hyperparameters["online_lr"])
            self.offline_optimizer = torch.optim.NAdam(self._neural_model.parameters(), lr=hyperparameters["offline_lr"])

    @staticmethod
    def _check_hyperparameters(hypm: typing.Dict):
        MSG_ = "not set in hyperparameters"
        if "classes" not in hypm:
            raise KeyError(f"Classes are {MSG_}")
        if "char_number" not in hypm:
            raise KeyError(f"Class number is {MSG_}")
        if "num_classes" not in hypm:
            raise KeyError(f"Number of classes is {MSG_}")
        if "num_layers" not in hypm:
            raise KeyError(f"Number of layers is {MSG_}")
        if "hidden_size" not in hypm:
            raise KeyError(f"Number of hidden size is {MSG_}")
        if "weights_path" not in hypm:
            raise KeyError(f"Weights path is {MSG_}")
        if "device" not in hypm:
            raise KeyError(f"Working device {MSG_}")

    def online_learning(self):
        pass

    def train(self):
        pass

    def evaluate(self):
        pass

    def test(self):
        pass

    def predict(self, x):
        h0, c0 = self._neural_model.init_hidden(self._hyperparameters['device'])
        return self._neural_model(x, h0, c0)

    def reflexive_train(self):
        pass
