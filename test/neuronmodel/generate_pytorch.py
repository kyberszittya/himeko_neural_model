from himeko.hbcm.elements.edge import HyperEdge
from himeko.hbcm.elements.executable.edge import ExecutableHyperEdge
from himeko.hbcm.elements.vertex import HyperVertex


INDENT_VAL = 4

class GenerateTorchCode(ExecutableHyperEdge):

    def __init__(self, name: str, timestamp: int, serial: int, guid: bytes, suid: bytes, label: str, parent):
        ExecutableHyperEdge.__init__(self, name, timestamp, serial, guid, suid, label, parent)
        self.linear_el = None
        self.activation_el = None
        self.activation_el_relu = None
        self.activation_el_softmax = None

    def load_meta(self, meta: HyperVertex):
        self.linear_el = meta["elements"]["linear"]
        self.activation_el = meta["activations"]["activation"]
        self.activation_el_relu = meta["activations"]["relu"]
        self.activation_el_softmax = meta["activations"]["relu"]


    def generate_torch_code(self, root: HyperVertex, root_edge: HyperEdge):
        text_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F


"""
        # Iterate over layers
        class_name = root.name.capitalize()
        text_class = text_imports + f"""class {class_name}(nn.Module):\n"""
        indent = 0
        indent += INDENT_VAL
        text_class += f"\n{' ' * indent}def __init__(self):"
        root_indent = indent
        indent += INDENT_VAL
        text_class += f"\n{' ' * indent}super({class_name}, self).__init__()"
        input_layers = set()
        output_layers = set()
        for layer in root.get_subelements(lambda x: isinstance(x, HyperVertex) and "neuron" in x.stereotype.nameset):
            layer: HyperVertex
            if self.linear_el in layer.stereotype.elements:
                # TODO overwrite stereotype values with descendant values
                text_class += f"\n{' ' * indent}self.{layer.name} = nn.Linear({int(layer['in_features'].value)}, {int(layer['out_features'].value)})"
                input_layers.add(layer)
                output_layers.add(layer)
        # Trail with newlines
        text_class += "\n"
        # Iterate through connections to construct forward pass
        indent = root_indent
        text_forward = f"""\n{' ' * indent}def forward(self, x):"""
        indent += INDENT_VAL
        # Get input connections
        for conn in root_edge.get_subelements(lambda x: isinstance(x, HyperEdge) and "activation" in x.stereotype.nameset):
            conn: HyperEdge
            for t in conn.in_relations():
                output_layers.remove(t.target)
            for t in conn.out_relations():
                input_layers.remove(t.target)
        # Get intermediate connections
        for conn in root_edge.get_subelements(lambda x: isinstance(x, HyperEdge) and "activation" in x.stereotype.nameset):
            conn: HyperEdge
            for c in conn.in_relations():
                text_forward += f"\n{' ' * indent}x = self.{c.target.name}(x)"
            if self.activation_el_relu in conn.stereotype.elements:
                text_forward += f"\n{' ' * indent}x = F.relu(x)"
            # Check output connection
            if conn.cnt_out_relations == 0 and conn.cnt_in_relations > 0:
                text_forward += f"\n{' ' * indent}return x"


        # Add output connections
        for layer in output_layers:
            text_forward += f"\n{' ' * indent}x = self.{layer.name}(x)"
            # Return the output
            text_forward += f"\n{' ' * indent}return x"

        text_class += text_forward
        text_class += "\n"
        return text_class

    def operate(self, *args, **kwargs):
        root = kwargs["root"]
        root_edge = kwargs["root_edge"]
        text_class = self.generate_torch_code(root, root_edge)
        return text_class