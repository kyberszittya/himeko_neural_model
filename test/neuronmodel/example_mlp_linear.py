from himeko.hbcm.elements.edge import HyperEdge
from himeko.hbcm.elements.vertex import HyperVertex
from processing.parse_description import ParseDescriptionEdgeFromFile

import os

BASEDIR = os.path.dirname(os.path.abspath(__file__))
PROJECTDIR = os.path.dirname(os.path.join(BASEDIR, "..","..", ".."))

def main():
    parserEdge = ParseDescriptionEdgeFromFile("parse_neural", 0, 0, b'0', b'0', "label", None)
    print(parserEdge)
    print(PROJECTDIR)
    p = os.path.join(PROJECTDIR, "data/neuralnetwork/entropycalc/mlp.himeko")
    library_path = os.path.join(PROJECTDIR, "data/neuralnetwork/")
    h = parserEdge.execute(path=p, library_path=library_path)
    meta = h[0]
    linear_el = meta["elements"]["linear"]
    activation_el = meta["activations"]["activation"]
    activation_el_relu = meta["activations"]["relu"]
    activation_el_softmax = meta["activations"]["relu"]
    root = h[-1]
    print(root.name)
    root_edge: HyperEdge = root["connections"]
    print(root_edge.count_composite_elements)
    e1: HyperEdge = root_edge["e1"]
    # Text generation
    # Import statements
    text_imports = """import torch
import torch.nn as nn
import torch.nn.functional as F


"""
    # Iterate over layers
    class_name = root.name.capitalize()
    text_class = text_imports + f"""class {class_name}(nn.Module):"""
    indent = 0
    indent += 2
    text_class += f"\n{' ' * indent}def __init__(self):"
    root_indent = indent
    indent += 2
    text_class += f"\n{' ' * indent}super({class_name}, self).__init__()"
    input_layers = set()
    output_layers = set()
    for layer in root.get_subelements(lambda x: isinstance(x, HyperVertex) and "neuron" in x.stereotype.nameset):
        layer: HyperVertex
        if linear_el in layer.stereotype.elements:
            # TODO overwrite stereotype values with descendant values
            text_class += f"\n{' ' * indent}self.{layer.name} = nn.Linear({int(layer['in_features'].value)}, {int(layer['out_features'].value)})"
            input_layers.add(layer)
            output_layers.add(layer)
    # Trail with newlines
    text_class += "\n"
    # Iterate through connections to construct forward pass
    indent = root_indent
    text_forward = f"""\n{' ' * indent}def forward(self, x):"""
    indent += 2
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
        if activation_el_relu in conn.stereotype.elements:
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
    print(text_class)
    # Save to file
    with open(os.path.join(PROJECTDIR, "test/neuronmodel/generated_mlp.py"), "w") as f:
        f.write(text_class)

if __name__ == "__main__":
    main()
