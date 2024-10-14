from himeko.hbcm.elements.edge import HyperEdge
from himeko.hbcm.elements.vertex import HyperVertex
from generate_pytorch import GenerateTorchCode
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

    root = h[-1]
    print(root.name)
    root_edge: HyperEdge = root["connections"]
    print(root_edge.count_composite_elements)
    e1: HyperEdge = root_edge["e1"]
    # Text generation
    # Import statements
    gen_torch = GenerateTorchCode("generate_torch", 0, 0, b'0', b'0', "label", None)
    gen_torch.load_meta(meta)
    text_class = gen_torch(root= root, root_edge= root_edge)
    print(text_class)
    # Save to file
    with open(os.path.join(PROJECTDIR, "test/neuronmodel/generated_mlp.py"), "w") as f:
        f.write(text_class)


if __name__ == "__main__":
    main()
