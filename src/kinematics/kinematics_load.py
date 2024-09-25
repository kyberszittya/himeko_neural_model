from himeko.hbcm.elements.edge import EnumRelationDirection
from himeko.hbcm.factories.creation_elements import FactoryHypergraphElements
from himeko.hbcm.visualization.graphviz import create_dot_graph, visualize_dot_graph
from himeko.transformations.text.generate_text import generate_text
from processing.parse_description import ParseDescriptionEdgeFromFile, ParseDescriptionEdge


def main():
    engine = FactoryHypergraphElements.create_vertex_default("engine", 0)
    n_i0 = FactoryHypergraphElements.create_vertex_default("dataset", 0, engine)
    n_h0 = FactoryHypergraphElements.create_vertex_default("hidden", 0, engine)
    n_o0 = FactoryHypergraphElements.create_vertex_default("output", 0, engine)
    e_parse: ParseDescriptionEdgeFromFile = FactoryHypergraphElements.create_edge_constructor_default(
        ParseDescriptionEdgeFromFile, "parse_edge", 0, engine)
    e_parse += (n_i0, EnumRelationDirection.IN, 1.0)
    e_parse += (n_h0, EnumRelationDirection.OUT, 1.0)
    p = "../../data/kinematics/chicken_kinematics.himeko"
    library_path = "../../data/kinematics/"
    FactoryHypergraphElements.create_attribute_default("path", p, "string", 0, n_i0)
    FactoryHypergraphElements.create_attribute_default("library_path", library_path, "string", 0, n_i0)
    e_parse.execute()
    print(n_h0["root"])
    G = create_dot_graph(n_h0["root"], composition=True, stereotype=True)
    visualize_dot_graph(G, "test.png")

    """    
    h = e.execute(path=p, library_path=library_path)
    root = h[-1]
    print(root.name)
    print([x.name for x in root["right_leg"].get_all_children(lambda x: True)])
    print([x.name for x in root["left_leg"].get_all_children(lambda x: True)])
    print([x.name for x in root["right_wing"].get_all_children(lambda x: True)])
    print([x.name for x in root["left_wing"].get_all_children(lambda x: True)])
    print(generate_text(root))
    root["epistropheus"]["position"].value[0] = 5.0

    root["left_wing"]["left_wing_coracoid"]["position"].value[0] = 5.0

    h_text = ParseDescriptionEdge(
        "parse_edge", 0, 0, b'0', b'0', "label", None)
    h_text.execute(text=generate_text(root), library_path=library_path)
    """


if __name__ == "__main__":
    main()
