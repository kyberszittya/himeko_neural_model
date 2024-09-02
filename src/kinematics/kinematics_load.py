from himeko.transformations.text.generate_text import generate_text
from processing.parse_description import ParseDescriptionEdgeFromFile, ParseDescriptionEdge


def main():
    e = ParseDescriptionEdgeFromFile("parse_edge", 0, 0, b'0', b'0', "label", None)
    p = "../../data/kinematics/chicken_kinematics.himeko"
    library_path = "../../data/kinematics/"
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


if __name__ == "__main__":
    main()
