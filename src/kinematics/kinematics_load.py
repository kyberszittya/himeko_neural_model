from himeko.transformations.text.generate_text import generate_text
from processing.parse_description import ParseDescriptionEdge


def main():
    e = ParseDescriptionEdge("parse_edge", 0, 0, b'0', b'0', "label", None)
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


if __name__ == "__main__":
    main()
