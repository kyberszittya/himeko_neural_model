from processing.parse_description import ParseDescriptionEdge


def main():
    e = ParseDescriptionEdge("parse_edge", 0, 0, b'0', b'0', "label", None)
    p = "../../data/kinematics/chicken_kinematics.himeko"
    library_path = "../../data/kinematics/"
    h = e.execute(path=p, library_path=library_path)
    root = h[-1]
    print(root.name)


if __name__ == "__main__":
    main()
