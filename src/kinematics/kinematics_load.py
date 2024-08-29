from processing.parse_description import ParseDescriptionEdge


def main():
    e = ParseDescriptionEdge("parse_edge", 0, 0, b'0', b'0', "label", None)
    p = "../../data/kinematics/chicken_kinematics.himeko"
    h = e.execute(path=p)
    print(h)


if __name__ == "__main__":
    main()
