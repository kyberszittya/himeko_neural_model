import sys
import json
import typing

import cv2
import numpy as np
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget, QTreeWidget, QTreeWidgetItem, QHBoxLayout, QSizePolicy
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QColor
from PyQt6.QtCore import Qt, QPoint, QRectF


class Keypoint:
    """Egy kulcspont objektum osztálya"""
    def __init__(self, x, y, label, parent=None):
        self.x = x
        self.y = y
        self.label = label
        self.parent = parent  # Szülő keypoint
        self.children = []    # Gyermek keypointok listája


class Skeleton:
    """Egy váz objektum osztálya"""
    def __init__(self, keypoints):
        self.keypoints = keypoints  # A vázhoz tartozó kulcspontok listája
        self.connections = []       # Kapcsolatok listája



class AnnotatorState():

    def __init__(self):
        self.image = None
        self.pixmap = None
        self.keypoints: typing.Dict[str, Keypoint] = {}
        self.scale_factor = 1.0
        self.selected_point: typing.Optional[Keypoint] = None




class KeypointAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Keypoint Annotator with Fixed Window")
        self.setGeometry(100, 100, 1000, 600)

        # UI elemek
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.load_button = QPushButton("Kép betöltése")
        self.save_button = QPushButton("Mentés JSON-ként")
        self.clear_button = QPushButton("Törlés")

        # Fa-struktúra
        self.tree_widget = QTreeWidget()
        self.tree_widget.setHeaderLabels(["Keypoint", "X", "Y"])

        # Layout
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.load_button)
        button_layout.addWidget(self.save_button)
        button_layout.addWidget(self.clear_button)

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.image_label)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.tree_widget)

        self.setLayout(main_layout)

        # Events
        self.load_button.clicked.connect(self.load_image)
        self.save_button.clicked.connect(self.save_annotations)
        self.clear_button.clicked.connect(self.clear_annotations)

        self.state = AnnotatorState()

    def load_image(self):
        """Kép betöltése az ablak méretének megváltoztatása nélkül"""
        file_path, _ = QFileDialog.getOpenFileName(self, "Kép betöltése", "", "Images (*.png *.jpg *.jpeg)")
        if file_path:
            self.state.image = cv2.imread(file_path)
            self.state.image = cv2.cvtColor(self.state.image, cv2.COLOR_BGR2RGB)
            self.display_image()

    def display_image(self):
        """Megjeleníti és skálázza a képet az ablak méretéhez, de az ablak mérete nem változik"""
        if self.state.image is not None:
            height, width, channel = self.state.image.shape
            bytes_per_line = channel * width
            q_image = QImage(self.state.image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888)
            self.state.pixmap = QPixmap.fromImage(q_image)

            self.resize_image_to_fit()
            self.update_display()

    def resize_image_to_fit(self):
        """Az `image_label` méretéhez igazítja a képet és a kulcspontokat, de az ablak mérete változatlan marad"""
        if self.state.pixmap is not None and not self.image_label.size().isEmpty():
            self.scaled_pixmap = self.state.pixmap.scaled(
                self.image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio
            )

            # Skálázási arány kiszámítása
            original_size = self.state.pixmap.size()
            self.state.scale_factor = self.scaled_pixmap.width() / original_size.width()

            self.image_label.setPixmap(self.scaled_pixmap)
            self.update_display()

    def resizeEvent(self, event):
        """Ablak átméretezésekor a kép is újraméreteződik, de az ablak maga nem változik"""
        self.resize_image_to_fit()

    def update_display(self):
        """Frissíti a képet és a kulcspontokat a skálázás figyelembevételével"""
        if self.scaled_pixmap is not None:
            temp_pixmap = self.scaled_pixmap.copy()
            painter = QPainter(temp_pixmap)
            pen = QPen(QColor(255, 0, 0, 150))  # Fekete vonalak a kapcsolatokhoz
            pen.setWidth(2)
            painter.setPen(pen)

            for keypoint in self.state.keypoints.values():

                if keypoint.parent is not None:
                    # Skálázott koordináták számítása
                    parent_x = int(keypoint.parent.x * self.state.scale_factor)
                    parent_y = int(keypoint.parent.y * self.state.scale_factor)
                    child_x = int(keypoint.x * self.state.scale_factor)
                    child_y = int(keypoint.y * self.state.scale_factor)

                    # Vonal rajzolása a szülő és a gyermek között
                    painter.drawLine(QPoint(parent_x, parent_y), QPoint(child_x, child_y))

            for i, keypoint in enumerate(self.state.keypoints.values()):
                scaled_x = int(keypoint.x * self.state.scale_factor)
                scaled_y = int(keypoint.y * self.state.scale_factor)

                if keypoint == self.state.selected_point:
                    color = QColor(0, 0, 255, 150)  # Kék szín
                else:
                    color = QColor(255, 0, 0, 150)  # Piros szín
                pen = QPen(color)
                pen.setWidth(3)
                painter.setPen(pen)
                painter.setBrush(color)
                painter.drawEllipse(QPoint(scaled_x, scaled_y), 8, 8)
                # Név megjelenítése a kör mellett
                painter.drawText(QPoint(scaled_x + 10, scaled_y - 10), keypoint.label)
            painter.end()
            self.image_label.setPixmap(temp_pixmap)

    def mouseMoveEvent(self, event):
        """Egérrel kulcspontok mozgatása"""
        if self.state.image is not None and self.state.selected_point is not None:
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            pixmap_width = self.scaled_pixmap.width()
            pixmap_height = self.scaled_pixmap.height()

            # Kép középre igazításának kiszámítása
            offset_x = (label_width - pixmap_width) // 2
            offset_y = (label_height - pixmap_height) // 2

            # Új koordináták kiszámítása
            move_x = (event.position().x() - self.image_label.x() - offset_x) / self.state.scale_factor
            move_y = (event.position().y() - self.image_label.y() - offset_y) / self.state.scale_factor

            # Csak akkor frissítjük a koordinátákat, ha az érvényes területen belül maradunk
            if 0 <= move_x < self.state.image.shape[1] and 0 <= move_y < self.state.image.shape[0]:
                self.state.selected_point.x = int(move_x)
                self.state.selected_point.y = int(move_y)
                self.update_tree_view()
                self.update_display()

    def mouseReleaseEvent(self, event):
        """Egérgomb felengedése után megszünteti a kiválasztást"""
        #self.state.selected_point = None
        pass

    def delete_keypoint(self, keypoint):
        """Törli a kijelölt pontot és annak gyermekeit"""

        def remove_children_recursive(point):
            """Rekurzív törlés, eltávolítja az összes gyermek pontot is"""
            for child in point.children:
                remove_children_recursive(child)
            lbl = point.label
            if lbl in self.state.keypoints:
                self.state.keypoints.pop(lbl)

        # Kijelölt pont eltávolítása
        keypoint_to_remove = keypoint

        # Ha van szülője, eltávolítjuk a szülő gyermekei közül
        if keypoint_to_remove.parent:
            keypoint_to_remove.parent.children.remove(keypoint_to_remove)

        # Töröljük az összes gyermekével együtt
        remove_children_recursive(keypoint_to_remove)

        # Ha az eltávolított pont volt a kiválasztott, akkor nincs kiválasztott pont
        self.state.selected_point = None

        self.update_tree_view()
        self.update_display()

    def keyPressEvent(self, event):
        """Billentyűleütések kezelése"""
        if event.key() == Qt.Key.Key_D:
            if self.state.selected_point is not None:
                self.delete_keypoint(self.state.selected_point)


    def mousePressEvent(self, event):
        """Egérkattintással új keypoint hozzáadása"""
        if self.state.image is not None and event.button() == Qt.MouseButton.LeftButton:
            label_width = self.image_label.width()
            label_height = self.image_label.height()
            pixmap_width = self.scaled_pixmap.width()
            pixmap_height = self.scaled_pixmap.height()

            # Kép középre igazításának kiszámítása
            offset_x = (label_width - pixmap_width) // 2
            offset_y = (label_height - pixmap_height) // 2

            # Kattintási pozíciót a valódi kép koordinátákhoz igazítjuk
            click_x = (event.position().x() - self.image_label.x() - offset_x) / self.state.scale_factor
            click_y = (event.position().y() - self.image_label.y() - offset_y) / self.state.scale_factor

            sensitivity = max(10, min(25, int(20 / self.state.scale_factor)))

            for i, keypoint in enumerate(self.state.keypoints.values()):
                if abs(click_x - keypoint.x) <= sensitivity and abs(click_y - keypoint.y) <= sensitivity:

                    if event.button() == Qt.MouseButton.RightButton:
                        self.delete_keypoint(i)
                        return
                    else:
                        self.state.selected_point = keypoint  # Kiválasztjuk a mozgatandó pontot
                        self.update_display()
                        return

                # Ellenőrizzük, hogy a kattintás a képen belül van-e
            if 0 <= click_x < self.state.image.shape[1] and 0 <= click_y < self.state.image.shape[0]:
                label = f"Point {len(self.state.keypoints) + 1}"

                # Ha van kijelölt pont, akkor az lesz a szülő
                parent_keypoint = None
                if self.state.selected_point is not None:
                    parent_keypoint = self.state.selected_point

                new_keypoint = Keypoint(int(click_x), int(click_y), label, parent_keypoint)

                # Ha van szülő, hozzáadjuk a gyermekek listájához
                if parent_keypoint:
                    parent_keypoint.children.append(new_keypoint)

                self.state.keypoints[label] = new_keypoint

                # Az új pont lesz az alapértelmezett kiválasztott pont
                self.state.selected_point = new_keypoint
                self.update_tree_view()
                self.update_display()


    def update_tree_view(self):
        """Frissíti a fa nézetet"""
        self.tree_widget.clear()

        # Segédfüggvény a hierarchia megjelenítésére
        def add_tree_item(parent_item, kp):
            item = QTreeWidgetItem([kp.label, str(kp.x), str(kp.y)])
            parent_item.addChild(item)
            for child in kp.children:
                add_tree_item(item, child)

        # Gyökér csomópontok hozzáadása
        for keypoint in self.state.keypoints.values():
            if keypoint.parent is None:  # Csak a legfelső szintű pontokat adjuk hozzá
                root_item = QTreeWidgetItem([keypoint.label, str(keypoint.x), str(keypoint.y)])
                self.tree_widget.addTopLevelItem(root_item)
                for child in keypoint.children:
                    add_tree_item(root_item, child)

    def save_annotations(self):
        """Mentés COCO JSON formátumban"""
        if not self.state.keypoints or self.state.image is None:
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Mentés JSON-ként", "", "JSON Files (*.json)")
        if save_path:
            data = {
                "images": [{"id": 1, "file_name": "annotated_image.jpg"}],
                "annotations": [{
                    "image_id": 1,
                    "keypoints": [coord for point in self.state.keypoints for coord in (point[0], point[1], 2)],
                    "num_keypoints": len(self.state.keypoints)
                }]
            }
            with open(save_path, 'w') as f:
                json.dump(data, f, indent=4)

    def clear_annotations(self):
        """Kulcspontok törlése"""
        self.state.keypoints = []
        self.tree_widget.clear()
        self.update_display()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = KeypointAnnotator()
    window.show()
    sys.exit(app.exec())
