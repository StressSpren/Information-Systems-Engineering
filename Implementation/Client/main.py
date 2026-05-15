import sys
import os
from PIL import Image
from datetime import datetime

from PyQt6.QtWidgets import (
    QApplication,
    QWidget,
    QLabel,
    QVBoxLayout,
    QDialog,
    QHBoxLayout,
    QScrollArea
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap, QFont

import functions.predictions
from functions.model import Net
from functions.ec2 import upload_ec2 as upe

ALLOWED_EXTENSIONS = [".jpg", ".jpeg", ".png", ".tiff"]
base_folder = "predicted_images"
os.makedirs(base_folder, exist_ok=True)

class PredictionWindow(QDialog):
    def __init__(self, predictions):
        super().__init__()
        self.setWindowTitle("Predictions")
        self.resize(500, 400)

        scroll = QScrollArea()
        container = QWidget()
        layout = QVBoxLayout(container)

        for img_path, prediction in predictions:
            row = QHBoxLayout()

            img_label = QLabel()
            pixmap = QPixmap(img_path)
            img_label.setPixmap(pixmap)
            img_label.setScaledContents(True)
            img_label.setFixedSize(100, 100)

            text_label = QLabel(f"Predicted: {prediction}")
            text_label.setFont(QFont("Arial", 14))

            row.addWidget(img_label)
            row.addWidget(text_label)
            layout.addLayout(row)

        scroll.setWidget(container)
        scroll.setWidgetResizable(True)

        main_layout = QVBoxLayout(self)
        main_layout.addWidget(scroll)

class DragDropWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Suffolk Archives")
        self.setGeometry(100, 100, 400, 100)
        self.setAcceptDrops(True)

        layout = QVBoxLayout()
        self.setLayout(layout)

        self.drop_label = QLabel("Drag and drop an image or folder here")
        self.drop_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.drop_label.setStyleSheet(
            "border: 2px dashed #aaa; font-size: 14px; padding: 40px;"
        )
        layout.addWidget(self.drop_label)

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.accept()
            self.drop_label.setStyleSheet(
                "border: 2px dashed #00f; font-size: 14px; padding: 40px; background-color: #eef;"
            )
        else:
            event.ignore()
            
    def dragLeaveEvent(self, event):
        self.drop_label.setStyleSheet(
            "border: 2px dashed #aaa; font-size: 14px; padding: 40px;"
        )

    def dropEvent(self, event):
        self.drop_label.setStyleSheet(
            "border: 2px dashed #aaa; font-size: 14px; padding: 40px;"
        )
        
        urls = event.mimeData().urls()
        predictions = []

        for url in urls:
            path = url.toLocalFile()

            if os.path.isfile(path) and path[-4:].lower() in ALLOWED_EXTENSIONS:
                pred = self.process_file(path)
                predictions.append((path, pred))

            elif os.path.isdir(path):
                for i in os.listdir(path):
                    file_path = os.path.join(path, i)
                    if os.path.isfile(file_path) and i[-4:].lower() in ALLOWED_EXTENSIONS:
                        pred = self.process_file(file_path)
                        predictions.append((file_path, pred))

        if predictions:
            self.show_predictions(predictions)

    def show_predictions(self, predictions):
        self.pred_window = PredictionWindow(predictions)
        self.pred_window.exec()

    def process_file(self, file_path):
        pred = functions.predictions.predict_image(file_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        image = Image.open(file_path)
        pred_folder = os.path.join(base_folder, str(pred))
        os.makedirs(pred_folder, exist_ok=True)
        ext = os.path.splitext(file_path)[1].lower()
        filename = f"{pred}_{timestamp}{ext}"
        save_path = os.path.join(pred_folder, filename)
        image.save(save_path)
        
        # Uploading images to S3 bucket for dataset
        upe(pred, f'predicted_images/{pred}/{filename}')
        print(f"Saved: [{filename}]")
        return pred

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = DragDropWindow()
    window.show()
    sys.exit(app.exec())
