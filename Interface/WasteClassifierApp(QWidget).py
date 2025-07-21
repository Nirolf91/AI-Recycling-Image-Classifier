import sys
import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from PyQt5.QtWidgets import QApplication, QLabel, QWidget, QVBoxLayout, QPushButton, QProgressBar, QGroupBox, QHBoxLayout

from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import QTimer
import torch.nn as nn
import urllib.request
import os
from datetime import datetime
import matplotlib.pyplot as plt


class WasteClassifierApp(QWidget):
    def __init__(self):

        super().__init__()
        self.setWindowTitle("Clasificare Deșeuri - ResNet101")
        self.setGeometry(200, 200, 800, 600)
        # Aplică stylesheet-ul extern
        with open("style.qss", "r") as f:
            self.setStyleSheet(f.read())
        # Layout principal
        self.layout = QVBoxLayout()
        self.frame_counter = 0

        # Titlu aplicație
        self.title_label = QLabel("Aplicație AI pentru clasificarea deșeurilor")
        self.title_label.setObjectName("title_label")
        self.layout.addWidget(self.title_label)

        # Etichetă pentru video
        self.image_label = QLabel("Camera este închisă.")
        self.image_label.setObjectName("camera_label")
        self.image_label.setFixedSize(640, 480)
        self.image_label.setScaledContents(False)

        image_container = QHBoxLayout()
        image_container.addStretch()
        image_container.addWidget(self.image_label)
        image_container.addStretch()
        self.layout.addLayout(image_container)

        # Etichetă pentru rezultat
        self.result_label = QLabel("")
        self.result_label.setStyleSheet("font-size: 20px; color: green;")
        self.layout.addWidget(self.result_label)

        # Bară încredere
        self.confidence_bar = QProgressBar()
        self.confidence_bar.setRange(0, 100)
        self.confidence_bar.setValue(0)
        self.layout.addWidget(self.confidence_bar)

        # Butoane
        button_layout = QHBoxLayout()

        self.start_button = QPushButton("Deschide Camera")
        self.start_button.setObjectName("start_button")
        self.start_button.setToolTip("Pornește camera pentru clasificarea deșeurilor")
        self.start_button.clicked.connect(self.start_camera)
        button_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Închide Camera")
        self.stop_button.setObjectName("stop_button")
        self.stop_button.clicked.connect(self.stop_camera)
        button_layout.addWidget(self.stop_button)

        self.graph_button = QPushButton("Afișează Grafic Statistic")
        self.graph_button.setObjectName("graph_button")
        self.graph_button.clicked.connect(self.plot_statistics)
        button_layout.addWidget(self.graph_button)

        self.layout.addLayout(button_layout)

        # 🔹 Încarcă modelul
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_path = "E:/Facultatea de Automatica si Calculatoare/Anul IV/Licenta/Modele/best_waste_classifier_13_resnet101.pth"
        self.model = models.resnet101(pretrained=False)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, 5)
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

        # 🔹 Transformări imagine
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        self.class_labels = ["Glass", "Metal", "Paper", "Plastic", "Trash"]

        self.results_dir = "clasificari"
        self.correct_dir = os.path.join(self.results_dir, "corecte")
        self.incorrect_dir = os.path.join(self.results_dir, "gresite")

        for base_dir in [self.correct_dir, self.incorrect_dir]:
            for cls in self.class_labels:
                os.makedirs(os.path.join(base_dir, cls), exist_ok=True)
        self.class_counts = {cls: {"total": 0, "corecte": 0} for cls in self.class_labels}

        # Grupă statisticile într-un QGroupBox
        stats_group = QGroupBox("Statistici clasificare")
        stats_layout = QVBoxLayout()

        self.stats_label = QLabel("Statistici clasificare:\n")
        stats_layout.addWidget(self.stats_label)

        stats_group.setLayout(stats_layout)
        self.layout.addWidget(stats_group)
        self.setLayout(self.layout)

        self.cap = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

    def start_camera(self):
        self.cap = cv2.VideoCapture("http://10.13.49.227:8080/video")  # actualizează IP-ul

        if self.cap.isOpened():
            self.result_label.setText("✅ Camera telefonului conectată!")
            self.timer.start(30)

        else:
            self.result_label.setText("❌ Fluxul video nu poate fi accesat.")
            print("❌ Eroare la deschiderea fluxului. Verifică IP-ul și conexiunea.")

        # Înlocuiește urllib cu VideoCapture pentru fluxul IP
        self.cap = cv2.VideoCapture("http://10.13.49.227:8080//video")  # IP Webcam
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        if self.cap.isOpened():
            self.result_label.setText("✅ Camera telefonului conectată!")
            self.timer.start(30)  # 30 ms pentru ~30 FPS
        else:
            self.result_label.setText("❌ Nu s-a putut accesa fluxul video.")
            print("❌ Eroare la deschiderea camerei DroidCam.")

    def stop_camera(self):
        if self.cap and self.cap.isOpened():
            self.timer.stop()
            self.cap.release()
            self.cap = None
            print("✅ Camera oprită corect.")

        self.image_label.clear()
        self.result_label.setText("Camera oprită.")
        self.image_label.setText("Camera este închisă.")

    def update_frame(self):
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Redă imaginea (fără clasificare)
                h, w, ch = rgb_image.shape
                bytes_per_line = ch * w
                q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
                self.image_label.setPixmap(QPixmap.fromImage(q_image))

                # Rulează clasificarea doar o dată la 5 frame-uri
                self.frame_counter += 1
                if self.frame_counter % 5 != 0:
                    return
                try:
                    pil_image = Image.fromarray(rgb_image)
                    img_tensor = self.transform(pil_image).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        outputs = self.model(img_tensor)
                        probabilities = torch.softmax(outputs, dim=1)
                        confidence, predicted_class = torch.max(probabilities, dim=1)
                        label = self.class_labels[predicted_class.item()]
                        confidence_pct = confidence.item() * 100

                    result_text = f"Predicție: {label} ({confidence_pct:.2f}%)"
                    self.result_label.setText(result_text)

                    self.confidence_bar.setValue(int(confidence_pct))

                    # 🔽 Salvează imaginea
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
                    filename = f"{label}_{confidence_pct:.2f}_{timestamp}.jpg"

                    self.class_counts[label]["total"] += 1

                    if confidence_pct > 70:
                        self.class_counts[label]["corecte"] += 1
                        save_path = os.path.join(self.correct_dir, label, filename)
                    else:
                        save_path = os.path.join(self.incorrect_dir, label, filename)

                    cv2.imwrite(save_path, cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

                    # 🔽 Actualizează statistici
                    stats_text = "Statistici clasificare:\n"
                    for cls in self.class_labels:
                        total = self.class_counts[cls]["total"]
                        corecte = self.class_counts[cls]["corecte"]
                        procent = (corecte / total * 100) if total > 0 else 0
                        stats_text += f"{cls}: {corecte}/{total} corecte ({procent:.1f}%)\n"

                    self.stats_label.setText(stats_text)

                except Exception as e:
                    print("❌ Eroare în procesare imagine:", e)
                    self.result_label.setText("❌ Eroare la procesarea imaginii.")

    def plot_statistics(self):
        labels = self.class_labels
        corecte = [self.class_counts[cls]["corecte"] for cls in labels]
        gresite = [self.class_counts[cls]["total"] - self.class_counts[cls]["corecte"] for cls in labels]

        x = np.arange(len(labels))
        width = 0.35

        fig, ax = plt.subplots()
        ax.bar(x - width / 2, corecte, width, label='Corecte', color='green')
        ax.bar(x + width / 2, gresite, width, label='Greșite', color='red')

        ax.set_ylabel('Număr imagini')
        ax.set_title('Clasificări corecte vs greșite')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()

        plt.tight_layout()
        plt.show()

    def closeEvent(self, event):
        self.stop_camera()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WasteClassifierApp()
    window.show()
    sys.exit(app.exec_())
