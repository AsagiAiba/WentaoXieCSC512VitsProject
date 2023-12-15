import shutil
import sys

import soundfile as sf
import torch
from PyQt5.QtCore import QUrl, QTimer
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtWidgets import QApplication, QMainWindow, QTextEdit, QPushButton, QVBoxLayout, QWidget, QFileDialog, \
    QProgressBar

import commons
import utils
from models import SynthesizerTrn
from text import text_to_sequence
from text.symbols import symbols

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()

        self.text_edit = QTextEdit()
        self.play_button = QPushButton("Play")
        self.save_button = QPushButton("Save Audio As")
        self.progress_bar = QProgressBar()
        self.play_button.clicked.connect(self.play)
        self.save_button.clicked.connect(self.save)

        layout = QVBoxLayout()
        layout.addWidget(self.text_edit)
        layout.addWidget(self.play_button)
        layout.addWidget(self.save_button)
        layout.addWidget(self.progress_bar)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

        self.media_player = QMediaPlayer()
        self.media_player.durationChanged.connect(self.update_progress_bar)
        self.media_player.positionChanged.connect(self.update_progress_bar)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress_bar)

    def play(self):
        text = self.text_edit.toPlainText()
        if text:
            print(f'start = {text}')
            pre_wav(text)
            print(f'End')
            self.media_player.setVolume(100) 

            content = QMediaContent(QUrl.fromLocalFile("/Users/vits/pre.wav"))
            self.media_player.setMedia(content)
            self.media_player.play()
            self.timer.start(1000)

    def save(self):
        """Save audio as"""
        file_dialog = QFileDialog()
        save_path = file_dialog.getSaveFileName(self, "Save File", "./", "Audio Files (*.mp3)")
        if save_path[0]:
            current_url = self.media_player.media().canonicalUrl().toString()
            shutil.copy(str(current_url).split(":")[1], save_path[0])

    def update_progress_bar(self):
        duration = self.media_player.duration()
        position = self.media_player.position()
        self.progress_bar.setMaximum(duration)
        self.progress_bar.setValue(position)

def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm

def pre_wav(txt):
    stn_tst = get_text(txt,hps)
    with torch.no_grad():
        x_tst = stn_tst.cpu().unsqueeze(0)
        x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).cpu()
        audio = net_g.infer(x_tst, x_tst_lengths, noise_scale=.667, noise_scale_w=0.8, length_scale=1)[0][
            0, 0].data.cpu().float().numpy()
    sf.write('pre.wav', audio, 22050)

if __name__ == '__main__':
    device = torch.device('cpu')

    hps = utils.get_hparams_from_file("./configs/ljs_base.json")

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cpu()
    _ = net_g.eval()

    _ = utils.load_checkpoint("resources/ljs_base_2/G_52000.pth", net_g, None)

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
