#!/usr/bin/env python3
"""
realtime_av_gui.py

Aplikasi GUI real-time untuk menangkap audio (microphone) dan video (webcam),
melakukan:
 - Audio: real-time waveform + FFT, opsi bandpass filter
 - Video: tampilan webcam + opsi edge detection (Canny)
 - Controls: mulai/berhenti audio/video, slider parameter, snapshot, simpan audio sample (opsional)

Dependencies:
    pip install pyqt5 opencv-python sounddevice numpy scipy pyqtgraph

Tested on: Python 3.8+
"""

import sys
import time
import threading
from collections import deque

import numpy as np
import cv2
import sounddevice as sd
from scipy.signal import butter, sosfilt
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

# ----------------------------
# Configurable defaults
# ----------------------------
AUDIO_FS = 44100           # sampling rate for audio
AUDIO_BLOCKSIZE = 2048     # frames per audio block callback
AUDIO_CHANNELS = 1         # mono
VIDEO_DEVICE = 0           # default webcam index
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480

# ----------------------------
# Utility: audio filter
# ----------------------------
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], analog=False, btype='band', output='sos')
    return sos

# ----------------------------
# Audio capture thread (uses sounddevice callback)
# ----------------------------
class AudioStream:
    def __init__(self, fs=AUDIO_FS, blocksize=AUDIO_BLOCKSIZE, channels=AUDIO_CHANNELS):
        self.fs = fs
        self.blocksize = blocksize
        self.channels = channels

        self.stream = None
        # ring buffers for waveform and spectrum visualization
        self.wave_buffer = deque(maxlen=10 * blocksize)  # store recent samples
        self.fft_buffer = None

        self.sos = None  # filter coefficients (if any)
        self.running = False

        # simple lock for thread-safety
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return
        self.running = True
        self.stream = sd.InputStream(
            samplerate=self.fs,
            blocksize=self.blocksize,
            channels=self.channels,
            callback=self._callback
        )
        self.stream.start()

    def stop(self):
        if not self.running:
            return
        self.running = False
        if self.stream:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    def _callback(self, indata, frames, time_info, status):
        # indata shape: (frames, channels)
        if status:
            # non-fatal notifications are printed here
            print("Audio callback status:", status, file=sys.stderr)
        samples = indata[:, 0].copy() if self.channels == 1 else indata.mean(axis=1)
        with self.lock:
            if self.sos is not None:
                try:
                    samples = sosfilt(self.sos, samples)
                except Exception as e:
                    print("Filter error:", e)
            self.wave_buffer.extend(samples)
            # update fft_buffer for display (compute on block)
            windowed = samples * np.hanning(len(samples))
            fft = np.fft.rfft(windowed)
            mag = np.abs(fft) / (len(windowed)/2)
            self.fft_buffer = mag

    def set_bandpass(self, lowcut, highcut):
        if lowcut is None or highcut is None or lowcut <= 0 or highcut <= 0 or lowcut >= highcut:
            self.sos = None
            return
        self.sos = butter_bandpass(lowcut, highcut, self.fs, order=4)

    def get_waveform(self):
        with self.lock:
            return np.array(self.wave_buffer)

    def get_fft(self):
        with self.lock:
            return self.fft_buffer

# ----------------------------
# Video capture thread
# ----------------------------
class VideoThread(QtCore.QThread):
    frame_ready = QtCore.pyqtSignal(np.ndarray)
    def __init__(self, device=VIDEO_DEVICE, width=VIDEO_WIDTH, height=VIDEO_HEIGHT, parent=None):
        super().__init__(parent)
        self.device = device
        self.width = width
        self.height = height
        self.running = False
        self.do_edge = False
        self.cap = None

    def run(self):
        self.cap = cv2.VideoCapture(self.device, cv2.CAP_DSHOW if sys.platform.startswith("win") else 0)
        # set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        last_time = time.time()
        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            if self.do_edge:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                # convert edges to 3-channel for display
                frame = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            # emit frame (BGR)
            self.frame_ready.emit(frame)
            # small sleep to yield CPU
            time.sleep(0.01)
        if self.cap:
            self.cap.release()
            self.cap = None

    def stop(self):
        self.running = False
        self.wait(500)

# ----------------------------
# Main GUI Window
# ----------------------------
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Real-time Audio/Video Processing (FFT, Filter, Edge Detection)")
        self.resize(1100, 700)

        # central widget
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        layout = QtWidgets.QGridLayout(central)

        # Video display (QLabel)
        self.video_label = QtWidgets.QLabel()
        self.video_label.setFixedSize(VIDEO_WIDTH, VIDEO_HEIGHT)
        self.video_label.setStyleSheet("background: black")
        layout.addWidget(self.video_label, 0, 0, 3, 1)

        # Video controls
        self.btn_video_start = QtWidgets.QPushButton("Start Video")
        self.btn_video_stop = QtWidgets.QPushButton("Stop Video")
        self.btn_snapshot = QtWidgets.QPushButton("Snapshot")
        self.chk_edge = QtWidgets.QCheckBox("Edge Detection (Canny)")

        vbox_vid_ctrl = QtWidgets.QVBoxLayout()
        vbox_vid_ctrl.addWidget(self.btn_video_start)
        vbox_vid_ctrl.addWidget(self.btn_video_stop)
        vbox_vid_ctrl.addWidget(self.btn_snapshot)
        vbox_vid_ctrl.addWidget(self.chk_edge)
        layout.addLayout(vbox_vid_ctrl, 0, 1)

        # Audio plots using pyqtgraph
        pg.setConfigOptions(antialias=True)
        self.pg_wave = pg.PlotWidget(title="Audio Waveform (recent samples)")
        self.pg_wave.setLabel('left', 'Amplitude')
        self.pg_wave.setLabel('bottom', 'Samples')
        self.pg_wave_curve = self.pg_wave.plot(pen='y')

        self.pg_fft = pg.PlotWidget(title="Audio Spectrum (FFT)")
        self.pg_fft.setLabel('left', 'Magnitude')
        self.pg_fft.setLabel('bottom', 'Frequency (Hz)')
        self.pg_fft_curve = self.pg_fft.plot(pen='c')

        layout.addWidget(self.pg_wave, 1, 1)
        layout.addWidget(self.pg_fft, 2, 1)

        # Audio controls
        self.btn_audio_start = QtWidgets.QPushButton("Start Audio")
        self.btn_audio_stop = QtWidgets.QPushButton("Stop Audio")
        self.lowcut_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.lowcut_slider.setRange(20, 2000)
        self.lowcut_slider.setValue(100)
        self.highcut_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.highcut_slider.setRange(200, 12000)
        self.highcut_slider.setValue(5000)

        self.lbl_low = QtWidgets.QLabel("Lowcut: 100 Hz")
        self.lbl_high = QtWidgets.QLabel("Highcut: 5000 Hz")

        audio_ctrl_layout = QtWidgets.QVBoxLayout()
        audio_ctrl_layout.addWidget(self.btn_audio_start)
        audio_ctrl_layout.addWidget(self.btn_audio_stop)
        audio_ctrl_layout.addWidget(self.lbl_low)
        audio_ctrl_layout.addWidget(self.lowcut_slider)
        audio_ctrl_layout.addWidget(self.lbl_high)
        audio_ctrl_layout.addWidget(self.highcut_slider)

        layout.addLayout(audio_ctrl_layout, 0, 2, 3, 1)

        # Status bar
        self.status = QtWidgets.QLabel("Ready")
        layout.addWidget(self.status, 3, 0, 1, 3)

        # Instantiate audio/video handlers
        self.audio = AudioStream(fs=AUDIO_FS, blocksize=AUDIO_BLOCKSIZE)
        self.video_thread = VideoThread(device=VIDEO_DEVICE)

        # Connect signals
        self.btn_video_start.clicked.connect(self.start_video)
        self.btn_video_stop.clicked.connect(self.stop_video)
        self.btn_snapshot.clicked.connect(self.save_snapshot)
        self.chk_edge.stateChanged.connect(self.toggle_edge)

        self.btn_audio_start.clicked.connect(self.start_audio)
        self.btn_audio_stop.clicked.connect(self.stop_audio)
        self.lowcut_slider.valueChanged.connect(self.update_filter_labels)
        self.highcut_slider.valueChanged.connect(self.update_filter_labels)
        self.lowcut_slider.sliderReleased.connect(self.apply_filter)
        self.highcut_slider.sliderReleased.connect(self.apply_filter)

        self.video_thread.frame_ready.connect(self.on_frame)

        # Timer for updating audio plots
        self.timer = QtCore.QTimer()
        self.timer.setInterval(50)  # 20 Hz update
        self.timer.timeout.connect(self.update_plots)

        # For snapshot
        self.last_frame = None

        # For FPS
        self._last_frame_time = time.time()
        self._fps = 0.0

    # ----------------------------
    # Video functions
    # ----------------------------
    def start_video(self):
        if not self.video_thread.isRunning():
            self.video_thread = VideoThread(device=VIDEO_DEVICE)
            self.video_thread.do_edge = self.chk_edge.isChecked()
            self.video_thread.frame_ready.connect(self.on_frame)
            self.video_thread.start()
            self.status.setText("Video started")
        else:
            self.status.setText("Video already running")

    def stop_video(self):
        self.video_thread.stop()
        self.status.setText("Video stopped")

    def toggle_edge(self, state):
        self.video_thread.do_edge = bool(state)
        self.status.setText("Edge detection: " + ("ON" if state else "OFF"))

    def on_frame(self, frame_bgr):
        # Convert BGR -> QImage -> set to QLabel
        self.last_frame = frame_bgr
        # update FPS
        now = time.time()
        dt = now - self._last_frame_time
        self._last_frame_time = now
        if dt > 0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / dt)
        display = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        h, w, ch = display.shape
        bytes_per_line = ch * w
        image = QtGui.QImage(display.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(image).scaled(self.video_label.size(), QtCore.Qt.KeepAspectRatio)
        self.video_label.setPixmap(pix)
        self.status.setText(f"Video FPS: {self._fps:.1f}")

    def save_snapshot(self):
        if self.last_frame is None:
            self.status.setText("No frame to save")
            return
        fname = QtWidgets.QFileDialog.getSaveFileName(self, "Save snapshot", "snapshot.png", "PNG files (*.png);;JPEG (*.jpg)")[0]
        if fname:
            cv2.imwrite(fname, self.last_frame)
            self.status.setText(f"Snapshot saved: {fname}")

    # ----------------------------
    # Audio functions
    # ----------------------------
    def start_audio(self):
        try:
            self.audio.start()
            self.timer.start()
            self.status.setText("Audio started")
        except Exception as e:
            self.status.setText(f"Audio start failed: {e}")

    def stop_audio(self):
        self.audio.stop()
        self.timer.stop()
        self.status.setText("Audio stopped")

    def update_filter_labels(self):
        low = self.lowcut_slider.value()
        high = self.highcut_slider.value()
        self.lbl_low.setText(f"Lowcut: {low} Hz")
        self.lbl_high.setText(f"Highcut: {high} Hz")

    def apply_filter(self):
        low = self.lowcut_slider.value()
        high = self.highcut_slider.value()
        if low >= high:
            # invalid: disable filter
            self.audio.set_bandpass(None, None)
            self.status.setText("Invalid bandpass range: filter disabled")
            return
        self.audio.set_bandpass(low, high)
        self.status.setText(f"Bandpass set: {low} - {high} Hz")

    def update_plots(self):
        # waveform
        wave = self.audio.get_waveform()
        if wave is not None and len(wave) > 0:
            x = np.arange(len(wave))
            self.pg_wave_curve.setData(x, wave)
        # fft
        mag = self.audio.get_fft()
        if mag is not None:
            # create freq axis
            freqs = np.fft.rfftfreq(self.audio.blocksize, 1.0 / self.audio.fs)
            # ensure freq length matches mag length
            if len(freqs) > len(mag):
                freqs = freqs[:len(mag)]
            elif len(mag) > len(freqs):
                mag = mag[:len(freqs)]
            self.pg_fft_curve.setData(freqs, mag)

    # ----------------------------
    # Close event (cleanup)
    # ----------------------------
    def closeEvent(self, event):
        try:
            self.stop_audio()
            self.stop_video()
        except Exception:
            pass
        event.accept()

# ----------------------------
# Main entrypoint
# ----------------------------
def main():
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
