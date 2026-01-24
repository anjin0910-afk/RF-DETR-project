# -*- coding: utf-8 -*-
import os
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets

try:
    from inference_sdk import InferenceHTTPClient  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    InferenceHTTPClient = None  # type: ignore

try:
    from inference import get_model  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    get_model = None  # type: ignore


DEFAULT_CONFIDENCE = float(os.getenv("RFDETR_CONFIDENCE", "0.3"))
DEFAULT_WORKERS = max(1, int(os.getenv("RFDETR_WORKERS", "2")))
DEFAULT_QUEUE_SIZE = max(1, int(os.getenv("RFDETR_QUEUE_SIZE", str(DEFAULT_WORKERS * 2))))
DROP_WHEN_QUEUE_FULL = os.getenv("RFDETR_DROP_ON_FULL", "1").lower() in {"1", "true", "yes"}


def ensure_credentials(api_key: str, api_url: str) -> str:
    if api_url:
        if InferenceHTTPClient is None:
            raise SystemExit(
                "InferenceHTTPClient unavailable. Install it with 'pip install inference-sdk'."
            )
        if not api_key:
            print("[info] Connecting to local inference server without an API key.")
        return api_key

    if not api_key:
        raise SystemExit(
            "Roboflow API key missing. Set ROBOFLOW_API_KEY or edit the script to hardcode it."
        )

    return api_key


def extract_predictions(result) -> Iterable[Dict[str, Any]]:
    if isinstance(result, dict):
        predictions = result.get("predictions", [])
    elif isinstance(result, list) and result:
        first = result[0]
        if hasattr(first, "predictions"):
            predictions = first.predictions
        elif isinstance(first, dict):
            predictions = first.get("predictions", [])
        else:
            predictions = []
    else:
        predictions = []

    for pred in predictions:
        data = pred.__dict__ if hasattr(pred, "__dict__") else pred
        if isinstance(data, dict):
            yield data


def draw_box(frame, det: Dict[str, Any]) -> None:
    x, y = det.get("x"), det.get("y")
    w, h = det.get("width"), det.get("height")
    if None in (x, y, w, h):
        return

    x1 = int(x - w / 2)
    y1 = int(y - h / 2)
    x2 = int(x + w / 2)
    y2 = int(y + h / 2)

    cls_name = det.get("class") or det.get("class_name") or str(det.get("class_id", ""))
    conf = det.get("confidence") or det.get("score")
    label = f"{cls_name} {conf:.2f}" if conf is not None else str(cls_name)

    cv2.rectangle(frame, (x1, y1), (x2, y2), (32, 122, 255), 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), (32, 122, 255), -1)
    cv2.putText(
        frame,
        label,
        (x1, y1 - baseline),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 0, 0),
        2,
        cv2.LINE_AA,
    )


class DetectionWorker(QtCore.QThread):
    frameReady = QtCore.pyqtSignal(object, float)
    statusMessage = QtCore.pyqtSignal(str)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        video_source: int | str,
        model_id: str,
        api_key: str,
        api_url: str,
        frame_size: Tuple[int, int],
        workers: int,
        queue_size: int,
        initial_confidence: float,
        drop_when_full: bool,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_source = video_source
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url
        self.frame_width, self.frame_height = frame_size
        self.worker_count = max(1, workers)
        self.queue_size = max(1, queue_size)
        self.drop_when_full = drop_when_full
        self._confidence = initial_confidence

        self.stop_event = threading.Event()
        self.frame_queue: "queue.Queue[Tuple[int, Any]]" = queue.Queue(maxsize=self.queue_size)
        self.result_queue: "queue.Queue[Tuple[int, Any, list[Dict[str, Any]], float]]" = queue.Queue()
        self.pending_results: Dict[int, Tuple[Any, list[Dict[str, Any]], float]] = {}
        self.next_display_id = 0
        self.fps_ema = 0.0
        self._client = None
        self._model = None
        self._infer_lock = threading.Lock()

    def set_confidence(self, value: float) -> None:
        value = max(0.01, min(0.99, value))
        with self._infer_lock:
            self._confidence = value
            if self._client is not None:
                try:
                    self._client.inference_configuration.confidence = value
                except AttributeError:
                    pass

    def stop(self) -> None:
        self.stop_event.set()

    def _build_backends(self) -> None:
        ensure_credentials(self.api_key, self.api_url)
        if self.api_url:
            self._client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key or None)
            try:
                self._client.inference_configuration.confidence = self._confidence
                self._client.inference_configuration.max_concurrent_requests = self.worker_count
            except AttributeError:
                pass
        else:
            if get_model is None:
                raise RuntimeError("Local inference backend unavailable. Install the 'inference' package for get_model support.")
            self._model = get_model(model_id=self.model_id, api_key=self.api_key)

    def _infer(self, frame: Any) -> Dict[str, Any]:
        with self._infer_lock:
            confidence = self._confidence
            client = self._client
            model = self._model

        if client is not None:
            try:
                client.inference_configuration.confidence = confidence
            except AttributeError:
                pass
            return client.infer(frame, model_id=self.model_id)

        assert model is not None
        return model.infer(frame, confidence=confidence)

    def run(self) -> None:  # noqa: C901
        try:
            self._build_backends()
        except Exception as exc:  # pragma: no cover
            self.errorOccurred.emit(str(exc))
            return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.errorOccurred.emit(f"Unable to open video source: {self.video_source}")
            return

        if isinstance(self.video_source, int):
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        self.statusMessage.emit("Streaming started. Press 'q' in window to stop.")

        workers: list[threading.Thread] = []
        stop_event = self.stop_event
        frame_queue = self.frame_queue
        result_queue = self.result_queue

        def worker_loop(worker_id: int) -> None:
            while not stop_event.is_set():
                try:
                    item = frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue

                if item is None:
                    frame_queue.task_done()
                    break

                frame_id, frame = item
                start = time.perf_counter()
                try:
                    result = self._infer(frame)
                    predictions = list(extract_predictions(result))
                except Exception as exc:  # pragma: no cover
                    self.statusMessage.emit(f"[warn] worker {worker_id} inference failed: {exc}")
                    predictions = []
                elapsed = time.perf_counter() - start
                result_queue.put((frame_id, frame, predictions, elapsed))
                frame_queue.task_done()

        for idx in range(self.worker_count):
            thread = threading.Thread(target=worker_loop, args=(idx,), daemon=True)
            workers.append(thread)
            thread.start()

        frame_id = 0
        try:
            while not stop_event.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.statusMessage.emit("No more frames from source. Finishing.")
                    break

                try:
                    frame_queue.put((frame_id, frame.copy()), timeout=0.01 if self.drop_when_full else None)
                    frame_id += 1
                except queue.Full:
                    if self.drop_when_full:
                        continue
                    frame_queue.put((frame_id, frame.copy()))
                    frame_id += 1

                self._pump_results(block=False)

            stop_event.set()
            self._signal_workers(workers)

            while frame_queue.unfinished_tasks or not result_queue.empty() or self.pending_results:
                if not self._pump_results(block=True):
                    continue
        finally:
            stop_event.set()
            self._signal_workers(workers)
            for thread in workers:
                thread.join(timeout=0.5)
            cap.release()
            self.statusMessage.emit("Streaming stopped.")

    def _signal_workers(self, workers: list[threading.Thread]) -> None:
        for _ in workers:
            placed = False
            while not placed and not self.stop_event.is_set():
                try:
                    self.frame_queue.put_nowait(None)
                    placed = True
                except queue.Full:
                    time.sleep(0.01)

    def _pump_results(self, block: bool) -> bool:
        displayed = False
        while True:
            try:
                item = self.result_queue.get(block, timeout=0.01 if block else 0)
            except queue.Empty:
                break

            frame_id, processed_frame, preds, inf_time = item
            self.pending_results[frame_id] = (processed_frame, preds, inf_time)
            self.result_queue.task_done()

        while self.next_display_id in self.pending_results:
            frame_to_show, preds, inf_time = self.pending_results.pop(self.next_display_id)
            self.next_display_id += 1

            for det in preds:
                draw_box(frame_to_show, det)

            inst_fps = 1.0 / inf_time if inf_time > 0 else 0.0
            self.fps_ema = 0.9 * self.fps_ema + 0.1 * inst_fps if self.fps_ema else inst_fps
            self.frameReady.emit(frame_to_show, self.fps_ema)
            displayed = True

        return displayed


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RF-DETR Demo")
        self.setMinimumSize(960, 640)
        self.setStyleSheet(
            "QMainWindow { background-color: #0b1623; }\n"
            "QLabel#VideoLabel { background-color: #05080f; border: 2px solid #1b2a41; }\n"
            "QPushButton { background-color: #1b2a41; color: #e0f2ff; padding: 8px 16px; border-radius: 6px; }\n"
            "QPushButton:hover { background-color: #254063; }\n"
            "QSlider::groove:horizontal { background: #1f2d3f; height: 6px; border-radius: 3px; }\n"
            "QSlider::handle:horizontal { background: #4aa3ff; width: 18px; margin: -6px 0; border-radius: 9px; }\n"
            "QLabel { color: #bcd4e6; }"
        )

        self.worker: DetectionWorker | None = None
        self.last_frame: Any | None = None
        self.last_fps: float = 0.0

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        control_layout = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        self.webcam_button = QtWidgets.QPushButton("웹캠")
        self.video_button = QtWidgets.QPushButton("동영상 파일")
        button_layout.addWidget(self.webcam_button)
        button_layout.addWidget(self.video_button)
        control_layout.addLayout(button_layout)
        control_layout.addStretch()

        slider_container = QtWidgets.QVBoxLayout()
        self.conf_label = QtWidgets.QLabel(f"신뢰도: {DEFAULT_CONFIDENCE:.2f}")
        slider_container.addWidget(self.conf_label, alignment=QtCore.Qt.AlignRight)
        self.conf_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.conf_slider.setMinimum(5)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(int(DEFAULT_CONFIDENCE * 100))
        slider_container.addWidget(self.conf_slider)
        control_layout.addLayout(slider_container)
        root_layout.addLayout(control_layout)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        root_layout.addWidget(self.video_label, stretch=1)

        self.status_label = QtWidgets.QLabel("준비 완료")
        root_layout.addWidget(self.status_label)

        self.webcam_button.clicked.connect(self.start_webcam)
        self.video_button.clicked.connect(self.start_video_file)
        self.conf_slider.valueChanged.connect(self.on_confidence_changed)

    def start_webcam(self) -> None:
        self.start_worker(0)

    def start_video_file(self) -> None:
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            "동영상 파일 선택",
            str(Path.cwd()),
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)",
        )
        if file_path:
            self.start_worker(file_path)

    def start_worker(self, source: int | str) -> None:
        self.stop_worker()

        api_key = os.getenv("ROBOFLOW_API_KEY", "")
        api_url = os.getenv("ROBOFLOW_API_URL", "")
        model_id = os.getenv("ROBOFLOW_MODEL_ID", "kvasir-kqkfx-sd4do/12")

        frame_size = (
            int(os.getenv("RFDETR_FRAME_WIDTH", "640")),
            int(os.getenv("RFDETR_FRAME_HEIGHT", "360")),
        )

        workers = max(1, int(os.getenv("RFDETR_WORKERS", str(DEFAULT_WORKERS))))
        queue_size = max(1, int(os.getenv("RFDETR_QUEUE_SIZE", str(DEFAULT_QUEUE_SIZE))))

        self.worker = DetectionWorker(
            video_source=source,
            model_id=model_id,
            api_key=api_key,
            api_url=api_url,
            frame_size=frame_size,
            workers=workers,
            queue_size=queue_size,
            initial_confidence=self.conf_slider.value() / 100.0,
            drop_when_full=DROP_WHEN_QUEUE_FULL,
        )

        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.statusMessage.connect(self.status_label.setText)
        self.worker.errorOccurred.connect(self.on_worker_error)

        self.worker.start()
        source_desc = "웹캠" if source == 0 else str(source)
        self.status_label.setText(f"{source_desc} 실행 중…")

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None

    def on_confidence_changed(self, value: int) -> None:
        conf = value / 100.0
        self.conf_label.setText(f"신뢰도: {conf:.2f}")
        if self.worker is not None:
            self.worker.set_confidence(conf)

    @QtCore.pyqtSlot(object, float)
    def on_frame_ready(self, frame: Any, fps: float) -> None:
        self.last_frame = frame.copy()
        self.last_fps = fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        painter = QtGui.QPainter(scaled)
        painter.setPen(QtGui.QPen(QtGui.QColor("white")))
        painter.setFont(QtGui.QFont("Arial", 10))
        painter.drawText(12, scaled.height() - 12, f"FPS: {fps:.1f}")
        painter.end()
        self.video_label.setPixmap(scaled)

    def on_worker_error(self, message: str) -> None:
        QtWidgets.QMessageBox.critical(self, "오류", message)
        self.status_label.setText(message)
        self.stop_worker()

    def keyPressEvent(self, event: QtGui.QKeyEvent) -> None:
        if event.key() == QtCore.Qt.Key_S:
            self.save_screenshot()
            event.accept()
        else:
            super().keyPressEvent(event)

    def save_screenshot(self) -> None:
        if self.last_frame is None:
            self.status_label.setText("저장할 프레임이 없습니다.")
            return

        save_dir = Path.cwd()
        timestamp = datetime.now().strftime("%y_%m_%d_%H%M")
        file_path = save_dir / f"{timestamp}.jpg"
        if cv2.imwrite(str(file_path), self.last_frame):
            self.status_label.setText(f"스크린샷 저장: {file_path.name}")
        else:
            self.status_label.setText("스크린샷 저장 실패")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.last_frame is not None:
            self.on_frame_ready(self.last_frame, self.last_fps)

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        self.stop_worker()
        super().closeEvent(event)


def main() -> None:
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    window.status_label.setText("원하는 입력을 선택하세요.")
    app.exec_()


if __name__ == "__main__":
    main()

