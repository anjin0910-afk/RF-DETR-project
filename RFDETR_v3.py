# -*- coding: utf-8 -*-
import csv
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

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

try:
    from openpyxl import Workbook  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    Workbook = None  # type: ignore

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    OpenAI = None  # type: ignore

DEFAULT_CONFIDENCE = float(os.getenv("RFDETR_CONFIDENCE", "0.3"))
DROP_WHEN_QUEUE_FULL = os.getenv("RFDETR_DROP_ON_FULL", "1").lower() in {"1", "true", "yes"}
INPUT_WIDTH = int(os.getenv("RFDETR_INPUT_WIDTH", "960"))
INPUT_HEIGHT = int(os.getenv("RFDETR_INPUT_HEIGHT", "540"))
CLIP_DETECTION_SECONDS = float(os.getenv("RFDETR_CLIP_TRIGGER", "4"))
CLIP_PRE_SECONDS = float(os.getenv("RFDETR_CLIP_PRE", "5"))
CLIP_POST_SECONDS = float(os.getenv("RFDETR_CLIP_POST", "5"))
PATIENT_BASE_DIR = Path(r"C:/Users/nozinu/Desktop/rfdetr/Patient")
try:
    PATIENT_BASE_DIR.mkdir(parents=True, exist_ok=True)
except Exception:
    PATIENT_BASE_DIR = Path.cwd() / "Patient"
    PATIENT_BASE_DIR.mkdir(parents=True, exist_ok=True)

AI_REPORT_ENABLED = os.getenv("RFDETR_ENABLE_REPORTS", "0").lower() in {"1", "true", "yes"}
AI_REPORT_MODEL = os.getenv("RFDETR_REPORT_MODEL", "gpt-5-nano")
AI_REPORT_MAX_PRED = int(os.getenv("RFDETR_REPORT_MAX_PRED", "5"))
AI_REPORT_PROMPT = os.getenv("RFDETR_REPORT_PROMPT", "Please draft a concise clinical impression summarizing endoscopy findings.")


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
    x = det.get("x")
    y = det.get("y")
    w = det.get("width")
    h = det.get("height")
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
    frameReady = QtCore.pyqtSignal(object, float, bool, float, object)
    statusMessage = QtCore.pyqtSignal(str)
    errorOccurred = QtCore.pyqtSignal(str)

    def __init__(
        self,
        video_source: int | str,
        model_id: str,
        api_key: str,
        api_url: str,
        frame_size: tuple[int, int],
        confidence: float,
        parent: QtCore.QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.video_source = video_source
        self.model_id = model_id
        self.api_key = api_key
        self.api_url = api_url
        self.frame_width, self.frame_height = frame_size
        self._confidence = max(0.01, min(0.99, confidence))

        self.stop_event = threading.Event()
        self._client = None
        self._model = None
        self._lock = threading.Lock()

    def stop(self) -> None:
        self.stop_event.set()

    def _build_backend(self) -> None:
        ensure_credentials(self.api_key, self.api_url)
        if self.api_url:
            self._client = InferenceHTTPClient(api_url=self.api_url, api_key=self.api_key or None)
            try:
                self._client.inference_configuration.confidence_threshold = self._confidence
            except AttributeError:
                pass
        else:
            if get_model is None:
                raise RuntimeError(
                    "Local inference backend unavailable. Install the 'inference' package for get_model support."
                )
            self._model = get_model(model_id=self.model_id, api_key=self.api_key)

    def _infer(self, frame: Any) -> Dict[str, Any]:
        with self._lock:
            client = self._client
            model = self._model
            confidence = self._confidence

        if client is not None:
            try:
                client.inference_configuration.confidence_threshold = confidence
            except AttributeError:
                pass
            return client.infer(frame, model_id=self.model_id)

        assert model is not None
        return model.infer(frame, confidence=confidence)

    def run(self) -> None:  # noqa: C901
        try:
            self._build_backend()
        except Exception as exc:  # pragma: no cover
            self.errorOccurred.emit(str(exc))
            return

        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            self.errorOccurred.emit(f"Unable to open video source: {self.video_source}")
            return

        is_live = isinstance(self.video_source, int)
        if is_live:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)

        raw_fps = cap.get(cv2.CAP_PROP_FPS) if not is_live else 0.0
        if not raw_fps or raw_fps != raw_fps:
            raw_fps = 30.0
        target_interval = 1.0 / raw_fps if not is_live and raw_fps > 0 else 0.0

        self.statusMessage.emit("스트리밍을 시작했습니다. 종료하려면 창에서 q 또는 닫기를 누르세요.")

        try:
            while not self.stop_event.is_set():
                loop_start = time.perf_counter()
                ok, frame = cap.read()
                if not ok or frame is None:
                    self.statusMessage.emit("영상 소스에서 프레임을 읽을 수 없습니다. 스트리밍을 종료합니다.")
                    break

                if INPUT_WIDTH > 0 and INPUT_HEIGHT > 0:
                    frame = cv2.resize(frame, (INPUT_WIDTH, INPUT_HEIGHT), interpolation=cv2.INTER_AREA)

                start = time.perf_counter()
                detected = False
                try:
                    result = self._infer(frame)
                    raw_predictions = list(extract_predictions(result))
                    detected = len(raw_predictions) > 0
                    predictions = []
                    for det in raw_predictions:
                        cls_name = det.get("class") or det.get("class_name") or str(det.get("class_id", ""))
                        conf_val = det.get("confidence") or det.get("score")
                        try:
                            conf_float = float(conf_val) if conf_val is not None else None
                        except (TypeError, ValueError):
                            conf_float = None
                        predictions.append(
                            {
                                "class": cls_name,
                                "confidence": conf_float,
                            }
                        )
                        draw_box(frame, det)
                    predictions = predictions[:10]
                except Exception as exc:  # pragma: no cover
                    self.statusMessage.emit(f"[warn] 추론 중 오류 발생: {exc}")
                processing_time = time.perf_counter() - start
                fps = 1.0 / processing_time if processing_time > 0 else 0.0
                frame_ts = time.time()
                self.frameReady.emit(frame.copy(), fps, detected, frame_ts, predictions)

                if target_interval > 0:
                    elapsed = time.perf_counter() - loop_start
                    remaining = target_interval - elapsed
                    if remaining > 0:
                        time.sleep(remaining)
        finally:
            cap.release()
            self.statusMessage.emit("스트리밍이 종료되었습니다.")

class PatientInfoDialog(QtWidgets.QDialog):
    def __init__(self, info: Dict[str, str], parent: QtWidgets.QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("환자 정보 입력")
        layout = QtWidgets.QFormLayout(self)
        self.setStyleSheet("QLabel { color: #000000; } QLineEdit { color: #000000; }")

        self.name_edit = QtWidgets.QLineEdit(info.get("name", ""))
        self.age_edit = QtWidgets.QLineEdit(info.get("age", ""))
        self.sex_edit = QtWidgets.QLineEdit(info.get("sex", ""))
        self.dob_edit = QtWidgets.QLineEdit(info.get("dob", ""))

        layout.addRow("이름", self.name_edit)
        layout.addRow("나이", self.age_edit)
        layout.addRow("성별", self.sex_edit)
        layout.addRow("생년월일", self.dob_edit)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def get_data(self) -> Dict[str, str]:
        return {
            "name": self.name_edit.text().strip(),
            "age": self.age_edit.text().strip(),
            "sex": self.sex_edit.text().strip(),
            "dob": self.dob_edit.text().strip(),
        }


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Konyang Univ - Team OmniDow REDETR")
        self.setMinimumSize(960, 640)
        self.setStyleSheet(
            """
            QMainWindow { background-color: #0b1623; }
            QLabel#VideoLabel { background-color: #05080f; border: 2px solid #1b2a41; }
            QPushButton { background-color: #1b2a41; color: #e0f2ff; padding: 8px 16px; border-radius: 6px; }
            QPushButton:hover { background-color: #254063; }
            QLabel { color: #bcd4e6; }
            QLineEdit { color: #000000; }
            """
        )

        self.worker: DetectionWorker | None = None
        self.last_frame: Any | None = None
        self.last_fps: float = 0.0

        self.is_recording = False
        self.record_path: Path | None = None
        self.video_writer: cv2.VideoWriter | None = None
        self.pending_writer_init = False

        self.clip_buffer = deque()
        self.clip_detection_start: float | None = None
        self.clip_pending_save = False
        self.clip_save_deadline = 0.0
        self.clip_window_start: float | None = None
        self.patient_info: Dict[str, str] = {"name": "", "age": "", "sex": "", "dob": ""}
        self.last_raw_frame: Any | None = None
        self.last_frame_ts: float = 0.0
        self.last_predictions: list[Dict[str, Any]] = []
        self.pending_report_context: Dict[str, Any] | None = None
        self.patient_folder: Path | None = None

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        root_layout = QtWidgets.QVBoxLayout(central)

        control_layout = QtWidgets.QHBoxLayout()
        button_layout = QtWidgets.QHBoxLayout()
        self.webcam_button = QtWidgets.QPushButton("웹캠")
        self.video_button = QtWidgets.QPushButton("동영상 파일")
        button_layout.addWidget(self.webcam_button)
        button_layout.addWidget(self.video_button)
        self.info_button = QtWidgets.QPushButton("환자 정보")
        button_layout.addWidget(self.info_button)
        control_layout.addLayout(button_layout)
        control_layout.addStretch()

        self.record_button = QtWidgets.QPushButton("녹화 시작")
        control_layout.addWidget(self.record_button)

        self.fps_label = QtWidgets.QLabel("FPS: --")
        control_layout.addWidget(self.fps_label)

        root_layout.addLayout(control_layout)

        self.video_label = QtWidgets.QLabel()
        self.video_label.setObjectName("VideoLabel")
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        root_layout.addWidget(self.video_label, stretch=1)

        self.status_label = QtWidgets.QLabel("준비 완료")
        root_layout.addWidget(self.status_label)

        self.webcam_button.clicked.connect(self.start_webcam)
        self.video_button.clicked.connect(self.start_video_file)
        self.record_button.clicked.connect(self.toggle_recording)
        self.info_button.clicked.connect(self.open_patient_info_dialog)

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

        self.worker = DetectionWorker(
            video_source=source,
            model_id=model_id,
            api_key=api_key,
            api_url=api_url,
            frame_size=frame_size,
            confidence=DEFAULT_CONFIDENCE,
        )
        self.worker.frameReady.connect(self.on_frame_ready)
        self.worker.statusMessage.connect(self.status_label.setText)
        self.worker.errorOccurred.connect(self.on_worker_error)
        self.worker.start()

        source_desc = "웹캠" if source == 0 else str(source)
        self.status_label.setText(f"{source_desc} 실행 중…")
        self.fps_label.setText("FPS: --")

    def stop_worker(self) -> None:
        if self.worker is not None:
            self.worker.stop()
            self.worker.wait(1000)
            self.worker = None
        self.stop_recording()

    def toggle_recording(self) -> None:
        if self.worker is None:
            self.status_label.setText("먼저 영상을 실행한 후 녹화할 수 있습니다.")
            return

        if self.is_recording:
            self.stop_recording()
            self.status_label.setText("녹화를 종료했습니다.")
        else:
            if self.patient_info.get("name") and self.patient_info.get("age"):
                self._ensure_patient_folder()
            target_dir = self.patient_folder or PATIENT_BASE_DIR
            timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
            self.record_path = target_dir / f"record_{timestamp}.mp4"
            try:
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.status_label.setText(f"폴더 생성 실패: {exc}")
                self.record_path = PATIENT_BASE_DIR / f"record_{timestamp}.mp4"
                self.record_path.parent.mkdir(parents=True, exist_ok=True)
            self.is_recording = True
            self.pending_writer_init = True
            self.record_button.setText("��ȭ ����")
            self.status_label.setText(f"��ȭ �غ�: {self.record_path.name}")

    def stop_recording(self) -> None:
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None
        self.is_recording = False
        self.pending_writer_init = False
        self.record_path = None
        self.record_button.setText("녹화 시작")

    @QtCore.pyqtSlot(object, float, bool, float, object)
    def on_frame_ready(self, frame: Any, fps: float, detected: bool, frame_ts: float, predictions: object) -> None:
        self.last_raw_frame = frame.copy()
        self.last_fps = fps
        self.last_frame_ts = frame_ts

        sanitized_preds: list[Dict[str, Any]] = []
        if isinstance(predictions, (list, tuple)):
            for item in predictions:
                if isinstance(item, dict):
                    sanitized_preds.append({
                        "class": str(item.get("class", "")),
                        "confidence": item.get("confidence"),
                    })
        self.last_predictions = sanitized_preds

        self.clip_buffer.append((frame_ts, frame.copy()))
        while self.clip_buffer and frame_ts - self.clip_buffer[0][0] > (CLIP_PRE_SECONDS + CLIP_POST_SECONDS + 2):
            self.clip_buffer.popleft()

        if detected:
            if self.clip_detection_start is None:
                self.clip_detection_start = frame_ts
            elif (not self.clip_pending_save) and (frame_ts - self.clip_detection_start >= CLIP_DETECTION_SECONDS):
                self.clip_pending_save = True
                self.clip_window_start = self.clip_detection_start
                self.clip_save_deadline = frame_ts + CLIP_POST_SECONDS
                if self.patient_info.get("name") and self.patient_info.get("age"):
                    self._ensure_patient_folder()
                self.pending_report_context = {
                    "trigger_time": frame_ts,
                    "clip_start": self.clip_window_start,
                    "predictions": [dict(p) for p in self.last_predictions[:AI_REPORT_MAX_PRED]],
                }
                self.status_label.setText("���� �߰�")
        else:
            if not self.clip_pending_save:
                self.clip_detection_start = None

        if self.clip_pending_save and frame_ts >= self.clip_save_deadline:
            clip_path = self._save_clip()
            if clip_path is not None:
                self._generate_ai_report(clip_path)

        display_frame = self.last_raw_frame.copy()
        display_frame = self._apply_patient_overlay(display_frame, frame_ts)
        self.last_frame = display_frame.copy()
        if self.is_recording:
            self._handle_recording(display_frame, fps)

        self._update_video_display(display_frame, fps)


    def _update_video_display(self, frame: Any, fps: float) -> None:
        if frame is None:
            return
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        qimg = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        scaled = pixmap.scaled(
            self.video_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(scaled)
        self.fps_label.setText(f"FPS: {fps:.1f}" if fps > 0 else "FPS: --")

    def _apply_patient_overlay(self, frame: Any, frame_ts: float | None = None) -> Any:
        info = getattr(self, "patient_info", {})
        if not info or not any(info.values()):
            return frame

        if frame_ts is not None:
            time_text = datetime.fromtimestamp(frame_ts).strftime('%H:%M:%S')
        else:
            time_text = datetime.now().strftime('%H:%M:%S')

        overlay_lines = [
            f"Name : {info.get('name', '')}",
            f"Age : {info.get('age', '')}",
            f"Sex : {info.get('sex', '')}",
            f"DOB : {info.get('dob', '')}",
            f"Time : {time_text}",
        ]
        x = 20
        y_start = 30
        for idx, line in enumerate(overlay_lines):
            y = y_start + idx * 28
            cv2.putText(
                frame,
                line,
                (x, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
        return frame

    def _handle_recording(self, frame: Any, fps: float) -> None:
        if not self.is_recording or self.record_path is None:
            return

        if self.video_writer is None and self.pending_writer_init:
            height, width = frame.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            fps_value = fps if fps > 1.0 else 30.0
            writer = cv2.VideoWriter(str(self.record_path), fourcc, fps_value, (width, height))
            if not writer.isOpened():
                self.status_label.setText("녹화 장치를 열 수 없습니다.")
                self.stop_recording()
                return
            self.video_writer = writer
            self.pending_writer_init = False
            self.status_label.setText(f"녹화 중… {self.record_path.name}")

        if self.video_writer is not None:
            self.video_writer.write(frame)

    def open_patient_info_dialog(self) -> None:
        dialog = PatientInfoDialog(self.patient_info, self)
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.patient_info = dialog.get_data()
            self._ensure_patient_folder()
            self._save_patient_info_excel()
            if self.last_raw_frame is not None:
                display_frame = self._apply_patient_overlay(
                    self.last_raw_frame.copy(), self.last_frame_ts or None
                )
                self.last_frame = display_frame.copy()
                self._update_video_display(display_frame, self.last_fps)

    def _save_clip(self) -> Optional[Path]:
        saved_path: Optional[Path] = None
        if self.clip_window_start is None:
            self.clip_pending_save = False
            self.clip_save_deadline = 0.0
            return None

        window_start = self.clip_window_start - CLIP_PRE_SECONDS
        window_end = self.clip_save_deadline
        clip_frames = [(ts, fr.copy()) for ts, fr in self.clip_buffer if window_start <= ts <= window_end]
        if len(clip_frames) < 2:
            self.status_label.setText("영상클립 저장 실패")
        else:
            duration = clip_frames[-1][0] - clip_frames[0][0]
            if duration <= 0:
                fps_value = self.last_fps if self.last_fps > 1.0 else 30.0
            else:
                fps_value = max(10.0, min(60.0, len(clip_frames) / duration))
            height, width = clip_frames[0][1].shape[:2]
            clip_name = datetime.now().strftime("%y_%m_%d_%H%M_clip.mp4")
            target_dir = self.patient_folder or PATIENT_BASE_DIR
            try:
                target_dir.mkdir(parents=True, exist_ok=True)
            except Exception as exc:
                self.status_label.setText(f"폴더 생성 실패: {exc}")
                target_dir = PATIENT_BASE_DIR
                target_dir.mkdir(parents=True, exist_ok=True)
            clip_path = target_dir / clip_name
            writer = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps_value, (width, height))
            if not writer.isOpened():
                self.status_label.setText("영상클립 저장 실패")
            else:
                for ts, fr in clip_frames:
                    writer.write(self._apply_patient_overlay(fr, ts))
                writer.release()
                self.status_label.setText("영상클립이 저장되었습니다.")
                saved_path = clip_path

        self.clip_pending_save = False
        self.clip_detection_start = None
        self.clip_window_start = None
        self.clip_save_deadline = 0.0
        return saved_path


    def _generate_ai_report(self, clip_path: Path) -> None:
        context = self.pending_report_context or {}
        self.pending_report_context = None
        if not AI_REPORT_ENABLED:
            return
        if OpenAI is None:
            self.status_label.setText("openai 패키지를 설치하면 보고서를 생성할 수 있습니다.")
            return
        api_key = os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            self.status_label.setText("OPENAI_API_KEY 환경 변수를 설정하세요.")
            return

        try:
            client = OpenAI(api_key=api_key)
        except Exception as exc:
            self.status_label.setText(f"OpenAI 클라이언트 초기화 실패: {exc}")
            return

        patient_info = {key: (val or "").strip() for key, val in self.patient_info.items()}
        predictions = context.get("predictions") or self.last_predictions
        trigger_time = context.get("trigger_time")
        clip_start = context.get("clip_start")

        pred_lines: list[str] = []
        for pred in predictions[:AI_REPORT_MAX_PRED]:
            cls_name = str(pred.get("class", ""))
            conf = pred.get("confidence")
            if isinstance(conf, (float, int)):
                pred_lines.append(f"- {cls_name} ({conf * 100:.1f}% 확신)")
            else:
                pred_lines.append(f"- {cls_name}")
        if not pred_lines:
            pred_lines.append("- 탐지된 병변 정보가 충분하지 않습니다.")

        trigger_text = datetime.fromtimestamp(trigger_time).strftime("%Y-%m-%d %H:%M:%S") if isinstance(trigger_time, (float, int)) else "알 수 없음"
        clip_start_text = datetime.fromtimestamp(clip_start).strftime("%Y-%m-%d %H:%M:%S") if isinstance(clip_start, (float, int)) else "알 수 없음"

        pred_section = "\n".join(pred_lines)
        prompt = f"""환자 정보:\n- 이름: {patient_info.get('name', '')}\n- 나이: {patient_info.get('age', '')}\n- 성별: {patient_info.get('sex', '')}\n- 생년월일: {patient_info.get('dob', '')}\n\n탐지 요약:\n발견 시각: {trigger_text}\n분석 시작 시각: {clip_start_text}\n탐지 결과:\n{pred_section}\n\n클립 위치: {clip_path}\n\n{AI_REPORT_PROMPT}\n"""

        try:
            response = client.chat.completions.create(
                model=AI_REPORT_MODEL,
                messages=[
                    {"role": "system", "content": "You are a medical documentation assistant for endoscopy findings."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
            )
            report_text = response.choices[0].message.content.strip() if response.choices else ""
        except Exception as exc:
            self.status_label.setText(f"보고서 생성 실패: {exc}")
            return

        if not report_text:
            self.status_label.setText("보고서 생성 응답이 비어 있습니다.")
            return

        target_dir = self.patient_folder or PATIENT_BASE_DIR
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.status_label.setText(f"보고서 저장 폴더 생성 실패: {exc}")
            return

        report_name = datetime.now().strftime("%y_%m_%d_%H%M_report.txt")
        report_path = target_dir / report_name
        try:
            report_path.write_text(report_text, encoding="utf-8")
        except Exception as exc:
            self.status_label.setText(f"보고서 저장 실패: {exc}")
            return

        self.status_label.setText(f"AI 보고서가 저장되었습니다: {report_path.name}")

    def _ensure_patient_folder(self) -> None:
        name = (self.patient_info.get("name") or "").strip()
        age = (self.patient_info.get("age") or "").strip()
        if not name or not age:
            self.patient_folder = None
            return

        dob = (self.patient_info.get("dob") or "").strip()
        folder_name = "_".join(filter(None, [
            self._sanitize_for_path(name),
            self._sanitize_for_path(age),
            self._sanitize_for_path(dob),
        ]))
        if not folder_name.strip('_'):
            self.status_label.setText("유효한 환자 정보를 입력하세요.")
            self.patient_folder = None
            return

        target_dir = PATIENT_BASE_DIR / folder_name
        try:
            target_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.status_label.setText(f"폴더 생성 실패: {exc}")
            self.patient_folder = None
        else:
            self.patient_folder = target_dir

    def _sanitize_for_path(self, value: str) -> str:
        cleaned = re.sub(r"[^\w\-]+", "_", value.strip())
        return cleaned.strip('_') or "unknown"

    def _save_patient_info_excel(self) -> None:
        if self.patient_folder is None:
            return
        if Workbook is None:
            csv_path = self.patient_folder / "patient_info.csv"
            try:
                with csv_path.open('w', newline='', encoding='utf-8-sig') as fp:
                    writer = csv.writer(fp)
                    writer.writerow(["Name", "Age", "Sex", "DOB", "Saved At"])
                    writer.writerow([
                        self.patient_info.get("name", ""),
                        self.patient_info.get("age", ""),
                        self.patient_info.get("sex", ""),
                        self.patient_info.get("dob", ""),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    ])
            except Exception as exc:
                self.status_label.setText(f"환자 정보 저장 실패: {exc}")
            else:
                self.status_label.setText("openpyxl 미설치로 CSV로 저장했습니다.")
            return

        wb = Workbook()
        ws = wb.active
        ws.title = "PatientInfo"
        ws.append(["Name", "Age", "Sex", "DOB", "Saved At"])
        ws.append([
            self.patient_info.get("name", ""),
            self.patient_info.get("age", ""),
            self.patient_info.get("sex", ""),
            self.patient_info.get("dob", ""),
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        ])
        try:
            wb.save(self.patient_folder / "patient_info.xlsx")
        except Exception as exc:
            self.status_label.setText(f"환자 정보 저장 실패: {exc}")
        else:
            self.status_label.setText("환자 정보가 저장되었습니다.")

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
        save_dir = self.patient_folder or PATIENT_BASE_DIR
        try:
            save_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            self.status_label.setText(f"폴더 생성 실패: {exc}")
            save_dir = PATIENT_BASE_DIR
            save_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%y_%m_%d_%H%M%S")
        file_path = save_dir / f"screenshot_{timestamp}.jpg"
        if cv2.imwrite(str(file_path), self.last_frame):
            self.status_label.setText(f"스크린샷 저장: {file_path.name}")
        else:
            self.status_label.setText("스크린샷 저장 실패")

    def resizeEvent(self, event: QtGui.QResizeEvent) -> None:
        super().resizeEvent(event)
        if self.last_frame is not None:
            self._update_video_display(self.last_frame, self.last_fps)

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
