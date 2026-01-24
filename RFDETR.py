import os
import time
from typing import Any, Callable, Dict, Iterable

import cv2

try:
    from inference import InferenceHTTPClient, get_model  # type: ignore
except ImportError:
    from inference import get_model  # type: ignore
    try:
        from inference_sdk import InferenceHTTPClient  # type: ignore
    except ImportError:  # pragma: no cover - fallback when SDK missing
        InferenceHTTPClient = None  # type: ignore

API_KEY = os.getenv("ROBOFLOW_API_KEY", "")
API_URL = os.getenv("ROBOFLOW_API_URL", "")
MODEL_ID = os.getenv("ROBOFLOW_MODEL_ID", "kvasir-kqkfx-sd4do/12")
VIDEO_SOURCE: int | str = 0
CONFIDENCE = float(os.getenv("RFDETR_CONFIDENCE", "0.3"))
FRAME_WIDTH = int(os.getenv("RFDETR_FRAME_WIDTH", "640"))
FRAME_HEIGHT = int(os.getenv("RFDETR_FRAME_HEIGHT", "360"))


def ensure_credentials(api_key: str, api_url: str) -> str:
    if api_url:
        if InferenceHTTPClient is None:
            raise SystemExit(
                "InferenceHTTPClient unavailable. Install 'pip install inference-sdk' to enable HTTP mode."
            )
        if not api_key:
            print("[info] Connecting to local inference server without an API key.")
        return api_key

    if not api_key:
        raise SystemExit(
            "Roboflow API key missing. Set ROBOFLOW_API_KEY or assign API_KEY in RFDETR.py."
        )

    return api_key


def build_infer_fn(api_key: str, api_url: str) -> Callable[[Any], Dict[str, Any]]:
    if api_url:
        client = InferenceHTTPClient(api_url=api_url, api_key=api_key or None)
        try:
            client.inference_configuration.confidence = CONFIDENCE
        except AttributeError:
            pass

        def infer_fn(frame: Any) -> Dict[str, Any]:
            return client.infer(frame, model_id=MODEL_ID)

        return infer_fn

    model = get_model(model_id=MODEL_ID, api_key=api_key)

    def infer_fn(frame: Any) -> Dict[str, Any]:
        return model.infer(frame, confidence=CONFIDENCE)

    return infer_fn


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

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, y1 - th - baseline), (x1 + tw, y1), (0, 200, 0), -1)
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


def main() -> None:
    api_key = ensure_credentials(API_KEY, API_URL)
    infer_fn = build_infer_fn(api_key, API_URL)

    endpoint = API_URL or "cloud"
    print(f"Using model '{MODEL_ID}' via {endpoint}.")

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        raise SystemExit(f"Unable to open video source: {VIDEO_SOURCE}")

    if isinstance(VIDEO_SOURCE, int):
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        print(f"Capture size: {FRAME_WIDTH}x{FRAME_HEIGHT}")

    print("Press 'q' in the window to quit.")

    fps = 0.0
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("Failed to read frame. Exiting.")
            break

        start = time.perf_counter()
        result = infer_fn(frame)
        for det in extract_predictions(result):
            draw_box(frame, det)

        end = time.perf_counter()
        dt = end - start
        inst_fps = 1.0 / dt if dt > 0 else 0.0
        fps = 0.9 * fps + 0.1 * inst_fps if fps else inst_fps
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("RF-DETR Demo", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
