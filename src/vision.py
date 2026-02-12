"""
Camera control and YOLO-based vision detection.

Merged from Helper.py + detection_functions.py.
Camera functions → Camera class; prediction/detection as module-level functions.
"""

import os
import statistics
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression


# ======================================================================
# Camera class (from Helper.py camera functions)
# ======================================================================

class Camera:
    """Manage a single camera device for image capture."""

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self.camera_id = camera_id
        self.width = width
        self.height = height
        self._cap: Optional[cv2.VideoCapture] = None

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_id, cv2.CAP_AVFOUNDATION)
        if not self._cap.isOpened():
            print(f"Failed to open camera {self.camera_id}")
            self._cap = None
            return False
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        return True

    def capture(self, save_path: str, warmup_frames: int = 10) -> Optional[str]:
        """Capture a single frame, save to *save_path*, return path or None."""
        if self._cap is None:
            print("Camera not opened")
            return None
        for _ in range(warmup_frames):
            self._cap.read()
        ret, frame = self._cap.read()
        if not ret:
            print("Failed to capture image")
            return None
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        cv2.imwrite(save_path, frame)
        print(f"Image captured: {save_path}")
        return save_path

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            print("Camera released")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *exc):
        self.release()


# ======================================================================
# One-shot capture helpers
# ======================================================================

def capture_image(project_name: str = "default_project",
                  camera_id: int = 0) -> Optional[str]:
    """Open camera, capture one image, release. Returns image path or None."""
    cam = Camera(camera_id=camera_id)
    if not cam.open():
        return None
    save_path = _make_image_path(project_name)
    result = cam.capture(save_path)
    cam.release()
    return result


def capture_image_with_run_id(run_id: str, step_name: str = "capture",
                              base_dir: str = "data",
                              camera_id: int = 0) -> Optional[str]:
    """Capture and save to <base_dir>/experiment/<run_id>/<step_name>_<ts>.jpg."""
    cam = Camera(camera_id=camera_id)
    if not cam.open():
        print("[VISION] Failed to initialize camera.")
        return None
    print("[VISION] Camera initialized successfully.")

    run_folder = os.path.join(base_dir, "experiment", run_id)
    os.makedirs(run_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    image_path = os.path.join(run_folder, f"{step_name}_{timestamp}.jpg")

    result = cam.capture(image_path)
    cam.release()
    return result


# ======================================================================
# Prediction parsing (from Helper.py)
# ======================================================================

def process_predictions(predictions: Any) -> Tuple[
    Dict[str, int],
    Dict[str, List[List[float]]],
    Dict[str, List[List[float]]],
    Dict[str, List[float]],
]:
    """
    Parse a super-gradients style predictions object.
    Returns (class_counts, bounding_boxes, bounding_box_centers, liquid_height_percentages).
    """
    class_counts: Dict[str, int] = {"Tip": 0, "Liquid": 0}
    bounding_boxes: Dict[str, List[List[float]]] = {"Tip": [], "Liquid": []}
    bounding_box_centers: Dict[str, List[List[float]]] = {"Tip": [], "Liquid": []}
    liquid_height_percentages: Dict[str, List[float]] = {"Liquid": []}

    if hasattr(predictions, "_images_prediction_lst"):
        prediction_list = predictions._images_prediction_lst
    else:
        prediction_list = [predictions]

    for image_prediction in prediction_list:
        prediction = image_prediction.prediction
        labels = prediction.labels
        bboxes = prediction.bboxes_xyxy
        class_names = image_prediction.class_names

        for label, bbox in zip(labels, bboxes):
            class_name = class_names[int(label)]
            if class_name not in class_counts:
                class_counts[class_name] = 0
            if class_name not in bounding_boxes:
                bounding_boxes[class_name] = []
            if class_name not in bounding_box_centers:
                bounding_box_centers[class_name] = []

            class_counts[class_name] += 1
            bbox_list = bbox.tolist()
            bounding_boxes[class_name].append(bbox_list)
            x_center = (bbox[0] + bbox[2]) / 2
            y_center = (bbox[1] + bbox[3]) / 2
            bounding_box_centers[class_name].append([float(x_center), float(y_center)])

        # Sort spatially (left to right)
        sorted_tips = sorted(bounding_boxes.get("Tip", []), key=lambda b: b[0])
        sorted_liquids = sorted(bounding_boxes.get("Liquid", []), key=lambda b: b[0])

        for cls_name in bounding_box_centers:
            bounding_box_centers[cls_name].sort(key=lambda c: c[0])

        # Match liquid to tip & calculate percentage
        calculated_levels: List[float] = []
        if not sorted_tips:
            if sorted_liquids:
                print("[MODEL WARNING] Liquid detected but no Tips found.")
        else:
            for liq_box in sorted_liquids:
                liq_x_center = (liq_box[0] + liq_box[2]) / 2
                closest_tip = min(
                    sorted_tips,
                    key=lambda t: abs(((t[0] + t[2]) / 2) - liq_x_center),
                )
                liquid_h = float(liq_box[3] - liq_box[1])
                tip_h = float(closest_tip[3] - closest_tip[1])
                pct = (liquid_h / tip_h) * 100.0 if tip_h > 0 else 0.0
                calculated_levels.append(pct)

        liquid_height_percentages["Liquid"] = calculated_levels
        print(f"[MODEL] Liquid Level Percentages (Sorted): {calculated_levels}")

    return class_counts, bounding_boxes, bounding_box_centers, liquid_height_percentages


def find_missing_tips(
    bounding_box_centers: Dict[str, List[List[float]]],
    expected_tip_count: int = 8,
) -> Tuple[List[int], List[int]]:
    """Infer missing tips based on x-coordinate spacing."""
    tip_centers = bounding_box_centers.get("Tip", [])
    tip_presence = [1] * expected_tip_count
    missing: List[int] = []

    if not tip_centers:
        return [0] * expected_tip_count, list(range(1, expected_tip_count + 1))
    if len(tip_centers) == expected_tip_count:
        return tip_presence, missing

    spacing = (tip_centers[-1][0] - tip_centers[0][0]) / max(expected_tip_count - 1, 1)
    for i in range(expected_tip_count):
        expected_x = tip_centers[0][0] + i * spacing
        found = any(
            abs(det[0] - expected_x) <= spacing / 2 for det in tip_centers
        )
        if not found:
            tip_presence[i] = 0
            missing.append(i + 1)

    return tip_presence, missing


# ======================================================================
# Prediction image saving
# ======================================================================

def save_prediction_image(image_path: str, predictions: Any) -> Optional[str]:
    """Save annotated prediction image next to the original. Returns path or None."""
    image_dir = os.path.dirname(image_path)
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    pred_path = os.path.join(image_dir, f"{base_name}_prediction.jpg")
    try:
        pred_list = (
            predictions._images_prediction_lst
            if hasattr(predictions, "_images_prediction_lst")
            else [predictions]
        )
        if not pred_list:
            return None
        pred_obj = pred_list[0]
        if not hasattr(pred_obj, "draw"):
            return None
        annotated = pred_obj.draw()
        cv2.imwrite(pred_path, annotated)
        print(f"[VISION] Prediction image saved: {pred_path}")
        return pred_path
    except Exception as exc:
        print(f"[VISION WARNING] Failed to save prediction image: {exc}")
        return None


def save_predictions(predictions: Any,
                     output_folder: str = "output_predictions") -> Optional[str]:
    """Save prediction visualisation and rename with timestamp."""
    os.makedirs(output_folder, exist_ok=True)
    try:
        predictions.save(output_folder=output_folder)
    except Exception as exc:
        print(f"Failed to save predictions: {exc}")
        return None
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    original = os.path.join(output_folder, "pred_0.jpg")
    new_path = os.path.join(output_folder, f"pred_{timestamp}.jpg")
    if not os.path.exists(original):
        return None
    os.rename(original, new_path)
    return new_path


# ======================================================================
# Regression calibration
# ======================================================================

def build_regression_function(csv_path: str, degree: int = 3):
    """Build polynomial regression: volume → expected height percentage."""
    df = pd.read_csv(csv_path)
    channel_cols = ["Ch1", "Ch2", "Ch3", "Ch4", "Ch5", "Ch6", "Ch7", "Ch8"]
    df["MeanHeight"] = df[channel_cols].mean(axis=1)

    X = df["Volume"].values.reshape(-1, 1)
    y = df["MeanHeight"].values

    poly = PolynomialFeatures(degree=degree)
    X_poly = poly.fit_transform(X)
    model = LinearRegression()
    model.fit(X_poly, y)

    def predict(volume_ul: float) -> float:
        vol = np.array([[volume_ul]])
        return float(model.predict(poly.transform(vol)))

    return predict


# ======================================================================
# High-level vision checks (from detection_functions.py)
# ======================================================================

def Predict(robot, model: Any, run_id: str, check_type: str,
            imaging_labware_id: str, imaging_well: str,
            imaging_offset: tuple, base_dir: str, step_name: str,
            **kwargs) -> Dict[str, Any]:
    """
    Full vision workflow: Move → Capture → Predict → return result.

    Args:
        robot: OT2Robot instance.
        model: Loaded YOLO model.
        run_id: Current run ID.
        check_type: 'pickup' or 'transfer'.
        imaging_labware_id / imaging_well / imaging_offset: imaging position.
        base_dir: Directory to save images.
        step_name: Name prefix for the image file.
        **kwargs: conf, expected_tips, volume.
    """
    print(f"[VISION] Moving to imaging position: {imaging_well}...")
    robot.move(labware_id=imaging_labware_id, wellname=imaging_well,
               offset=imaging_offset)

    print("[VISION] Capturing image...")
    img_path = capture_image_with_run_id(
        run_id=run_id, step_name=step_name, base_dir=base_dir,
    )

    if check_type == "pickup":
        result = check_tip_with_model(
            model=model, image_path=img_path,
            conf_threshold=kwargs.get("conf", 0.6),
            expected_tips=kwargs.get("expected_tips", 8),
        )
    elif check_type == "transfer":
        result = check_liquid_level(
            model=model, image_path=img_path,
            conf_threshold=kwargs.get("conf", 0.6),
            expected_vol=kwargs.get("volume", 0),
            expected_tips=kwargs.get("expected_tips", 8),
        )
    else:
        raise ValueError(f"Unknown check_type: {check_type}")

    return result


def check_tip_with_model(model: Any, image_path: str,
                         conf_threshold: float = 0.4,
                         expected_tips: int = 8) -> Dict[str, Any]:
    """Check if all expected tips are present."""
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    print(f"[VISION] Tip Check: {image_path}")
    predictions = model.predict(image_path, conf=conf_threshold)
    prediction_image_path = save_prediction_image(image_path, predictions)
    class_counts, _, centers, liq_heights = process_predictions(predictions)
    print(f"[VISION] Detected: {class_counts}")

    try:
        tip_presence, missing_positions = find_missing_tips(centers, expected_tips)
        passed = all(v == 1 for v in tip_presence)
        print(f"[VISION] Presence: {tip_presence}, Missing: {missing_positions}")
    except Exception as exc:
        print(f"[VISION ERROR] Analysis failed: {exc}")
        passed = class_counts.get("Tip", 0) >= expected_tips
        tip_presence = [1] if passed else [0]
        missing_positions = [] if passed else ["unknown"]

    return {
        "passed": passed,
        "tip_presence": tip_presence,
        "missing_positions": missing_positions,
        "class_counts": class_counts,
        "centers": centers,
        "prediction_image_path": prediction_image_path,
        "liquid_height_percentages": liq_heights,
        "predictions": predictions,
    }


def check_liquid_level(model: Any, image_path: str,
                       conf_threshold: float = 0.4,
                       expected_tips: int = 8,
                       expected_vol: float = 0.0,
                       calibration_csv: Optional[str] = None) -> Dict[str, Any]:
    """Check liquid levels against expected volume."""
    if not image_path or not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    tol_percent = float(os.getenv("LLD_CHANNEL_TOLERANCE_PERCENT", "5.0"))

    if calibration_csv is None:
        calibration_csv = os.getenv("LLD_CALIBRATION_CSV", "")
    if not calibration_csv or not os.path.exists(calibration_csv):
        raise FileNotFoundError(f"Calibration data missing: {calibration_csv}")

    predict_height = build_regression_function(calibration_csv, degree=3)
    expected_height = predict_height(expected_vol)

    print(f"[VISION] Liquid Check: {image_path}")
    print(f"[VISION] Target: {expected_vol}uL -> {expected_height:.2f}% (Tol: {tol_percent}%)")

    predictions = model.predict(image_path, conf=conf_threshold)
    prediction_image_path = save_prediction_image(image_path, predictions)
    class_counts, _, centers, liq_heights = process_predictions(predictions)

    tip_count = class_counts.get("Tip", 0)
    detected_levels = liq_heights.get("Liquid", [])
    liquid_count = len(detected_levels)

    channel_pass_status: List[bool] = []
    error_msg: Optional[str] = None
    passed = False

    if tip_count == 0:
        error_msg = "No tips detected."
    else:
        for i, lvl in enumerate(detected_levels):
            diff = abs(lvl - expected_height)
            ok = diff <= tol_percent
            channel_pass_status.append(ok)
            print(f"[VISION] Ch{i}: {lvl:.2f}% (Diff: {diff:.2f}%) -> {'PASS' if ok else 'FAIL'}")

        all_levels_ok = len(channel_pass_status) > 0 and all(channel_pass_status)
        count_mismatch = liquid_count != tip_count
        tips_correct = tip_count == expected_tips

        if not all_levels_ok:
            passed = False
            failed_idx = [i for i, ok in enumerate(channel_pass_status) if not ok]
            error_msg = f"Levels out of range on channels {failed_idx}. Expected {expected_height:.2f}%."
        elif count_mismatch:
            strict_pass = all(abs(lvl - expected_height) <= tol_percent for lvl in detected_levels)
            if tips_correct and strict_pass:
                passed = True
                print(f"[VISION] WARNING: Tip count {tip_count} matches but only {liquid_count} liquids found.")
            else:
                passed = False
                error_msg = (f"Liquid count mismatch ({liquid_count}/{tip_count}) and "
                             f"levels/tips did not meet strict criteria.")
        else:
            passed = True
            print("[VISION] Success: All levels valid and counts match.")

    if error_msg:
        print(f"[VISION ERROR] {error_msg}")

    return {
        "passed": passed,
        "detected_levels": detected_levels,
        "channel_pass_status": channel_pass_status,
        "class_counts": class_counts,
        "centers": centers,
        "prediction_image_path": prediction_image_path,
        "predictions": predictions,
        "expected_vol": expected_vol,
        "expected_height_percent": expected_height,
        "expected_tips": expected_tips,
        "tip_count": tip_count,
        "liquid_count": liquid_count,
        "error_msg": error_msg,
    }


# ======================================================================
# Helpers
# ======================================================================

def _make_image_path(project_name: str) -> str:
    folder = os.path.join(os.path.abspath("."), project_name)
    os.makedirs(folder, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return os.path.join(folder, f"{project_name}_{ts}_hd.jpg")
