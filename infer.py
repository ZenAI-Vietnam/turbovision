import argparse
import os
import time
from typing import Iterator, List, Tuple

import cv2
import numpy as np
import supervision as sv
from pydantic import BaseModel
from tqdm import tqdm
from ultralytics import YOLO

from sports.common.team import TeamClassifier


PLAYER_DETECTION_MODEL_PATH = "football-players-detection/yolo26m.pt_50_1280_new/weights/best.pt"
PITCH_DETECTION_MODEL_PATH = "football-pitch-detection/yolo26x-pose.pt_100_640/weights/best.pt"
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

TEAM_1_CLASS_ID = 6
TEAM_2_CLASS_ID = 7

STRIDE = 60


class BoundingBox(BaseModel):
    x1: int
    y1: int
    x2: int
    y2: int
    cls_id: int
    conf: float


class TVFrameResult(BaseModel):
    frame_id: int
    boxes: list[BoundingBox]
    keypoints: list[Tuple[int, int]]


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def run_infer_results(
    source_video_path: str,
    device: str = "cpu",
) -> List[TVFrameResult]:
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    all_player_crops: List[np.ndarray] = []
    per_frame_data: List[dict] = []

    t_pass1_start = time.perf_counter()
    for frame in tqdm(frame_generator, desc="Pass 1: detect & collect crops"):
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints_sv = sv.KeyPoints.from_ultralytics(pitch_result)

        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        balls = detections[detections.class_id == BALL_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        if len(players) > 0:
            crops = get_crops(frame, players)
            start_idx = len(all_player_crops)
            all_player_crops.extend(crops)
            global_indices = list(range(start_idx, start_idx + len(crops)))
        else:
            global_indices = []

        frame_info: dict = {
            "balls_xyxy": balls.xyxy.copy() if len(balls) > 0 else np.zeros((0, 4), dtype=float),
            "balls_conf": balls.confidence.copy() if len(balls) > 0 else np.zeros((0,), dtype=float),
            "goalkeepers_xyxy": goalkeepers.xyxy.copy() if len(goalkeepers) > 0 else np.zeros((0, 4), dtype=float),
            "goalkeepers_conf": goalkeepers.confidence.copy() if len(goalkeepers) > 0 else np.zeros((0,), dtype=float),
            "players_xyxy": players.xyxy.copy() if len(players) > 0 else np.zeros((0, 4), dtype=float),
            "players_conf": players.confidence.copy() if len(players) > 0 else np.zeros((0,), dtype=float),
            "players_global_idx": np.array(global_indices, dtype=int),
            "referees_xyxy": referees.xyxy.copy() if len(referees) > 0 else np.zeros((0, 4), dtype=float),
            "referees_conf": referees.confidence.copy() if len(referees) > 0 else np.zeros((0,), dtype=float),
            "keypoints": [],
        }

        if keypoints_sv.xy is not None and len(keypoints_sv.xy) > 0:
            pts = keypoints_sv.xy[0]
            confs = keypoints_sv.confidence[0]
            for pt, conf in zip(pts, confs):
                x, y = pt
                if x > 0 and y > 0 and conf > 0.3 and x < frame.shape[1] and y < frame.shape[0]:
                    frame_info["keypoints"].append((int(x), int(y)))

        per_frame_data.append(frame_info)

    t_pass1 = time.perf_counter() - t_pass1_start
    print(f"[infer] pass1_detect_collect: {t_pass1:.3f}s")

    players_team_id_all: np.ndarray | None = None
    if len(all_player_crops) > 0:
        team_classifier = TeamClassifier(device=device, batch_size=128)
        t_fit_start = time.perf_counter()
        players_team_id_all = team_classifier.fit_predict(all_player_crops)
        t_fit = time.perf_counter() - t_fit_start
        print(f"[infer] fit_and_predict_team: {t_fit:.3f}s")

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    frame_id = 0
    t_pass2_start = time.perf_counter()
    results: List[TVFrameResult] = []
    for frame, frame_info in zip(frame_generator, per_frame_data):
        boxes: list[BoundingBox] = []

        # Balls
        balls_xyxy = frame_info["balls_xyxy"]
        balls_conf = frame_info["balls_conf"]
        for i in range(len(balls_xyxy)):
            x1, y1, x2, y2 = balls_xyxy[i]
            boxes.append(
                BoundingBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    cls_id=BALL_CLASS_ID,
                    conf=float(balls_conf[i]),
                )
            )

        # Goalkeepers
        gk_xyxy = frame_info["goalkeepers_xyxy"]
        gk_conf = frame_info["goalkeepers_conf"]
        for i in range(len(gk_xyxy)):
            x1, y1, x2, y2 = gk_xyxy[i]
            boxes.append(
                BoundingBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    cls_id=GOALKEEPER_CLASS_ID,
                    conf=float(gk_conf[i]),
                )
            )

        # Referees
        ref_xyxy = frame_info["referees_xyxy"]
        ref_conf = frame_info["referees_conf"]
        for i in range(len(ref_xyxy)):
            x1, y1, x2, y2 = ref_xyxy[i]
            boxes.append(
                BoundingBox(
                    x1=int(x1),
                    y1=int(y1),
                    x2=int(x2),
                    y2=int(y2),
                    cls_id=REFEREE_CLASS_ID,
                    conf=float(ref_conf[i]),
                )
            )

        pl_xyxy = frame_info["players_xyxy"]
        pl_conf = frame_info["players_conf"]
        pl_idx = frame_info["players_global_idx"]

        if len(pl_xyxy) > 0:
            if players_team_id_all is not None:
                players_team_id = players_team_id_all[pl_idx]
                mapped_class_ids = np.where(
                    players_team_id == 0,
                    TEAM_1_CLASS_ID,
                    TEAM_2_CLASS_ID,
                )
            else:
                mapped_class_ids = np.full(len(pl_xyxy), PLAYER_CLASS_ID, dtype=int)

            for i in range(len(pl_xyxy)):
                x1, y1, x2, y2 = pl_xyxy[i]
                boxes.append(
                    BoundingBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        cls_id=int(mapped_class_ids[i]),
                        conf=float(pl_conf[i]),
                    )
                )

        results.append(
            TVFrameResult(
                frame_id=frame_id,
                boxes=boxes,
                keypoints=frame_info["keypoints"],
            )
        )
        frame_id += 1

    t_pass2 = time.perf_counter() - t_pass2_start
    print(f"[infer] pass2_apply_and_yield: {t_pass2:.3f}s")

    return results


def visualize_from_results(
    source_video_path: str,
    target_video_path: str,
    results: Iterator[TVFrameResult],
) -> None:
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame, frame_result in zip(frame_generator, results):
            annotated_frame = frame.copy()

            for x, y in frame_result.keypoints:
                cv2.circle(annotated_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

            for box in frame_result.boxes:
                color = (0, 0, 255)
                if box.cls_id == TEAM_1_CLASS_ID:
                    color = (255, 0, 0)
                elif box.cls_id == TEAM_2_CLASS_ID:
                    color = (0, 255, 255)

                cv2.rectangle(
                    annotated_frame,
                    (box.x1, box.y1),
                    (box.x2, box.y2),
                    color=color,
                    thickness=2,
                )
                label = f"{box.cls_id}:{box.conf:.2f}"
                cv2.putText(
                    annotated_frame,
                    label,
                    (box.x1, max(0, box.y1 - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color,
                    1,
                    cv2.LINE_AA,
                )

            sink.write_frame(annotated_frame)


def main() -> None:
    parser = argparse.ArgumentParser(description="Soccer inference script (API + CLI)")
    parser.add_argument("--source_video_path", type=str, required=True)
    parser.add_argument(
        "--target_video_path",
        type=str,
        default=None,
        help="Path to save annotated video (visualize).",
    )
    parser.add_argument(
        "--json_output_path",
        type=str,
        default=None,
        help="Path to save detections as JSONL (one TVFrameResult per line).",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Enable visualization (if no target_video_path, will auto-generate).",
    )
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.json_output_path is not None or args.visualize:
        f = open(args.json_output_path, "w", encoding="utf-8") if args.json_output_path is not None else None
        results_for_visualize = run_infer_results(
            source_video_path=args.source_video_path,
            device=args.device,
        )
        for tv_frame in results_for_visualize:
            if f is not None:
                f.write(tv_frame.model_dump_json())
                f.write("\n")
        if f is not None:
            f.close()

    if args.visualize:
        if args.target_video_path is None:
            base, ext = os.path.splitext(args.source_video_path)
            if not ext:
                ext = ".mp4"
            args.target_video_path = f"{base}_vis{ext}"

        visualize_from_results(
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
            results=iter(results_for_visualize),
        )

    if args.json_output_path is None and not args.visualize:
        _ = run_infer_results(
            source_video_path=args.source_video_path,
            device=args.device,
        )


if __name__ == "__main__":
    main()

