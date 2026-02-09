import argparse
import os
from typing import Iterator, List, Tuple

import cv2
import numpy as np
import supervision as sv
from pydantic import BaseModel
from tqdm import tqdm
from ultralytics import YOLO

from sports.common.team import TeamClassifier
from sports.configs.soccer import SoccerPitchConfiguration


PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/football-player-detection.pt")
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, "data/football-pitch-detection.pt")

# Original class ids from the detector
BALL_CLASS_ID = 0          # Person.BALL
GOALKEEPER_CLASS_ID = 1    # Person.GOALIE
PLAYER_CLASS_ID = 2        # Person.PLAYER
REFEREE_CLASS_ID = 3       # Person.REFEREE

# New virtual classes for teams
TEAM_1_CLASS_ID = 6        # "team 1"
TEAM_2_CLASS_ID = 7        # "team 2"

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ["#FF1493", "#00BFFF", "#FF6347", "#FFD700"]

ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2,
)

ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex("#FFFFFF"),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


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


def _prepare_team_classifier(
    source_video_path: str,
    device: str,
) -> TeamClassifier | None:
    """
    Helper to collect crops and fit TeamClassifier from a video.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)

    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path,
        stride=STRIDE,
    )

    crops: List[np.ndarray] = []
    for frame in tqdm(frame_generator, desc="collecting crops for team classifier"):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        if len(players) == 0:
            continue
        crops += get_crops(frame, players)

    if len(crops) == 0:
        return None

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    return team_classifier


def run_infer_results(
    source_video_path: str,
    device: str = "cpu",
) -> Iterator[TVFrameResult]:
    """
    Generator: infer on video and yield TVFrameResult for each frame.
    - Bounding boxes: BALL(0), GOALIE(1), REFEREE(3), TEAM_1(6), TEAM_2(7)
    - Keypoints: list of (x, y) from pitch keypoints.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)

    team_classifier = _prepare_team_classifier(
        source_video_path=source_video_path,
        device=device,
    )

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    frame_id = 0
    for frame in frame_generator:
        # Pitch keypoints
        pitch_result = pitch_detection_model(frame, verbose=False)[0]
        keypoints_sv = sv.KeyPoints.from_ultralytics(pitch_result)

        # Person detections
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        # Split by base classes
        balls = detections[detections.class_id == BALL_CLASS_ID]
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        referees = detections[detections.class_id == REFEREE_CLASS_ID]

        # Team classification for players -> CLASS_ID 6 / 7
        if len(players) > 0 and team_classifier is not None:
            player_crops = get_crops(frame, players)
            players_team_id = team_classifier.predict(player_crops)  # 0 or 1

            mapped_class_ids = np.where(
                players_team_id == 0,
                TEAM_1_CLASS_ID,
                TEAM_2_CLASS_ID,
            )
            players.class_id = mapped_class_ids  # type: ignore[assignment]

        # Merge detections
        merged = sv.Detections.merge([balls, goalkeepers, players, referees])

        boxes: list[BoundingBox] = []
        if len(merged) > 0:
            xyxy = merged.xyxy
            class_ids = merged.class_id
            confidences = merged.confidence
            for i in range(len(merged)):
                x1, y1, x2, y2 = xyxy[i]
                boxes.append(
                    BoundingBox(
                        x1=int(x1),
                        y1=int(y1),
                        x2=int(x2),
                        y2=int(y2),
                        cls_id=int(class_ids[i]),
                        conf=float(confidences[i]),
                    )
                )

        # Keypoints: flatten first frame of keypoints, filter valid ones
        kpts: list[Tuple[int, int]] = []
        if keypoints_sv.xy is not None and len(keypoints_sv.xy) > 0:
            pts = keypoints_sv.xy[0]
            for x, y in pts:
                if x > 1 and y > 1:
                    kpts.append((int(x), int(y)))

        yield TVFrameResult(
            frame_id=frame_id,
            boxes=boxes,
            keypoints=kpts,
        )
        frame_id += 1


def visualize_from_results(
    source_video_path: str,
    target_video_path: str,
    results: Iterator[TVFrameResult],
) -> None:
    """
    Visualize from pre-computed TVFrameResult (không chạy lại model).
    - Vẽ keypoints
    - Vẽ boxes với cls_id, conf
    """
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    video_info = sv.VideoInfo.from_video_path(source_video_path)

    with sv.VideoSink(target_video_path, video_info) as sink:
        for frame, frame_result in zip(frame_generator, results):
            annotated_frame = frame.copy()

            # Vẽ keypoints
            for x, y in frame_result.keypoints:
                cv2.circle(annotated_frame, (x, y), radius=4, color=(0, 255, 0), thickness=-1)

            # Vẽ boxes
            for box in frame_result.boxes:
                color = (0, 0, 255)
                if box.cls_id == TEAM_1_CLASS_ID:
                    color = (255, 0, 0)  # team 1
                elif box.cls_id == TEAM_2_CLASS_ID:
                    color = (0, 255, 255)  # team 2

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

            cv2.imshow("infer", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cv2.destroyAllWindows()


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
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument(
        "--show_radar",
        action="store_true",
        help="Overlay a small radar view using keypoints + team positions",
    )
    args = parser.parse_args()

    # Nếu dùng như library/API thì gọi trực tiếp run_infer_results(...)
    # CLI: có thể chọn xuất JSON, visualize video, hoặc cả hai.
    results_for_visualize: List[TVFrameResult] | None = None
    if args.visualize:
        results_for_visualize = []

    # Chạy inference base model trước
    if args.json_output_path is not None or args.visualize:
        with open(args.json_output_path, "w", encoding="utf-8") if args.json_output_path is not None else None as f:  # type: ignore[assignment]
            for tv_frame in run_infer_results(
                source_video_path=args.source_video_path,
                device=args.device,
            ):
                if args.json_output_path is not None and f is not None:
                    f.write(tv_frame.model_dump_json())
                    f.write("\n")
                if results_for_visualize is not None:
                    results_for_visualize.append(tv_frame)

    # Visualization chỉ chạy khi có flag --visualize
    if args.visualize:
        # Nếu --visualize mà chưa truyền target_video_path
        # thì tự sinh một tên file mới.
        if args.target_video_path is None:
            base, ext = os.path.splitext(args.source_video_path)
            if not ext:
                ext = ".mp4"
            args.target_video_path = f"{base}_vis{ext}"

        if results_for_visualize is None:
            # Trường hợp hiếm: visualize nhưng không collect trước (không nên xảy ra)
            results_for_visualize = list(
                run_infer_results(
                    source_video_path=args.source_video_path,
                    device=args.device,
                )
            )

        visualize_from_results(
            source_video_path=args.source_video_path,
            target_video_path=args.target_video_path,
            results=iter(results_for_visualize),
        )

    if args.json_output_path is None and not args.visualize:
        # Không output gì -> chỉ chạy thử để check pipeline (không khuyến khích).
        # Ở đây ta vẫn chạy generator để đảm bảo không lỗi.
        for _ in run_infer_results(
            source_video_path=args.source_video_path,
            device=args.device,
        ):
            pass


if __name__ == "__main__":
    main()

