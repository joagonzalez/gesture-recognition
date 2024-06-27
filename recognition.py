import cv2
import tqdm
import itertools
import numpy as np
import rerun as rr
import mediapipe as mp
import numpy.typing as npt
from typing import Iterable
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.components.containers import NormalizedLandmark



class GestureDetectorLogger2:

    def __init__(self, video_mode: bool = False):
        self._video_mode = video_mode

        base_options = python.BaseOptions(
            model_asset_path='gesture_recognizer.task'
        )
        options = vision.GestureRecognizerOptions(
            base_options=base_options,
            running_mode=mp.tasks.vision.RunningMode.VIDEO if self._video_mode else mp.tasks.vision.RunningMode.IMAGE
        )
        self.recognizer = vision.GestureRecognizer.create_from_options(options)

        rr.log(
            "/",
            rr.AnnotationContext(
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="Hand3D"),
                    keypoint_connections=mp.solutions.hands.HAND_CONNECTIONS
                )
            ),
            timeless=True,
        )
        rr.log("Hand3D", rr.ViewCoordinates.RIGHT_HAND_X_DOWN, timeless=True)

    def detect(self, image: npt.NDArray[np.uint8]) -> None:
          image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

          # Get results from Gesture Detection model
          recognition_result = self.recognizer.recognize(image)

          for i, gesture in enumerate(recognition_result.gestures):
              # Get the top gesture from the recognition result
              print("Top Gesture Result: ", gesture[0].category_name)

          if recognition_result.hand_landmarks:
              # Obtain hand landmarks from MediaPipe
              hand_landmarks = recognition_result.hand_landmarks
              print("Hand Landmarks: " + str(hand_landmarks))

              # Obtain hand connections from MediaPipe
              mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
              print("Hand Connections: " + str(mp_hands_connections))

    def convert_landmarks_to_image_coordinates(self, hand_landmarks: list[list[NormalizedLandmark]], width: int, height: int) -> list[tuple[int, int]]:
        return [(int(lm.x * width), int(lm.y * height)) for hand_landmark in hand_landmarks for lm in hand_landmark]

    def convert_landmarks_to_3d(self, hand_landmarks: list[list[NormalizedLandmark]]) -> list[tuple[float, float, float]]:
        return [(lm.x, lm.y, lm.z) for hand_landmark in hand_landmarks for lm in hand_landmark]

    def detect_and_log(self, image: npt.NDArray[np.uint8], frame_time_nano: int | None) -> None:
        # Recognize gestures in the image
        height, width, _ = image.shape
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)

        recognition_result = (
            self.recognizer.recognize_for_video(image, int(frame_time_nano / 1e6))
            if self._video_mode
            else self.recognizer.recognize(image)
        )

        # Clear the values
        for log_key in ["Media/Points", "Media/Connections"]:
            rr.log(log_key, rr.Clear(recursive=True))

        for i, gesture in enumerate(recognition_result.gestures):
            # Get the top gesture from the recognition result
            gesture_category = gesture[0].category_name if recognition_result.gestures else "None"
            print("Gesture Category: ", gesture_category) # Log the detected gesture


        if recognition_result.hand_landmarks:
            hand_landmarks = recognition_result.hand_landmarks


            landmark_positions_3d = self.convert_landmarks_to_3d(hand_landmarks=hand_landmarks)
            if landmark_positions_3d is not None:
                rr.log(
                    "Hand3D/Points",
                    rr.Points3D(landmark_positions_3d, radii=20, class_ids=0, keypoint_ids=[i for i in range(len(landmark_positions_3d))]),
                )

            # Convert normalized coordinates to image coordinates
            points = self.convert_landmarks_to_image_coordinates(hand_landmarks, width, height)

            # Log points to the image and Hand Entity
            rr.log(
               "Media/Points",
                rr.Points2D(points, radii=10, colors=[255, 0, 0])
            )

            # Obtain hand connections from MediaPipe
            mp_hands_connections = mp.solutions.hands.HAND_CONNECTIONS
            points1 = [points[connection[0]] for connection in mp_hands_connections]
            points2 = [points[connection[1]] for connection in mp_hands_connections]

            # Log connections to the image and Hand Entity
            rr.log(
               "Media/Connections",
                rr.LineStrips2D(
                   np.stack((points1, points2), axis=1),
                   colors=[255, 165, 0]
                )
             )


def run_from_sample_image(path)-> None:
    image = cv2.imread(str(path))
    show_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    logger = GestureDetectorLogger2(video_mode=False)
    logger.detect(show_image)

def run_from_video_capture(vid: int | str, max_frame_count: int | None) -> None:
    """
    Run the detector on a video stream.

    Parameters
    ----------
    vid:
        The video stream to run the detector on. Use 0/1 for the default camera or a path to a video file.
    max_frame_count:
        The maximum number of frames to process. If None, process all frames.
    """
    cap = cv2.VideoCapture(vid)
    fps = cap.get(cv2.CAP_PROP_FPS)

    detector = GestureDetectorLogger2(video_mode=True)

    try:
        it: Iterable[int] = itertools.count() if max_frame_count is None else range(max_frame_count)

        for frame_idx in tqdm.tqdm(it, desc="Processing frames"):
            ret, frame = cap.read()
            if not ret:
                break

            if np.all(frame == 0):
                continue

            frame_time_nano = int(cap.get(cv2.CAP_PROP_POS_MSEC) * 1e6)
            if frame_time_nano == 0:
                frame_time_nano = int(frame_idx * 1000 / fps * 1e6)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            rr.set_time_sequence("frame_nr", frame_idx)
            rr.set_time_nanos("frame_time", frame_time_nano)
            detector.detect_and_log(frame, frame_time_nano)
            rr.log(
                "Media/Video",
                rr.Image(frame)
            )

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(e)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # Run the gesture recognition on a sample image
    # run_from_sample_image("./2.jpg")
    rr.init("rerun_example_my_data", spawn=True)
    run_from_video_capture(vid="/dev/video4", max_frame_count=None)
