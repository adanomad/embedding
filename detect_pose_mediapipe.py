# takes an image path as a parameter,
# uses the MediaPipe Pose Landmarker task to detect pose landmarks,
# and outputs the results.


import mediapipe as mp
import sys

from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

model_path = "pose_landmarker_lite.task"


def detect_pose(image_path):
    BaseOptions = mp.tasks.BaseOptions
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    VisionRunningMode = mp.tasks.vision.RunningMode

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.IMAGE,
        output_segmentation_masks=True,
    )

    detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)
    if detection_result.pose_landmarks:
        print("Pose landmarks detected:")
        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), detection_result
        )
        # Visualize the pose landmarks on the image.
        cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
        cv2.waitKey(0)  # Wait for a key press to close the window

        # Check if segmentation masks are available
        if detection_result.segmentation_masks:
            segmentation_mask = detection_result.segmentation_masks[0].numpy_view()
            visualized_mask = (
                np.repeat(segmentation_mask[:, :, np.newaxis], 3, axis=2) * 255
            )
            cv2.imshow("Segmentation Mask", visualized_mask)
            cv2.waitKey(0)  # Wait for a key press to close the window

    else:
        print("No pose landmarks detected.")


def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected poses to visualize.
    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        # Draw the pose landmarks.
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend(
            [
                landmark_pb2.NormalizedLandmark(
                    x=landmark.x, y=landmark.y, z=landmark.z
                )
                for landmark in pose_landmarks
            ]
        )
        solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            solutions.pose.POSE_CONNECTIONS,
            solutions.drawing_styles.get_default_pose_landmarks_style(),
        )
    return annotated_image


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python pose_landmark_detector.py <image_path>")
        sys.exit(1)

    image_path = sys.argv[1]
    detect_pose(image_path)
