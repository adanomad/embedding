# takes an image path as a parameter,
# uses the MediaPipe Pose Landmarker task to detect pose landmarks,
# and outputs the results.

import json
import sys
import time
import mediapipe as mp
import argparse
import glob
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

model_path = "pose_landmarker_lite.task"

# Initialize the pose detector
BaseOptions = mp.tasks.BaseOptions
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
    output_segmentation_masks=True,
    num_poses=10,  # maximum number of poses that can be detected
)

detector = mp.tasks.vision.PoseLandmarker.create_from_options(options)


def detect_pose(image_path):
    start_time = time.time()
    mp_image = mp.Image.create_from_file(image_path)
    detection_result = detector.detect(mp_image)
    if detection_result.pose_landmarks:
        print(f"Pose landmarks detected in {image_path}")
        annotated_image = draw_landmarks_on_image(
            mp_image.numpy_view(), detection_result
        )

        # Save the annotated image
        cv2.imwrite(
            image_path.replace(".jpg", ".pose.jpg"),
            cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR),
        )

        # Process and save landmarks and world landmarks as JSON
        landmarks_data = {
            "landmarks": [
                {"x": lm.x, "y": lm.y, "z": lm.z, "visibility": lm.visibility}
                for pose_landmarks in detection_result.pose_landmarks
                for lm in pose_landmarks
            ],
        }

        json_path = image_path.replace(".jpg", ".json")
        with open(json_path, "w") as json_file:
            json.dump(landmarks_data, json_file)

        print(f"Landmark data saved to {json_path}")

    else:
        print(f"No pose landmarks detected in {image_path}.")

    time_taken = time.time() - start_time
    print(f"{image_path}: {time_taken:.2f} seconds")


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
    parser = argparse.ArgumentParser(description="Pose Landmark Detector")
    parser.add_argument("--image", help="Path to an image file")
    parser.add_argument("--pattern", help="Glob pattern for images (e.g., ./*.jpg)")
    args = parser.parse_args()

    if args.image:
        detect_pose(args.image)
    elif args.pattern:
        # Retrieve all files matching the pattern and filter out .mask. and .pose.
        image_files = [
            file
            for file in glob.glob(args.pattern)
            if ".mask." not in file and ".pose." not in file
        ]
        image_files.sort()  # Sort the files

        # Process each image in sorted order
        for image_file in image_files:
            detect_pose(image_file)

    else:
        print(
            "No input provided. Use --image for a single image or --pattern for a glob pattern of images."
        )
        sys.exit(1)
