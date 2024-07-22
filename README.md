mediapipe face detection
# mediapipe face detection 

import requests
import cv2
import numpy as np
import math
from typing import Tuple, Union
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 시각화 함수 정의 (앞서 설명한 함수들)c
MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red


def _normalized_to_pixel_coordinates(
        normalized_x: float, normalized_y: float, image_width: int,
        image_height: int) -> Union[None, Tuple[int, int]]:
    """Converts normalized value pair to pixel coordinates."""

    def is_valid_normalized_value(value: float) -> bool:
        return (value > 0 or math.isclose(0, value)) and (value < 1 or
                                                          math.isclose(1, value))

    if not (is_valid_normalized_value(normalized_x) and
            is_valid_normalized_value(normalized_y)):
        return None
    x_px = min(math.floor(normalized_x * image_width), image_width - 1)
    y_px = min(math.floor(normalized_y * image_height), image_height - 1)
    return x_px, y_px


def visualize(image, detection_result) -> np.ndarray:
    """Draws bounding boxes and keypoints on the input image and return it.
    Args:
      image: The input RGB image.
      detection_result: The list of all "Detection" entities to be visualize.
    Returns:
      Image with bounding boxes.
    """
    annotated_image = image.copy()
    height, width, _ = image.shape

    for detection in detection_result.detections:
        # Draw bounding_box
        bbox = detection.bounding_box
        start_point = int(bbox.origin_x), int(bbox.origin_y)
        end_point = int(bbox.origin_x + bbox.width), int(bbox.origin_y + bbox.height)
        cv2.rectangle(annotated_image, start_point, end_point, TEXT_COLOR, 3)

        # Draw keypoints
        for keypoint in detection.keypoints:
            keypoint_px = _normalized_to_pixel_coordinates(keypoint.x, keypoint.y,
                                                           width, height)
            if keypoint_px:
                color, thickness, radius = (0, 255, 0), 2, 2
                cv2.circle(annotated_image, keypoint_px, thickness, color, radius)

        # Draw label and score
        category = detection.categories[0]
        category_name = category.category_name
        category_name = '' if category_name is None else category_name
        probability = round(category.score, 2)
        result_text = category_name + ' (' + str(probability) + ')'
        text_location = (MARGIN + int(bbox.origin_x),
                         MARGIN + ROW_SIZE + int(bbox.origin_y))
        cv2.putText(annotated_image, result_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                    FONT_SIZE, TEXT_COLOR, FONT_THICKNESS)

    return annotated_image


# Mediapipe FaceDetector를 사용하여 얼굴 감지 및 시각화
def detect_and_visualize_faces(image_path):
    # STEP 1: Create a FaceDetector object.
    base_options = python.BaseOptions(model_asset_path='C:/Users/Mediapipeee/detector.tflite')
    options = vision.FaceDetectorOptions(base_options=base_options)
    detector = vision.FaceDetector.create_from_options(options)

    # STEP 2: Load the input image.
    image = mp.Image.create_from_file(image_path)

    # STEP 3: Detect faces in the input image.
    detection_result = detector.detect(image)

    # STEP 4: Process the detection result. In this case, visualize it.
    image_copy = np.copy(image.numpy_view())
    annotated_image = visualize(image_copy, detection_result)
    rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)

    # 시각화된 이미지 표시
    cv2.imshow('Annotated Image', rgb_annotated_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 이미지 URL
image_url = 'https://cdn.pixabay.com/photo/2015/10/08/05/06/brother-977170_1280.jpg'
image_path = 'image.jpg'

# 이미지 다운로드
response = requests.get(image_url)
if response.status_code == 200:
    with open(image_path, 'wb') as f:
        f.write(response.content)

# 이미지 파일 읽기
image = cv2.imread(image_path)

# 이미지가 제대로 읽혔는지 확인
if image is not None:
    # Mediapipe FaceDetector를 사용하여 얼굴 감지 및 시각화
    detect_and_visualize_faces(image_path)
else:
    print("이미지를 불러오지 못했습니다.")
