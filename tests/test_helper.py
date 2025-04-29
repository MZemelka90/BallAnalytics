import unittest
from unittest.mock import MagicMock, patch, call
import numpy as np
import os
import cv2
from helper_functions.helper import get_file_path_in_project, draw_trail, draw_ball_statistics, initialize_video_writer, \
    extract_ball_positions_and_bounding_boxes


class TestFunctions(unittest.TestCase):

    @patch('os.path.dirname')
    @patch('os.path.abspath')
    def test_get_file_path_in_project(self, mock_abspath, mock_dirname):
        # Setup
        mock_abspath.return_value = "/project/root"
        mock_dirname.return_value = "/project"

        dir_name = "data"
        file_name = "test_file.txt"

        # Call function
        result = get_file_path_in_project(dir_name, file_name)

        # Verify
        expected = r"/project\data\test_file.txt"
        self.assertEqual(result, expected)

    @patch('cv2.line')
    def test_draw_trail(self, mock_line):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ball_positions = {
            0: [(10, 10), (20, 20), (30, 30)],  # Ball 0
            1: [(40, 40), (50, 50), (60, 60)]  # Ball 1
        }

        # Call function
        draw_trail(frame, ball_positions)

        # Verify the calls to cv2.line were made with the correct arguments
        expected_calls = [
            call(frame, (10, 10), (20, 20), (0, 255, 255), 2),
            call(frame, (20, 20), (30, 30), (0, 255, 255), 2),
            call(frame, (40, 40), (50, 50), (255, 0, 255), 2),
            call(frame, (50, 50), (60, 60), (255, 0, 255), 2)
        ]

        mock_line.assert_has_calls(expected_calls, any_order=True)

    @patch('cv2.putText')
    def test_draw_ball_statistics(self, mock_putText):
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ball_detections = {'balls': 10}
        frame_count = 100

        # Call function
        draw_ball_statistics(frame, ball_detections, frame_count)

        # Verify
        mock_putText.assert_called_with(frame,
                                        'Ball detection percentage: 3.33%',
                                        (20, 100),
                                        cv2.FONT_HERSHEY_SIMPLEX,
                                        1.2,
                                        (0, 0, 0),
                                        3)

    @patch('cv2.VideoCapture')  # Patch the VideoCapture class
    @patch('cv2.VideoWriter')  # Patch the VideoWriter class
    def test_initialize_video_writer(self, MockVideoWriter, MockVideoCapture):
        # Create a mock VideoCapture object
        mock_cap = MagicMock()
        MockVideoCapture.return_value = mock_cap  # Make sure VideoCapture returns the mock object

        # Mock the values that cap.get() should return
        mock_cap.get.side_effect = lambda prop: {
            cv2.CAP_PROP_FRAME_WIDTH: 640,
            cv2.CAP_PROP_FRAME_HEIGHT: 480,
            cv2.CAP_PROP_FPS: 30
        }.get(prop, None)

        # Mock the VideoWriter to avoid creating an actual file
        mock_video_writer = MagicMock()
        MockVideoWriter.return_value = mock_video_writer

        # Call the function that uses VideoCapture
        output_path = 'output_video.avi'
        video_writer = initialize_video_writer(mock_cap, output_path)

        # Ensure that VideoWriter was called with the correct parameters
        expected_frame_width = 640
        expected_frame_height = 480
        expected_fps = 30

        # Verify that VideoWriter was called with expected arguments
        MockVideoWriter.assert_called_with(output_path, cv2.VideoWriter_fourcc(*"XVID"), expected_fps,
                                           (expected_frame_width, expected_frame_height))

        # Verify that get() was called for the correct properties
        mock_cap.get.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH)
        mock_cap.get.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT)
        mock_cap.get.assert_any_call(cv2.CAP_PROP_FPS)

    @patch('ultralytics.YOLO')  # Patch YOLO class
    def test_extract_ball_positions_and_bounding_boxes(self, MockYOLO):
        # Create a mock instance of YOLO
        mock_yolo_instance = MagicMock()
        MockYOLO.return_value = mock_yolo_instance  # This ensures that when YOLO() is called, it returns mock_yolo_instance

        # Simulate the detection results from YOLO
        mock_result = MagicMock()

        # Mock the result from boxes.cpu().numpy() for two balls detected
        mock_result.boxes.cpu().numpy.return_value = [
            MagicMock(conf=[0.8], cls=[0], xyxy=[[10, 10, 30, 30]]),  # Ball 1
            MagicMock(conf=[0.9], cls=[0], xyxy=[[40, 40, 60, 60]])  # Ball 2
        ]

        # Instead of patching __call__, we'll patch model(frame) directly to return the mock result
        mock_yolo_instance.__getitem__.return_value = [
            mock_result]  # This simulates model(frame) call returning mock_result

        # Prepare frame and ball_detections
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        ball_detections = {'balls': 0}

        # Call the function
        measurements, bounding_boxes = extract_ball_positions_and_bounding_boxes(mock_yolo_instance, frame,
                                                                                 ball_detections)

        # Verify the expected results
        self.assertEqual(len(measurements), 2)  # There should be 2 balls detected
        self.assertEqual(len(bounding_boxes), 2)  # There should be 2 bounding boxes
        self.assertEqual(ball_detections['balls'], 2)  # The ball detection count should be 2
        self.assertEqual(measurements[0].tolist(), [[10], [10]])  # First ball's center (10, 10)
        self.assertEqual(bounding_boxes[0], (10, 10, 30, 30))  # First ball's bounding box


if __name__ == '__main__':
    unittest.main()
