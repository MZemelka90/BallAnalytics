import unittest
import numpy as np
import cv2
from Kalman_Approach.kalman_filter import KalmanFilter, create_cost_matrix, update_kalman_filters


class TestKalmanFilter(unittest.TestCase):
    def setUp(self):
        """Set up a basic Kalman filter for testing."""
        self.kf = KalmanFilter()

    def test_kalman_initialization(self):
        """Test if the Kalman filter initializes correctly."""
        self.assertTrue(np.array_equal(self.kf.kf.transitionMatrix, np.array([
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)))
        self.assertTrue(np.array_equal(self.kf.kf.measurementMatrix, np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)))
        self.assertEqual(self.kf.velocity, 0.0)

    def test_kalman_prediction(self):
        """Test the predict method of the Kalman filter."""
        predicted = self.kf.predict()
        self.assertEqual(predicted.flatten().shape, (4,))

    def test_kalman_update(self):
        """Test the update method with a sample measurement."""
        measurement = np.array([[10], [15]], dtype=np.float32)
        updated = self.kf.update(measurement)
        self.assertEqual(updated.reshape(-1).shape, (4,))

    def test_color_histogram(self):
        """Test setting and comparing color histograms."""
        # Create a dummy frame with solid color
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        cv2.rectangle(frame, (10, 10), (50, 50), (0, 255, 0), -1)

        # Set color histogram
        self.kf.set_color_histogram(frame, (10, 10, 50, 50))
        self.assertIsNotNone(self.kf.color_histogram)

        # Compare with another region
        similarity = self.kf.compare_color_histogram(frame, (10, 10, 50, 50))
        self.assertAlmostEqual(similarity, 0.0, delta=0.01)


class TestCostMatrix(unittest.TestCase):
    def test_create_cost_matrix(self):
        """Test creation of a cost matrix."""
        kf_objects = [KalmanFilter(), KalmanFilter()]
        measurements = [np.array([10, 15], dtype=np.float32), np.array([20, 25], dtype=np.float32)]
        bounding_boxes = [(10, 10, 20, 20), (15, 15, 30, 30)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        cost_matrix = create_cost_matrix(kf_objects, measurements, bounding_boxes, frame)
        self.assertEqual(cost_matrix.shape, (2, 2))
        self.assertTrue(np.all(cost_matrix >= 0))

    def test_empty_cost_matrix(self):
        """Test handling of empty input lists."""
        cost_matrix = create_cost_matrix([], [], [], None)
        self.assertIsNone(cost_matrix)


class TestUpdateKalmanFilters(unittest.TestCase):
    def test_update_kalman_filters(self):
        """Test updating Kalman filters with measurements."""
        kf_objects = [KalmanFilter()]
        measurements = [np.array([10, 15], dtype=np.float32)]
        bounding_boxes = [(10, 10, 20, 20)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        # Compute cost matrix
        cost_matrix = create_cost_matrix(kf_objects, measurements, bounding_boxes, frame)

        # Update Kalman filters
        update_kalman_filters(kf_objects, measurements, bounding_boxes, frame, cost_matrix)

        # Check if Kalman filter was updated
        corrected_state = kf_objects[0].update(measurements[0])
        self.assertTrue(np.allclose(corrected_state[:2].reshape(-1), measurements[0], atol=1e-5))

    def test_add_new_kalman_filters(self):
        """Test adding new Kalman filters if there are more measurements."""
        kf_objects = []
        measurements = [np.array([10, 15], dtype=np.float32)]
        bounding_boxes = [(10, 10, 20, 20)]
        frame = np.zeros((100, 100, 3), dtype=np.uint8)

        update_kalman_filters(kf_objects, measurements, bounding_boxes, frame, None)
        self.assertEqual(len(kf_objects), 1)


if __name__ == "__main__":
    unittest.main()
