# Ball Analysis with various approaches (Kalman Filter at the moment)

This Project is a work in progress. I am currently working on a Kalman Filter approach.
The main goal is to track football balls of varying color at the same time while also having objects obstruct the view
at some times.

## Kalman Filter

The Kalman filter is a mathematical method that uses a combination of prediction and measurement updates to estimate 
the state of a system. It is commonly used in object tracking applications such as computer vision and robotics.

The prediction step uses the current state estimate and the system dynamics to predict the new state of the system. 
The measurement update step uses the current measurement and the system dynamics to update the state estimate. 
The prediction and measurement update steps are repeated in a loop to continuously track the system state.

The Kalman filter algorithm works as follows:

1. Initialize the state estimate and the error covariance matrix.
2. Predict the new state of the system using the current state estimate and the system dynamics.
3. Measure the system state using a sensor or a camera.
4. Update the state estimate using the measurement and the system dynamics.
5. Calculate the error covariance matrix using the measurement and the system dynamics.
6. Repeat steps 2-5 until the desired accuracy is achieved.

The Kalman filter has many advantages over other object tracking algorithms, such as being able to handle non-linear 
systems, being able to track multiple objects at once, and being able to handle noisy measurements. However, the Kalman 
filter can be computationally expensive and requires a good understanding of the system dynamics.

## Hungarian Algorithm

The Hungarian algorithm is a graph matching algorithm that is used to find the best matching between two sets of objects. 
It is used in object tracking applications such as computer vision and robotics.

The Hungarian algorithm works as follows:

1. Initialize a cost matrix that represents the cost of matching each object in one set to each object in the other set.
2. Run the Hungarian algorithm to find a matching between the two sets of objects.
3. Use the matching to find the best matching between the two sets of objects.
4. Repeat steps 1-3 until the desired accuracy is achieved.

The Hungarian algorithm has many advantages over other object tracking algorithms, such as being able to handle noisy 
measurements, being able to track multiple objects at once, and being able to handle non-linear systems. However, the 
Hungarian algorithm can be computationally expensive and requires a good understanding of the cost matrix.

## Running the code

The code can be run using the following command:

```bash
python main.py
```

## Install dependencies
The dependencies are listed in the pyproject.toml file. To install them, you can use pip and the pyproject.toml:

```bash
pip install -e .
```

