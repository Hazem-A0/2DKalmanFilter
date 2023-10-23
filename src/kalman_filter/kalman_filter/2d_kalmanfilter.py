import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
import numpy as np
from numpy.linalg import inv
import matplotlib.pyplot as plt

class KalmanFilter(Node):
    def __init__(self):
        super().__init__('kalman_filter_node')
        # Initialize kalman variables

        self.A = np.eye(4)  # State transition matrix
        self.B = np.zeros((4, 2))  # Control matrix (not used in this example)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])  # Measurement matrix
        self.R = np.eye(2)  # Measurement noise covariance
        self.Q = np.eye(4)  # Process noise covariance
        self.P = np.eye(4)  # Initial estimate covariance
        self.x_hat = np.zeros((4, 1))  # Initial state estimate [x, y, vx, vy]

         # Lists for plotting
        self.estimated_positions = []
        self.ground_truth_positions = []

        # Subscribe to the /odom_noise topic
        self.subscription = self.create_subscription(Odometry,
                                                     '/odom_noise',
                                                     self.odom_callback,
                                                     1)
        
        #publish the estimated reading
        self.estimated_pub=self.create_publisher(Odometry,
                                                 "/odom_estimated",1)

    def odom_callback(self, msg):
        # Extract the position measurements from the Odometry message
        z = np.array([[msg.pose.pose.position.x], [msg.pose.pose.position.y]])
        # Prediction step
        self.x_hat = np.dot(self.A, self.x_hat)  # Predicted state estimate
        self.P = np.dot(np.dot(self.A, self.P), self.A.T) + self.Q  # Predicted estimate covariance
        
        # Update step
        y = z - np.dot(self.H, self.x_hat)  # Measurement residual
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Residual covariance
        K = np.dot(np.dot(self.P, self.H.T), inv(S))  # Kalman gain
        self.x_hat = self.x_hat + np.dot(K, y)  # Updated state estimate
        self.P = np.dot(np.eye(4) - np.dot(K, self.H), self.P)  # Updated estimate covariance
        
        self.estimated_positions.append((self.x_hat[0, 0], self.x_hat[1, 0]))
        self.ground_truth_positions.append((msg.pose.pose.position.x, msg.pose.pose.position.y))

        #publish the estimated reading
        estimated_msg = Odometry()
        estimated_msg.pose.pose.position.x = self.x_hat[0, 0]
        estimated_msg.pose.pose.position.y = self.x_hat[1, 0]
        self.estimated_pub.publish(estimated_msg)

        # Plot the estimated and ground truth positions
        self.plot_positions()

    def plot_positions(self):
        est_x ,est_y = zip(*self.estimated_positions)
        gt_x, gt_y = zip(*self.ground_truth_positions)
        plt.figure(figsize=(8, 6))
        plt.plot(est_x, est_y, label='Estimated')
        plt.plot(gt_x, gt_y, label='Ground Truth')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend()
        plt.title('Kalman Filter Position Estimation')
        plt.grid(True)
        plt.show()

        pass    

def main(args=None):
    rclpy.init(args=args)
    node = KalmanFilter()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
