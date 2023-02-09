#!/usr/bin/env python

from math import *
import numpy as np
import rospy
from geometry_msgs.msg import Pose2D, Pose
from sensor_msgs.msg import Imu


from EKFSLAM import *
from copy import copy


class EKFSLAMNode(object):
    # ---------------------------------------------------------------------
    # ---------------------- initialization -------------------------------
    # ---------------------------------------------------------------------

    def __init__(self):
        # motion model: 0 --> velocity based motion model, 1 --> constant velocity model,
        # 2 --> rpm model, 3 --> IMU model
        self.motionmodel = 1

        # initial pose of SAM
        x0 = np.array([0., 0., 0.])

        # initialize EKF SLAM
        self.ekf = EKFSLAM(x0)

        # initialize publishers & subscribers
        self.subscribers = {}
        self.init_subscribers()

        self.publishers = {}
        self.init_publishers()

        # 3D acceleration of SAM
        self.acc = []

        # gyro data
        self.gyro = []

        self.dt = 0.01

    def init_subscribers(self):
        self.subscribers["RelativePose"] = rospy.Subscriber("~/rel_pose", Pose, self.updateEKF)
        self.subscribers["imu"] = rospy.Subscriber("~/imu_data", Imu, self.process_imu)
        self.subscribers["control"] = rospy.Subscriber("~/odometry_data", Pose, self.predictEKF)

    def init_publishers(self):
        """ initialize ROS publishers and stores them in a dictionary"""
        # position of segway in world frame
        self.publishers["robot_pose"] = rospy.Publisher("~robot_pose", Pose2D, queue_size=1)
        self.publishers["station_pose"] = rospy.Publisher("~covariance", Pose2D, queue_size=1)

    def process_imu(self, msg):
        self.acc = [msg.linear_acceleration.x,
                    msg.linear_acceleration.y,
                    msg.linear_acceleration.z]

        # TODO: which coordinate frame?
        self.gyro = [msg.angular_velocity.x,
                     msg.angular_velocity.y,
                     msg.angular_velocity.z]

    def predictEKF(self, msg):
        controls = msg.data
        self.ekf.predict(controls, self.dt)
        self.publish_poses()

    def updateEKF(self, msg):
        meas = np.array([msg.position.x, msg.position.y])
        self.ekf.update(meas)
        self.publish_poses()

    def publish_poses(self):
        SAM_pose = Pose2D()
        SAM_pose.x = self.ekf.x[0]
        SAM_pose.y = self.ekf.x[1]
        SAM_pose.theta = self.ekf.x[2]

        self.publishers["robot_pose"].publish(SAM_pose)

        landmark_pose = Pose2D()
        landmark_pose.x = self.ekf.x[3]
        landmark_pose.y = self.ekf.x[4]
        landmark_pose.theta = 0.0

        self.publishers["landmark_pose"].publish(landmark_pose)



def main():
    """Starts the EKF SLAM Node"""
    rospy.init_node("EKF_Node")
    EKFSLAMNode()
    rospy.spin()


if __name__ == "__main__":
    main()





