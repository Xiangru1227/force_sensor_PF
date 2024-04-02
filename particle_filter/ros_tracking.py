import rospy
import cv2
import time
import os
import sys

from geometry_msgs.msg import Vector3
from sensor_msgs.msg import Image, JointState
from force_sensor.msg import DoubleWrenchStamped
from scipy.spatial.transform import Rotation as rot
from message_filters import ApproximateTimeSynchronizer, Subscriber

script_path = os.path.dirname(os.path.abspath(__file__))
module_path = os.path.dirname(script_path)
if module_path not in sys.path:
    sys.path.append(module_path)
# print(module_path)

from core.RobotLink import *
from core.StereoCamera import *
from core.ParticleFilter import *
from core.probability_functions import *
from core.utils import *

# File inputs
robot_file    = script_path + '/../calibration_param/LND_erie.json'
camera_file   = script_path + '/../calibration_param/C1_fullrec_03072024.yaml'
hand_eye_file = script_path + '/../calibration_param/handeye_fixed_cam.yaml'

# ROS Topics
# left_camera_topic  = '/davinci_endo/left/image_color'
# right_camera_topic = '/davinci_endo/right/image_color'
left_camera_topic  = '/camera1/image'
right_camera_topic = '/camera2/image'
robot_joint_topic  = '/PSM2/measured_js'
robot_gripper_topic = '/PSM2/jaw/measured_js'
wrench_topic = '/wrench'

# Globals for callback function
cam = None # cam is global so we can "processImage" in callback
cb_detected_keypoints_l = None
cb_detected_keypoints_r = None
cb_left_img = None
cb_right_img = None
cb_joint_angles = None
left_force = None
right_force = None
new_cb_data = False
received_wrench = False

# ROS Callback for images and joint observations
def gotData(l_img_msg:Image, r_img_msg:Image, j_msg:JointState, g_msg:JointState, dualwrench:DoubleWrenchStamped):
    global cam, new_cb_data, cb_detected_keypoints_l, cb_detected_keypoints_r, cb_left_img, cb_right_img, cb_joint_angles, left_force, right_force, received_wrench
    try:
        # convert images to ndarrays
        _cb_left_img  = np.ndarray(shape=(l_img_msg.height, l_img_msg.width, 3),
                                      dtype=np.uint8, buffer=l_img_msg.data)
        _cb_right_img = np.ndarray(shape=(r_img_msg.height, r_img_msg.width, 3),
                                      dtype=np.uint8, buffer=r_img_msg.data)
        # resize and remap images
        _cb_left_img, _cb_right_img = cam.processImage(_cb_left_img, _cb_right_img)
    except:
        return
    # flip the images
    _cb_left_img = cv2.rotate(_cb_left_img, cv2.ROTATE_180)
    _cb_right_img = cv2.rotate(_cb_right_img, cv2.ROTATE_180)
    
    # find painted point features on tool
    cb_detected_keypoints_l, _cb_left_img  = segmentColorAndGetKeyPoints(_cb_left_img,  draw_contours=True)
    cb_detected_keypoints_r, _cb_right_img = segmentColorAndGetKeyPoints(_cb_right_img, draw_contours=True)

    cb_right_img = np.copy(_cb_right_img)
    cb_left_img  = np.copy(_cb_left_img)
    
    cb_joint_angles = np.array(j_msg.position + g_msg.position)
    cb_joint_angles[6] = cb_joint_angles[6] #- 1.3934 # 80-degree offset for the force sensor
    
    left_force = dualwrench.left_wrench.force
    right_force = dualwrench.right_wrench.force
    
    received_wrench = True
    new_cb_data = True

def publish_transformed_force(transformMatList, LF, RF):
        leftT = transformMatList[-2]
        rightT = transformMatList[-1]
        
        rotation_left = leftT[:3, :3]
        rotation_right = rightT[:3, :3]
        
        # convert vector to array for processing
        LF_arr = vector3_to_arr(LF)
        RF_arr = vector3_to_arr(RF)
        
        transformed_LF = np.dot(rotation_left, LF_arr)
        transformed_RF = np.dot(rotation_right, RF_arr)
        
        # convert back to vector for publishing
        transformed_LF_msg = arr_to_vector3(transformed_LF)
        transformed_RF_msg = arr_to_vector3(transformed_RF)
        
        transformed_LF_pub.publish(transformed_LF_msg)
        transformed_RF_pub.publish(transformed_RF_msg)
        
        # print(f"Transformed Left Force: {transformed_LF}")
        # print(f"Transformed Right Force: {transformed_RF}\n")

if __name__ == "__main__":
    # Initialize ROS node
    rospy.init_node('force_sensor_PF', anonymous=True)
    
    # Initialize transformed force publishers
    transformed_LF_pub = rospy.Publisher('/transformed_LF', Vector3, queue_size=10)
    transformed_RF_pub = rospy.Publisher('/transformed_RF', Vector3, queue_size=10)
    
    # Initialize subscribers
    l_image_sub = Subscriber(left_camera_topic, Image)
    r_image_sub = Subscriber(right_camera_topic, Image)
    robot_j_sub = Subscriber(robot_joint_topic, JointState)
    gripper_j_sub = Subscriber(robot_gripper_topic, JointState)
    wrenchStamped_sub = Subscriber(wrench_topic, DoubleWrenchStamped)
    
    # synchronize all topics
    ats = ApproximateTimeSynchronizer([l_image_sub, r_image_sub, robot_j_sub, gripper_j_sub, wrenchStamped_sub], 
                                      queue_size=5, slop=0.1) # slop = 0.015
    ats.registerCallback(gotData)
    
    # instantiate robot and camera
    robot_arm = RobotLink(robot_file)
    cam = StereoCamera(camera_file, rectify=False, downscale_factor=1)
    
    # load hand-eye transform
    f = open(hand_eye_file)
    hand_eye_data = yaml.load(f, Loader=yaml.FullLoader)
    # process hand-eye data
    cam_T_b = np.eye(4)
    cam_T_b[:-1, -1] = np.array(hand_eye_data['PSM1_tvec']) # /1000.0 # Input unit: m
    rot_axis = rot.from_euler('xyz',hand_eye_data['PSM1_rvec'],degrees=True)
    cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(rot_axis.as_rotvec())
    # cam_T_b[:-1, :-1] = axisAngleToRotationMatrix(np.radians(hand_eye_data['PSM1_rvec'])) # Input unit: degree
    
    # # if interocular is from left to right
    # rec_mat = np.array([[ 9.98826e-01,  3.88551e-03, -4.83061e-02, -2.62537e-03],
    #                     [-4.40321e-03,  9.99934e-01, -1.06167e-02, -2.70546e-04],
    #                     [ 4.82623e-02,  1.08175e-02,  9.98776e-01, -5.26980e-05],
    #                     [ 0.00000e+00,  0.00000e+00,  0.00000e+00,  1.00000e+00]])
    # # if from right to left
    # rec_mat = np.array([[1.00000202e+00, 1.06453370e-07, 5.85865807e-07, 2.62710013e-03],
    #                     [1.06453370e-07, 1.00000012e+00, 5.66247589e-07, 2.59791894e-04],
    #                     [5.85865807e-07, 5.66247589e-07, 9.99999692e-01, 1.82456194e-04],
    #                     [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
    # # dot product with transformation from cam center to left cam frame
    # cam_T_b = np.dot(cam_T_b, rec_mat)

    # Initialize filter
    pf = ParticleFilter(num_states=9, 
                        initialDistributionFunc=sampleNormalDistribution,
                        motionModelFunc=lumpedErrorMotionModel,
                        obsModelFunc=pointFeatureObs,
                        num_particles=200)

    init_kwargs = {
                    "std": np.array([1.0e-3, 1.0e-3, 1.0e-3, # pos
                                    1.0e-2, 1.0e-2, 1.0e-2, # ori
                                    #5.0e-3, 5.0e-3, 0.02
                                    0.0, 0.0, 0.0])   # joints
                  }

    pf.initializeFilter(**init_kwargs)
    
    rospy.loginfo("Particle filter initialized.")
       
    # Main loop:
    rate = rospy.Rate(60)
    prev_joint_angles = None
    
    while not rospy.is_shutdown():
        
        if new_cb_data and received_wrench:
            start_t = time.time()
            
            # Copy all the new data so they don't get over-written by callback
            new_detected_keypoints_l = np.copy(cb_detected_keypoints_l)
            new_detected_keypoints_r = np.copy(cb_detected_keypoints_r)
            new_left_img = np.copy(cb_left_img)
            new_right_img = np.copy(cb_right_img)
            new_joint_angles = np.copy(cb_joint_angles)
            new_cb_data = False
            received_wrench = False
            
            # First time
            if prev_joint_angles is None:
                prev_joint_angles = new_joint_angles
            
            # Predict Particle Filter
            robot_arm.updateJointAngles(new_joint_angles)
            j_change = new_joint_angles - prev_joint_angles

            std_j = np.abs(j_change)*0.01
            std_j[-3:] = 0.0

            pred_kwargs = {
                            "std_pos": 2.5e-5, 
                            "std_ori": 1.0e-4,
                            "robot_arm": robot_arm, 
                            "std_j": std_j,
                            "nb": 4
                          }
            pf.predictionStep(**pred_kwargs) 
            
            # Update Particle Filter
            upd_kwargs = {
                            "point_detections": (new_detected_keypoints_l, new_detected_keypoints_r), 
                            "robot_arm": robot_arm, 
                            "cam": cam, 
                            "cam_T_b": cam_T_b,
                            "joint_angle_readings": new_joint_angles,
                            "gamma": 0.15
            }

            # pf.updateStep(**upd_kwargs)
            prev_joint_angles = new_joint_angles

            correction_estimation = pf.getMeanParticle()
            
            T = poseToMatrix(correction_estimation[:6])
            new_joint_angles[-(correction_estimation.shape[0]-6):] += correction_estimation[6:]
            robot_arm.updateJointAngles(new_joint_angles)
            
            # rospy.loginfo("Time to predict & update {}".format(time.time() - start_t))

            publish_transformed_force(robot_arm.baseToJointT, left_force, right_force)
            
            # Project skeleton
            img_list = projectSkeleton(robot_arm.getSkeletonPoints(), np.dot(cam_T_b, T),
                                       [new_left_img, new_right_img], cam.projectPoints)
            
            # display images with point features and skeleton
            cv2.imshow("Left Img", img_list[0][:,:,[2,1,0]])
            cv2.imshow("Right Img", img_list[1][:,:,[2,1,0]])
            cv2.waitKey(1)
        
        rate.sleep()