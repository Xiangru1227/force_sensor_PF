import rospy
import numpy as np
from sensor_msgs.msg import Image
from message_filters import ApproximateTimeSynchronizer, Subscriber
from force_sensor.msg import DoubleWrenchStamped

def test_callback(image1: Image, image2: Image, wrench: DoubleWrenchStamped):
    
    rospy.loginfo("success")
    
    try:
        im1 = np.ndarray(shape=(image1.height, image1.width, 3), 
                         dtype=np.uint8, buffer=image1.data)
        im2 = np.ndarray(shape=(image2.height, image2.width, 3), 
                         dtype=np.uint8, buffer=image2.data)
    except Exception as e:
        rospy.logerr("Error converting images: {}".format(e))
        return

    LF = wrench.left_wrench.force

if __name__ == '__main__':
    rospy.init_node('test_node')
    
    l_image_sub = Subscriber('/camera1/image', Image)
    r_image_sub = Subscriber('/camera2/image', Image)
    wrenchStamped_sub = Subscriber('/wrench', DoubleWrenchStamped)
    
    ats = ApproximateTimeSynchronizer([l_image_sub, r_image_sub, wrenchStamped_sub], 
                                      queue_size=5, slop=0.1)
    ats.registerCallback(test_callback)

    rospy.spin()
