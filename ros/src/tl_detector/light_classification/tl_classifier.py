# updated
import rospy
import os
from styx_msgs.msg import TrafficLight
import six.moves.urllib as urllib
import tarfile
from PIL import Image
import tensorflow as tf 
import numpy as np 
from os import path
from utils import label_map_util
import cv2
import time
import yaml

MAX_IMAGE_WIDTH = 300
MAX_IMAGE_HEIGHT = 225
# number of classes for my dataset
NUM_CLASSES = 4

# Path to frozen detection graph. This is the actual model that is used for the object detection.
# PATH_TO_CKPT = 'model/frozen_inference_graph_sim.pb'


class TLClassifier(object):
    def __init__(self):
        #TODO load classifier
        self.current_tl_light = TrafficLight.UNKNOWN
        self.model_graph = tf.Graph()
        with self.model_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(r'light_classification/model/frozen_inference_graph.pb', 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name= '')

        self.classes = {1: {'id': 1, 'name': 'Red'},
                        2: {'id': 2, 'name': 'Yellow'},
                        3: {'id': 3, 'name': 'Green'}, 
                        4: {'id': 4, 'name': 'off'}}   

        # create tensorflow session for detection
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.sess = tf.Session(graph=self.model_graph, config=config)
        self.image_tensor = self.model_graph.get_tensor_by_name('image_tensor:0')
        self.detection_boxes = self.model_graph.get_tensor_by_name('detection_boxes:0')
        # output
        self.detection_scores = self.model_graph.get_tensor_by_name('detection_scores:0')
        self.detection_classes = self.model_graph.get_tensor_by_name('detection_classes:0')
        self.num_detections = self.model_graph.get_tensor_by_name('num_detections:0')

    def get_classification(self, image):
        """Determines the color of the traffic light in the image

        Args:
            image (cv::Mat): image containing the traffic light

        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)

        """
        #TODO implement light color prediction
        image_res = cv2.resize(image, None, fx=0.25, fy=0.25)
        image_rgb = cv2.cvtColor(image_res, cv2.COLOR_BGR2RGB)
        image_np = np.expand_dims(image_rgb,axis=0)


        with self.model_graph.as_default():
            (boxes, scores, classes, num) = self.sess.run([self.detection_boxes, self.detection_scores,self.detection_classes, self.num_detections],feed_dict={self.image_tensor: image_np})
        boxes = np.squeeze(boxes)
        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)
        print('scores: ', scores[0])
        print('classes: ', classes[0])
        print('boxes: ', boxes[0])
        if len(scores) < 1 or scores[0] < 0.5:
            return TrafficLight.UNKNOWN

        if scores[0] > 0.5:
            if classes[0] == 1:
                return TrafficLight.RED
            elif classes[0] == 2:
                return TrafficLight.YELLOW
            elif classes[0] == 3:
                return TrafficLight.GREEN


        return TrafficLight.UNKNOWN