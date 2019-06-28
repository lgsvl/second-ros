#!/usr/bin/env python

import os
import sys
import time
import json
import rospy
import numpy as np
from numpy.lib.recfunctions import structured_to_unstructured
from pyquaternion import Quaternion
from std_msgs.msg import Header
from sensor_msgs.msg import PointCloud2, PointField
import sensor_msgs.point_cloud2 as pc2
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

sys.path.append("/root/second.pytorch")
import second.core.box_np_ops as box_np_ops
from second.pytorch.inference import TorchInferenceContext
from second.data.kitti_common import _extend_matrix, get_kitti_image_info


# mappings between PointField types and numpy types
type_mappings = [(PointField.INT8, np.dtype('int8')), (PointField.UINT8, np.dtype('uint8')), (PointField.INT16, np.dtype('int16')),
                 (PointField.UINT16, np.dtype('uint16')), (PointField.INT32, np.dtype('int32')), (PointField.UINT32, np.dtype('uint32')),
                 (PointField.FLOAT32, np.dtype('float32')), (PointField.FLOAT64, np.dtype('float64'))]
pftype_to_nptype = dict(type_mappings)
nptype_to_pftype = dict((nptype, pftype) for pftype, nptype in type_mappings)

# sizes (in bytes) of PointField types
pftype_sizes = {PointField.INT8: 1, PointField.UINT8: 1, PointField.INT16: 2, PointField.UINT16: 2,
                PointField.INT32: 4, PointField.UINT32: 4, PointField.FLOAT32: 4, PointField.FLOAT64: 8}

DUMMY_FIELD_PREFIX = '__'


class SecondROS:
    def __init__(self):
        rospy.init_node('second_ros')

        # Subscriber
        self.sub_lidar = rospy.Subscriber("/kitti/velo/pointcloud", PointCloud2, self.lidar_callback, queue_size=1)
        # self.sub_lidar = rospy.Subscriber("/apollo/sensor/velodyne64/compensator/PointCloud2", PointCloud2, self.lidar_callback, queue_size=1)

        # Publisher
        self.pub_bbox = rospy.Publisher("/detections", BoundingBoxArray, queue_size=1)

        # Kitti data path
        data_path = '/root/data/kitti'

        # KITTI
        # config_path = '/root/model/kitti/pipeline.config'
        # ckpt_path = '/root/model/kitti/voxelnet-99040.tckpt'

        # LGSVL
        config_path = '/root/model/hybrid_v4/pipeline.config'
        ckpt_path = '/root/model/hybrid_v4/voxelnet-817104.tckpt'

        self.model = SecondModel(data_path, config_path, ckpt_path)
        self.model.initialize()

        rospy.spin()
    
    def lidar_callback(self, msg):
        if self.model.inference_ctx is None or self.model.inference_ctx.anchor_cache is None:
            return

        for field in msg.fields:
            if field.name == "i" or field.name == "intensity":
                intensity_fname = field.name
                intensity_dtype = field.datatype
            else:
                intensity_fname = None
                intensity_dtype = None
            
        dtype_list = self._fields_to_dtype(msg.fields, msg.point_step)
        pc_arr = np.frombuffer(msg.data, dtype_list)
        
        if intensity_fname:
            pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z", intensity_fname]]).copy()
            if intensity_dtype == 2:
                pc_arr[:, 3] = pc_arr[:, 3] / 255
        else:
            pc_arr = structured_to_unstructured(pc_arr[["x", "y", "z"]]).copy()
            pc_arr = np.hstack((pc_arr, np.zeros((pc_arr.shape[0], 1))))

        lidar_boxes = self.model.predcit(pc_arr)
        
        num_detects = len(lidar_boxes)
        arr_bbox = BoundingBoxArray()
        for i in range(num_detects):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()

            bbox.pose.position.x = float(lidar_boxes[i][0])
            bbox.pose.position.y = float(lidar_boxes[i][1])
            bbox.pose.position.z = float(lidar_boxes[i][2]) + float(lidar_boxes[i][5]) / 2
            bbox.dimensions.x = float(lidar_boxes[i][3])  # width
            bbox.dimensions.y = float(lidar_boxes[i][4])  # length
            bbox.dimensions.z = float(lidar_boxes[i][5])  # height

            q = Quaternion(axis=(0, 0, 1), radians=float(lidar_boxes[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w

            arr_bbox.boxes.append(bbox)

        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        print("Number of detections: {}".format(num_detects))
        
        self.pub_bbox.publish(arr_bbox)

    def _fields_to_dtype(self, fields, point_step):
        '''Convert a list of PointFields to a numpy record datatype.
        '''
        offset = 0
        np_dtype_list = []
        for f in fields:
            while offset < f.offset:
                # might be extra padding between fields
                np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
                offset += 1

            dtype = pftype_to_nptype[f.datatype]
            if f.count != 1:
                dtype = np.dtype((dtype, f.count))

            np_dtype_list.append((f.name, dtype))
            offset += pftype_sizes[f.datatype] * f.count

        # might be extra padding between points
        while offset < point_step:
            np_dtype_list.append(('%s%d' % (DUMMY_FIELD_PREFIX, offset), np.uint8))
            offset += 1

        return np_dtype_list


class SecondModel:
    def __init__(self, data_path, config_path, ckpt_path, calib_idx=0):
        self.data_path = data_path
        self.config_path = config_path
        self.ckpt_path = ckpt_path
        self.calib_idx = calib_idx

        self.calib_info = None
        self.inference_ctx = None
    
    def initialize(self):
        image_infos = get_kitti_image_info(
            self.data_path,
            training=True,
            label_info=False,
            calib=True,
            image_ids=[self.calib_idx]
        )
        self.calib_info = image_infos[0]
        self._build()
        self._restore()
    
    def predcit(self, pointclouds):
        t = time.time()
        result_annos = self._inference(pointclouds)
        print("Inference time: {} ms".format(int((time.time() - t) * 1000)))
        kitti_anno = self.remove_low_score(result_annos[0])
        lidar_boxes = self.kitti_cam_to_lidar(kitti_anno)

        return lidar_boxes
    
    def _build(self):
        print("Start build...")
        self.inference_ctx = TorchInferenceContext()
        self.inference_ctx.build(self.config_path)
        print("Build succeeded.")

    def _restore(self):
        print("Start restore...")
        self.inference_ctx.restore(self.ckpt_path)
        print("Restore succeeded.")

    def _inference(self, pointclouds):
        inputs = self.inference_ctx.get_inference_input_dict(self.calib_info, pointclouds)
        det_annos = self.inference_ctx.inference(inputs)
        return det_annos
    
    def kitti_cam_to_lidar(self, kitti_anno):
        rect = self.calib_info['calib/R0_rect']
        Tr_velo_to_cam = self.calib_info['calib/Tr_velo_to_cam']
        dims = kitti_anno['dimensions']
        loc = kitti_anno['location']
        rots = kitti_anno['rotation_y']
        boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1)
        boxes_lidar = box_np_ops.box_camera_to_lidar(boxes_camera, rect, Tr_velo_to_cam)

        return boxes_lidar

    def remove_low_score(self, annos, threshold=0.5):
        img_filtered_annotations = {}
        relevant_annotation_indices = [i for i, s in enumerate(annos['score']) if s >= threshold]
        for key in annos.keys():
            img_filtered_annotations[key] = (annos[key][relevant_annotation_indices])

        return img_filtered_annotations


if __name__ == "__main__":
    second_ros = SecondROS()
