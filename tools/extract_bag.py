#!/usr/bin/env python

import rosbag_pandas
import sys
import os

to_read = [
# '/a/cmd_subject_ctrl_state',
# '/a/distance_to_task_pose',
# '/a/h2o_sample/',
# '/a/land_waypose',
# '/a/launch_waypose',
# '/a/pid_input',
# '/a/pid_waypose',
# '/a/pump_ctrl_debug',
# '/a/quad_ctrl_debug',
# '/a/quad_ctrl_input',
# '/a/robot_gps',
# '/a/robot_imu',
# '/a/robot_status',
# '/a/subject_ctrl_state',
# '/a/subject_pose',
# '/a/subject_pose_vel',
# '/a/subject_status',
# '/a/task_waypose',
# '/mark_action/',
# '/keyboard_keydown',
# '/keyboard_keyup',
# '/mark_no_action',
# '/mark_action',
# '/marks1/mark_action',
# '/marks0/mark_no_action',
# '/marks1/mark_action',
# '/marks1/mark_no_action',
# '/marks2/mark_action',
# '/marks2/mark_no_action',
'/threshold_information',
]

f = sys.argv[1]
print f
f = os.path.abspath(f)
print f
b,_ = os.path.splitext(f)
print b
df = rosbag_pandas.bag_to_dataframe(f, include=to_read)
print len(df)
df.to_csv(b + '.csv')
