__author__ = 'ataylor'
import requests
from bs4 import BeautifulSoup


urls = [
"http://wiki.ros.org/amcl?distro=jade",
"http://wiki.ros.org/base_local_planner?distro=jade",
"http://wiki.ros.org/carrot_planner?distro=jade",
"http://wiki.ros.org/clear_costmap_recovery?distro=jade",
"http://wiki.ros.org/costmap_2d?distro=jade",
"http://wiki.ros.org/dwa_local_planner?distro=jade",
"http://wiki.ros.org/fake_localization?distro=jade",
"http://wiki.ros.org/global_planner?distro=jade",
"http://wiki.ros.org/map_server?distro=jade",
"http://wiki.ros.org/move_base?distro=jade",
"http://wiki.ros.org/move_base_msgs?distro=jade",
"http://wiki.ros.org/move_slow_and_clear?distro=jade",
"http://wiki.ros.org/nav_core?distro=jade",
"http://wiki.ros.org/navfn?distro=jade",
"http://wiki.ros.org/robot_pose_ekf?distro=jade",
"http://wiki.ros.org/rotate_recovery?distro=jade",
"http://wiki.ros.org/voxel_grid?distro=jade",
]


def get_param_thing(soup):
    param1 = soup.find(id='ROS_Parameters')
    if param1 is not None:
        return param1
    param2 = soup.find(id='Parameters')
    if param2 is not None:
        return param2
    param3 = soup.find(id="Parameters")
    if param3 is not None:
        return param3
    else:
        'no here'
        return None



total = 0
for url in urls:
    r = requests.get(url)
    count = 0
    soup = BeautifulSoup(r.text)
    print url
    param = get_param_thing(soup)
    for x in r.text:
        if x == '~':
            count += 1
    print count
    print '-----'
    total += count

print 'TOTAL: ', total



