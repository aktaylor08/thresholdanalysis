#!/usr/bin/python

import subprocess
import smtplib
import shutil
import os
import time
from email.mime.text import MIMEText
import argparse

import signal

CLANG_RESULTS_DIR = '/home/ataylor/clang_results/'
RESULTS_DIR = '/home/ataylor/results/'

FROM_EMAIL = 'rostestmailer@gmail.com'
TO_EMAIL = 'aktaylor08@gmail.com'

PASSWORD = "#MailMeResults"

class TestObject:
    """Hold test information needed to run the test"""

    def __init__(self, source, res, name):
        self.source_dir = source
        if self.source_dir[-1] != '/':
            self.source_dir += '/'
        self.test_name = name
        self.result_dir = res
        if self.result_dir[-1] != '/':
            self.result_dir += '/'

tests = [
    TestObject('/home/ataylor/ros_ws/pass_tests/', 'pass_test', 'pass unit tests'),
    TestObject('/home/ataylor/asctec_ws/', 'asctec', 'asctec base'),
    TestObject('/home/ataylor/ros_ws/water/', 'water', 'water sampler'),
    TestObject('/home/ataylor/ros_ws/crop_surveying/', 'crop_surveying', 'crop surveying'),
    TestObject('/home/ataylor/care_o_bot/cob_extern', 'cob_extern', 'cob external'),
    TestObject('/home/ataylor/care_o_bot/cob_common', 'cob_common', 'cob common'),
    TestObject('/home/ataylor/care_o_bot/cob_robots', 'cob_robots', 'cob_robots'),
    TestObject('/home/ataylor/care_o_bot/cob_environments', 'cob_environments', 'Cob Environments'),
    TestObject('/home/ataylor/care_o_bot/cob_command_tools', 'cob_command_tools', 'Cob Command Tools'),
    TestObject('/home/ataylor/care_o_bot/cob_calibration_data', 'cob_calibration_data', 'Cob calibration data'),
    TestObject('/home/ataylor/care_o_bot/cob_substitue', 'cob_substitute', 'Cob substitute'),
    TestObject('/home/ataylor/care_o_bot/cob_control', 'cob_control', 'careobot contro'),
    TestObject('/home/ataylor/care_o_bot/cob_percpetion_common', 'cob_perception_common', 'careobot perception'),
    TestObject('/home/ataylor/care_o_bot/cob_manipulation', 'cob_manipulation', 'careobot manipulation'),
    TestObject('/home/ataylor/care_o_bot/cob_enviornment_perception', 'cob_environment_perception', 'careobot evnironment perception'),
    TestObject('/home/ataylor/care_o_bot/cob_navigation', 'cob_navigation', 'careobot navigation perception'),
    TestObject('/home/ataylor/ros_ws/navigation/', 'navigation', 'navigation stack'),
    TestObject('/home/ataylor/ros_ws/baxter/', 'baxter', 'Baxter Robot'),
    TestObject('/home/ataylor/controllers/realtime_tools', 'realtime_tools', 'Realtime tools'),
    TestObject('/home/ataylor/controllers/control_toolbox', 'control_toolbox', 'Control Toolbox'),
    TestObject('/home/ataylor/controllers/ros_control', 'ros_control', 'Ros Control'),
    TestObject('/home/ataylor/controllers/ros_controllers', 'ros_controllers', 'Ros Controllers'),
    TestObject('/home/ataylor/bwi/', 'bwi', 'BWI from Texas'),
    TestObject('/home/ataylor/ros_ws/ar_track_alvar/', 'artrack', 'AR Tracking'),
    TestObject('/home/ataylor/ros_ws/apriltags/', 'apriltags', 'AprilTags Tracking'),
    TestObject('/home/ataylor/ros_ws/arbotix_ros/', 'arbotix', 'Airbotix ros package'),
    TestObject('/home/ataylor/ros_ws/asctec_mav/', 'asctec_mav', 'Asctec Mav Pacakge'),
    TestObject('/home/ataylor/ros_ws/uos_tools/', 'uos_tools', 'UOS Tools'),
    TestObject('/home/ataylor/ros_ws/calvin/', 'calvin', 'Calvin Ros Stack'),
    TestObject('/home/ataylor/ros_ws/crazyflie_ros/', 'crazyflie', 'CrazyFlie Ros Stack'),
    TestObject('/home/ataylor/rocon/', 'rocon', 'Ros Concert'),
    TestObject('/home/ataylor/ros_ws/create/', 'create', 'ROS Create Driver'),
    TestObject('/home/ataylor/ros_ws/darwin', 'darwin', 'ROS Darwin'),
    TestObject('/home/ataylor/ros_ws/descartes', 'descartes', 'ROS Descartes'),
    TestObject('/home/ataylor/ros_ws/fanuc', 'faunc', 'Func Maninulators'),
    TestObject('/home/ataylor/ros_ws/filter', 'filters', 'ros filter library'), TestObject('/home/ataylor/ros_ws/graft/', 'graft', 'graft'), TestObject('/home/ataylor/ros_ws/rail_pick_and_place/', 'rail_pick_place', 'Rail Pick and place library'),
    TestObject('/home/ataylor/ros_ws/grizzly/', 'grizzly', 'Grizzly robot'),
    TestObject('/home/ataylor/ros_ws/hector_slam/', 'hector_slam', 'Hector Slam'),
    TestObject('/home/ataylor/ros_ws/hector_small_arm_common/', 'hector_small_arm', 'Hector Arm'),
    TestObject('/home/ataylor/ros_ws/hector_navigation/', 'hector_nav', 'Hector navigation'),
    TestObject('/home/ataylor/ros_ws/hector_diagnostics/', 'hector_diagnostics', 'Hector diagnostics'),
    TestObject('/home/ataylor/ros_ws/hector_turtlebot/', 'hector_turtlebot', 'Hector turtlebot'),
    TestObject('/home/ataylor/ros_ws/husky/', 'husky', 'Husky Ros'),
    TestObject('/home/ataylor/ros_ws/icart_mini/', 'icart_mini', 'ICART MINI'),
    TestObject('/home/ataylor/ros_ws/innok_heros_driver/', 'innok_heros_driver', 'innok_heros'),
    TestObject('/home/ataylor/ros_ws/jaco/', 'jaco', 'Jaco Robot Arm'),
    TestObject('/home/ataylor/ros_ws/calibration/', 'calibration', 'Calibration'),
    TestObject('/home/ataylor/jsk/jsk_db/', 'jsk_db', 'jsk_db'),
    TestObject('/home/ataylor/jsk/jsk_recognition', 'jsk_recognition', 'jsk_recognition'),
    TestObject('/home/ataylor/jsk/jsk_smart_apps', 'jsk_smart_apps', 'jsk smart apps'),
    TestObject('/home/ataylor/jsk/jsk_visualization', 'jsk_vis', 'jsk_vis'),
    TestObject('/home/ataylor/jsk/jsk_control', 'jsk_control', 'jsk_control'),
    TestObject('/home/ataylor/jsk/jsk_planning', 'jski_planning', 'jsk_planning'),
    TestObject('/home/ataylor/jsk/jsk_roseus', 'jsk_roseus', 'jsk_roseus'),
    TestObject('/home/ataylor/jsk/jsk_travis', 'jsk_travis', 'jsk_travis'),
    TestObject('/home/ataylor/ros_ws/kobuki/', 'kobuki', 'kobuki'),
    TestObject('/home/ataylor/ros_ws/kobuki_soft/', 'kobuki_soft', 'kobuki_soft'),
    TestObject('/home/ataylor/ros_ws/mavros/', 'mavros', 'Mav Ros'),
    TestObject('/home/ataylor/ros_ws/maxwell/', 'maxwell', 'Maxwell'),
    TestObject('/home/ataylor/ros_ws/motoman/', 'motoman', 'Motoman'),
    TestObject('/home/ataylor/ros_ws/nao/', 'nao', 'NAO Ros'),
    TestObject('/home/ataylor/ros_ws/nao_robot/', 'nao_robot', 'NAO Robot repo'),
    TestObject('/home/ataylor/ros_ws/nao_extras/', 'nao_extras', 'NAO Extras'),
    TestObject('/home/ataylor/ros_ws/nao_interaction/', 'nao_interaction', 'NAO interaction'),
    TestObject('/home/ataylor/ros_ws/naopi_bridge/', 'naopi_bridge', 'Naopi bridge'),
    TestObject('/home/ataylor/ros_ws/nao_camera/', 'nao_camera', 'nao camera'),
    TestObject('/home/ataylor/ros_ws/nao_virtual/', 'nao_virtual', 'nao virtual'),
    TestObject('/home/ataylor/ros_ws/nao_viz/', 'nao_viz', 'nao viz'),
    TestObject('/home/ataylor/ros_ws/nao_sensors/', 'nao_sensors', 'nao sensors'),
    TestObject('/home/ataylor/ros_ws/nav2_platform/', 'nav2_platform', 'Nav2 platform'),
    TestObject('/home/ataylor/ros_ws/neo/', 'neo', 'neo robot'),
    TestObject('/home/ataylor/ros_ws/next_stage/', 'next_stage', 'next stage'),
    TestObject('/home/ataylor/ros_ws/novatel_spann/', 'novatel_spann', 'novatel_spann'),
    TestObject('/home/ataylor/ros_ws/p2os/', 'p2os', 'p2 os robot'),
    TestObject('/home/ataylor/ros_ws/p3/', 'p3', 'robot rescue'),
    TestObject('/home/ataylor/ros_ws/people/', 'people', 'People tracking ros'),
    TestObject('/home/ataylor/ros_ws/pepper/', 'pepper', 'Pepper robot for stuff'),
    TestObject('/home/ataylor/ros_ws/rail_segmentaiton','rail_segmentation','rail_segmentation'),
    TestObject('/home/ataylor/ros_ws/rail_ceiling','rail_ceiling','rail ceiling'),
    TestObject('/home/ataylor/ros_ws/ric','ric','robitician ric'),
    TestObject('/home/ataylor/ros_ws/segbot','segbot','segbot'),
    TestObject('/home/ataylor/ros_ws/segbot_apps','segbot_apps','segbot apps'),
    TestObject('/home/ataylor/ros_ws/turtlebot_apps','turtlebot_apps','turtlebot apps'),
    TestObject('/home/ataylor/ros_ws/turtlebot_interactions','turtlebot_interactions','turtlebot interactions'),
    TestObject('/home/ataylor/ros_ws/turtlebot','turtlebot','turtlebot'),
    TestObject('/home/ataylor/ros_ws/turtlebot_create','turtlebot_crate','turtlebot create'),
    TestObject('/home/ataylor/ros_ws/turtlebot_arm','turtlebot_arm','turtlebot arm'),
    TestObject('/home/ataylor/ros_ws/universial_robot','universial_robot','ros universial robot '),
    TestObject('/home/ataylor/ros_ws/yujin_ocs','yugin_ocs','ocs library'),
    TestObject('/home/ataylor/ros_ws/sr_utils','sr_utils','sr utils'),
    TestObject('/home/ataylor/ros_ws/sr_manipulation','sr_manipulation','sr manipulation'),
    TestObject('/home/ataylor/ros_ws/sr_demo','sr_demo','sr demo'),
    TestObject('/home/ataylor/pr2/app_manager', 'app_manager', 'app_manager'),
    TestObject('/home/ataylor/pr2/arm_navigation_msgs', 'arm_nav', 'arm_nav'),
    TestObject('/home/ataylor/pr2/p2_hack_the_future', 'pr2_hack_the_future', 'pr futre'),
    TestObject('/home/ataylor/pr2/pr2_apps', 'pr2_apps', 'pre2 apps'),
    TestObject('/home/ataylor/pr2/pr2_colibraiton','pr2_colibraiton','pr2_colibraiton'),
    TestObject('/home/ataylor/pr2/pr2_common','pr2_common','/pr2_common'),
    TestObject('/home/ataylor/pr2/pr2_common_actions','pr2_common_actions','pr2_common_actions'),
    TestObject('/home/ataylor/pr2/pr2_delivery','pr2_delivery','pr2_delivery'),
    TestObject('/home/ataylor/pr2/pr2_developer_key','pr2_developer_key','pr2_developer_key'),
    TestObject('/home/ataylor/pr2/pr2_doors','pr2_doors','pr2_doors'),
    TestObject('/home/ataylor/pr2/pr2_kinematics','pr2_kinematics','pr2_kinematics'),
    TestObject('/home/ataylor/pr2/pr2_mechanism_msgs','pr2_mechanism_msgs','pr2_mechanism_msgs'),
    TestObject('/home/ataylor/pr2/pr2_navigation','pr2_navigation','pr2_navigation'),
    TestObject('/home/ataylor/pr2/pr2_navigation_apps','pr2_navigation_apps','pr2_navigation_apps'),
    TestObject('/home/ataylor/pr2/pr2_pbd','pr2_pbd','pr2_pbd'),
    TestObject('/home/ataylor/pr2/pr2_precise_trajectory','pr2_precise_trajectory','pr2_precise_trajectory'),
    TestObject('/home/ataylor/pr2/pr2_self_test','pr2_self_test','pr2_self_test'),
    TestObject('/home/ataylor/pr2/pr2_sheild_telop','pr2_sheild_telop','pr2_sheild_telop'),
    TestObject('/home/ataylor/pr2/pr2_surrogate','pr2_surrogate','pr2_surrogate'),
    TestObject('/home/ataylor/pr2/rqt_pr2_dashboard','rqt_pr2_dashboard','rqt_pr2_dashboard'),
    TestObject('/home/ataylor/pr2/willow_maps','willow_maps','willow_maps'),


    #fail first time
    TestObject('/home/ataylor/care_o_bot/schunk_modular_robotics', 'schunk_modular_robotics', 'shcunk modular'),
    TestObject('/home/ataylor/care_o_bot/cob_driver', 'cob_driver', 'cob_driver'),
    TestObject('/home/ataylor/care_o_bot/cob_people_perception', 'cob_people_preception', 'careobot people preception'),
    TestObject('/home/ataylor/care_o_bot/cob_object_preception', 'cob_object_perception', 'careobot object perception'),
    TestObject('/home/ataylor/care_o_bot/ipa_canopen', 'ipa_canopen', 'IPA Canopen'),
    TestObject('/home/ataylor/bwi_common/', 'bwi_common', 'Another bwi thing..'),
    TestObject('/home/ataylor/ros_ws/carl/', 'carl', 'Carl Ros Stack'),
    TestObject('/home/ataylor/ros_ws/jaco_ros/', 'jaco_ros', 'Jaco Ros clone from texas'),
    TestObject('/home/ataylor/jsk/jsk_common/', 'jsk_common', 'jsk_common'),
    TestObject('/home/ataylor/ros_ws/orcos/', 'ocros', 'Ocoros kinematics'),
    TestObject('/home/ataylor/ros_ws/segbot_arm','segbot_arm','segbot arm'),
    TestObject('/home/ataylor/ros_ws/hector_quadrotor/', 'hector_quad', 'Hector quad'),
    TestObject('/home/ataylor/ros_ws/hector_localization/', 'hector_localization', 'Hector Localization'),
    TestObject('/home/ataylor/ros_ws/sr_ros_interface','sr_ros_interface','sr interface'),
    TestObject('/home/ataylor/ros_ws/sr_teleop','sr_teleop','sr teleop'),
    TestObject('/home/ataylor/ros_ws/sr_ronex','sr_ronex','sr ronex'),
    TestObject('/home/ataylor/pr2/kinematic_msgs', 'kin_msg', 'kn_msgs'),
    TestObject('/home/ataylor/pr2/pr2_dashboard','pr2_dashboard','pr2_dashboard'),
    TestObject('/home/ataylor/pr2/pr2_gripper_sensor','pr2_gripper_sensor','pr2_gripper_sensor'),
    TestObject('/home/ataylor/pr2/pr2_object_manipulation','pr2_object_manipulation','pr2_object_manipulation'),
    TestObject('/home/ataylor/pr2/pr2_plugs','pr2_plugs','pr2_plugs'),
    TestObject('/home/ataylor/pr2/pr2_power_drivers','pr2/pr2_power_drivers','pr2/pr2_power_drivers'),
    TestObject('/home/ataylor/pr2/pr2_sith','pr2_sith','pr2_sith'),

    # # Timeouts
    TestObject('/home/ataylor/pr2/pr2_robot','pr2_robot','pr2_robot'),
    TestObject('/home/ataylor/pr2/pr2_mechanism','pr2_mechanism','pr2_mechanism'),
    TestObject('/home/ataylor/ros_ws/moveit', 'moveit', 'Move it stack'),
    TestObject('/home/ataylor/pr2/pr2_ethercat_drivers','pr2_ethercat_drivers','pr2_ethercat_drivers'),
    TestObject('/home/ataylor/ros_ws/roseus','roseus','roseus_robot'),
    TestObject('/home/ataylor/ros_ws/ardrone/', 'ardrone', 'ardrone autonomy'),
    TestObject('/home/ataylor/pr2/pr2_controllers','pr2_controllers','pr2_controllers'),
    ]



class Alarm(Exception):
    pass

def alarm_handler(signum, frame):
    raise Alarm 


def mail_results(test_name, times, result, output, subject=None):
    message = """
    {:s}: {:s}
    \n
    Times: {:f}\t{:f}\t{:f}
    {:s}
    """.format(test_name, result, times[0], times[1], times[1] - times[0], output)


    msg = MIMEText(message)
    if subject is None:
        msg['Subject'] = '{:s}:{:s}'.format(test_name, result)
    else:
        msg['Subject'] = subject
    msg['From'] = FROM_EMAIL
    msg['To'] = TO_EMAIL
    try:
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.ehlo()
        session.starttls()
        session.ehlo()
        ## ADD EMAIL PASSWORD!!!!
        session.login(FROM_EMAIL, PASSWORD)
        session.sendmail(FROM_EMAIL, [TO_EMAIL], msg.as_string())
        session.quit()
    except Exception:
        print "Error: unable to send email"
        print msg.as_string()


def move_cmakes(directory, clean=True):
    for dir_path, names, files in os.walk(directory):
        # if we have a cmakelists and package.xml than we are in a thing to modify
        if "CMakeLists.txt_mod" in files and "CMakeLists.txt_backup" in files:
            cmn = dir_path + '/CMakeLists.txt'
            mod = dir_path + '/CMakeLists.txt_mod'
            back = dir_path + '/CMakeLists.txt_backup'
            if clean:
                shutil.copy(back, cmn)
            else:
                shutil.copy(mod, cmn)


def remove_build_devl(source_dir):
    build = source_dir + "devel/"
    devel = source_dir + "build/"
    if os.path.exists(build):
        shutil.rmtree(build)
    if os.path.exists(devel):
        shutil.rmtree(devel)


def run_test(test_obj):

    max_run_time = 60 * 10
    # clear clang results
    files = [CLANG_RESULTS_DIR + x for x in os.listdir(CLANG_RESULTS_DIR)]
    for i in files:
        os.remove(i)

    # setup clean
    move_cmakes(test_obj.source_dir + 'src/')
    remove_build_devl(test_obj.source_dir)


    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(max_run_time)
    try:
        # # build clean
        os.chdir(test_obj.source_dir)
        st = time.time()
        cmd = ["catkin_make", "-C", test_obj.source_dir,
               "-DCMAKE_CXX_COMPILER=/home/ataylor/llvm_src/gitllvm/Release+Asserts/bin/clang++"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()
        et = time.time()
        t_clean = et - st
        if exit_code != 0:
            mail_results(test_obj.test_name, [t_clean, 0],  "Failed: Clean", err, "FAILED: {:s}".format(test_obj.test_name))
            return 'fail' 
    except Alarm:
        mail_results(test_obj.test_name, [0, 0],  "Failed: Timeout Clean", "TIMEOUT ON CLEAN BUILD", "FAILED: {:s} TIMEOUT".format(test_obj.test_name))
        return 'timeout'


    # setup modified
    move_cmakes(test_obj.source_dir + 'src/', False)
    remove_build_devl(test_obj.source_dir)

    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(max_run_time * 2)
    try:
        cmd = ["catkin_make", "-C", test_obj.source_dir,
               "-DCMAKE_CXX_COMPILER=/home/ataylor/llvm_src/gitllvm/Release+Asserts/bin/clang++"]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = process.communicate()
        exit_code = process.wait()
        et = time.time()
        t_analyze = et - st
        if exit_code != 0:
            mail_results(test_obj.test_name, [t_clean, t_analyze], "Failed: Modified", err, "FAILED: {:s}".format(test_obj.test_name))
            return 'fail' 
    except Alarm:
        mail_results(test_obj.test_name, [0, 0],  "Failed: Timeout Modified", "TIMEOUT ON MOIFIED BUILD", "FAILED: {:s} TIMEOUT".format(test_obj.test_name))
        return 'timeout'



    # do python analysis
    cmd = ['python', '/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/tools/analyze_python_files.py',
           test_obj.source_dir + '/src/', '-o' '/home/ataylor/clang_results/']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        mail_results(test_obj.test_name, [t_clean, t_analyze], "Python Analysis Failed", err, "FAILED: {:s}".format(test_obj.test_name))
        return 'fail'

    # clean results directory if it has stuff in it
    results_dir = RESULTS_DIR + test_obj.result_dir 
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    files = [results_dir + x for x in os.listdir(results_dir)]
    for i in files:
        os.remove(i)

    # move everything
    files = [x for x in os.listdir(CLANG_RESULTS_DIR)]
    for i in files:
        shutil.copy(CLANG_RESULTS_DIR + i, results_dir + i)

    # run threshold analysis script...
    cmd = ['python', '/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/tools/threshold_report.py', results_dir]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        mail_results(test_obj.test_name,[t_clean, t_analyze], "failed on Final Analysis", err, 'FAILED {:s}'.format(test_obj.test_name))
        return 'fail' 

    # email results
    val = output.split('\n')[4].split()[-1]
    mail_results(test_obj.test_name,[t_clean, t_analyze], "Done with test: " + test_obj.test_name,
            output, "DONE: {:s} -> {:s} Thresholds".format(test_obj.test_name, val))
    val = output.split()[-1]
    return '{:f},{:f},{:s}'.format(t_clean, t_analyze - t_clean, val)


def main():

    passed = []
    failed = []
    results = []
    for i in tests:
        try:
            val = run_test(i)
            if val != 'fail' and val != 'timeout':
                passed.append(i)
                res_str = '{:s},{:s}'.format(i.test_name, val)
                results.append(res_str)
                print res_str
            else:
                failed.append(i)
                print i.test_name,  val

        except KeyboardInterrupt:
            print 'Aborted tests'
            break
        except Exception as e:
            print e
            failed.append(i)

    if len(failed) > 0:
        print "\nThese tests failed:"
        for i in failed:
            print i.test_name
    with open('/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/tools/fails.txt', 'w') as openf:
        for i in failed:
            openf.write(i.test_name + '\n')

    with open('/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/tools/results.csv', 'w') as openf:
        openf.write('\n')
        for i in results:
            openf.write(i + '\n')
        openf.close()


def display_src_information():
    results = []
    for test in tests:
        cmd = ['cloc', test.source_dir + 'src/']
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        (output, err) = process.communicate()
        process.wait()
        pyl = 0
        cppl = 0
        chl = 0
        pyf = 0
        cppf = 0
        chf = 0
        for line in output.split('\n'):
            if line.startswith('Python'):
                _, pyf, _, _,pyl = line.split()
            if line.startswith('C++'):
                _, cppf, _, _, cppl = line.split()
            if line.startswith('C/C++ Header'):
                _, _, chf, _, _, chl = line.split()
        pyl = int(pyl)
        cppl = int(cppl)
        chl = int(chl)
        chf = int(chf)
        cppf = int(cppf)
        pyf = int(pyf)
        val = '{:s},{:d},{:d},{:d},{:d},{:d},{:d}'.format(test.test_name, cppf, cppl, chf, chl, pyf, pyl)
        results.append(val)
        print val
    with open('/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/tools/system_info.csv', 'w') as outf:
        for res in results:
            outf.write(res + '\n')


    

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Test runner and information")
    parser.add_argument('--src-info', help="create table of source code information", action="store_true")
    args = parser.parse_args()

    if args.src_info:
        display_src_information()
    else:
        if PASSWORD == '':
            print "FILL IN PASSWORD"
        else:
            main()

