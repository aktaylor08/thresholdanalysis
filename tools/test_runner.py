#!/usr/bin/python

import subprocess
import smtplib
import shutil
import os
import time
from email.mime.text import MIMEText


CLANG_RESULTS_DIR = '/home/ataylor/clang_results/'
RESULTS_DIR = '/home/ataylor/results/'



class TestObject:
    source_dir = ''
    test_name = ''
    result_dir = ''

    def __init__(self):
        pass


def mail_results(test_name, result, output):
    message = """
    {:s} {:s}
    \n
    {:s}
    """.format(test_name, result, output)

    fadd = 'rostestmailer@gmail.com'
    toadd = 'aktaylor08@gmail.com'

    msg = MIMEText(message)
    msg['Subject'] = '{:s}:{:s}'.format(test_name, result)
    msg['From'] = fadd
    msg['To'] = toadd
    try:
        session = smtplib.SMTP('smtp.gmail.com', 587)
        session.ehlo()
        session.starttls()
        session.ehlo()
        ## ADD EMAIL PASSWORD!!!!
        session.login("rostestmailer@gmail.com", "")
        session.sendmail(fadd, [toadd], msg.as_string())
        session.quit()
    except smtplib.SMTPException:
        print "Error: unable to send email"


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
    build = source_dir + "/devel"
    devel = source_dir + "/build"
    if os.path.exists(build):
        shutil.rmtree(build)
    if os.path.exists(devel):
        shutil.rmtree(devel)


def run_test(test_obj):
    print "running: ", test_obj.test_name
    # clear clang results
    files = [CLANG_RESULTS_DIR + x for x in os.listdir(CLANG_RESULTS_DIR)]
    for i in files:
        os.remove(i)

    # setup clean
    move_cmakes(test_obj.source_dir + 'src/')
    remove_build_devl(test_obj.source_dir)


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
        mail_results(test_obj.test_name, "failed on clean build", err + "Time: " + str(t_clean))
        return

    # setup modified
    move_cmakes(test_obj.source_dir + 'src/', False)
    remove_build_devl(test_obj.source_dir)
    cmd = ["catkin_make", "-C", test_obj.source_dir,
           "-DCMAKE_CXX_COMPILER=/home/ataylor/llvm_src/gitllvm/Release+Asserts/bin/clang++"]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    et = time.time()
    t_analyze = et - st
    if exit_code != 0:
        mail_results(test_obj.test_name, "failed on modified build", err + str(t_analyze))
        return



    # do python analysis
    cmd = ['python', '/home/ataylor/ros_ws/thresholds/src/thresholdanalysis/tools/analyze_python_files.py',
           test_obj.source_dir + '/src/', '-o' '/home/ataylor/clang_results/']
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (output, err) = process.communicate()
    exit_code = process.wait()
    if exit_code != 0:
        mail_results(test_obj.test_name, "failed on python threshold analysis", err + str(t_analyze))

    # clean results directory if it has stuff in it
    results_dir = RESULTS_DIR + test_obj.result_dir + '/'
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
        mail_results(test_obj.test_name, "failed on Final Analysis", err)
        return

    # email results
    mail_results(test_obj.test_name, "Done with test: " + test_obj.test_name,
                 "times:\n" + str(t_clean) + "\n" + str(t_analyze) + "\n\n" + output)


def main():
    tests = []
    # t0 = TestObject()
    # t0.source_dir = '/home/ataylor/asctec_ws/'
    # t0.result_dir = 'asctec'
    # t0.test_name = 'asctec base'
    # tests.append(t0)

    # t1 = TestObject()
    # t1.source_dir = '/home/ataylor/ros_ws/water/'
    # t1.result_dir = 'water'
    # t1.test_name = 'water sampler'
    # tests.append(t1)

    # t2 = TestObject()
    # t2.source_dir = '/home/ataylor/ros_ws/navigation/'
    # t2.result_dir = 'navigation'
    # t2.test_name = 'navigation stack'
    # tests.append(t2)

    # t3 = TestObject()
    # t3.source_dir = '/home/ataylor/ros_ws/crop_surveying/'
    # t3.result_dir = 'crop_surveying'
    # t3.test_name = 'crop surveying'
    # tests.append(t3)

    # t4 = TestObject()
    # t4.source_dir = '/home/ataylor/ros_ws/ardrone/'
    # t4.result_dir = 'ardrone'
    # t4.test_name = 'ardrone autonomy'
    # tests.append(t4)

    # t = TestObject()
    # t.source_dir = '/home/ataylor/ros_ws/pass_tests/'
    # t.result_dir = 'pass_test'
    # t.test_name = 'pass unit tests'
    # tests.append(t)

    # t5 = TestObject()
    # t5.source_dir = '/home/ataylor/ros_ws/baxter/'
    # t5.result_dir = 'baxter'
    # t5.test_name = 'Baxter Robot'
    # tests.append(t5)

    t6 = TestObject()
    t6.source_dir = '/home/ataylor/controllers/realtime_tools'
    t6.result_dir = 'realtime_tools'
    t6.test_name = 'Realtime tools'
    tests.append(t6)

    t7 = TestObject()
    t7.source_dir = '/home/ataylor/controllers/control_toolbox'
    t7.result_dir = 'control_toolbox'
    t7.test_name = 'Control Toolbox'
    tests.append(t7)

    t8 = TestObject()
    t8.source_dir = '/home/ataylor/controllers/ros_control'
    t8.result_dir = 'ros_control'
    t8.test_name = 'Ros Control'
    tests.append(t8)

    t9 = TestObject()
    t9.source_dir = '/home/ataylor/controllers/ros_controllers'
    t9.result_dir = 'ros_controllers'
    t9.test_name = 'Ros Controllers'
    tests.append(t9)


    for i in tests:
        run_test(i)


if __name__ == '__main__':
    main()

