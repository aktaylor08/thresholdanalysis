#!/usr/bin/env python

import argparse
from collections import defaultdict

import os
import sys
import re
import shutil
from wx._windows_ import DirDialog_GetPath

INSTRUMENT_FILE_LOCATION = "/home/ataylor/llvm/llvm/lib/Transforms/RosThresholds/instruments/instrument.cpp"
# INSTRUMENT_FILE_LOCATION = "/Users/ataylor/Research/llvm_src/plugin_llvm/llvm/lib/Transforms/RosThresholds/instruments/instrument.cpp"


def hack_inside_group(asdgfasd):
    new_list = []
    cur_val = ""
    inside_string = False
    for i in asdgfasd:
        if i.strip().lstrip().startswith('"') and i.strip().lstrip().endswith('"'):
            new_list.append(i)
        elif i.strip().lstrip().startswith('"'):
            if inside_string:
                raise Exception("Error parsing CMAKE FILE should not be inside string")
            inside_string = True
            cur_val = i
        elif i.strip().lstrip().endswith('"'):
            if inside_string:
                cur_val += " " + i
                new_list .append(cur_val)
                inside_string = False
            else:
                print asdgfasd
                raise Exception("Error parsing CMAKE FILE should not be not inside string")
        else:
            if inside_string:
                cur_val += " " + i
            else:
                new_list.append(i)
    return new_list


def get_groups(lines):
    """Get the groups of settings that are defined in the CMakeLists.txt file"""
    arguments = []
    cur_cmd = ''
    pcount = 0
    inside_args = []
    # did we match everything?
    for i in lines:
        # regexes and mapping
        match_all = re.search("(.*?)\s*\((.*)\)", i)
        match_start = re.search("(.*?)\s*\((.*)", i)
        match_end = re.search("(.*)\s*\)", i)

        # try and match between the parenthesis
        if match_all:
            inside = hack_inside_group(match_all.groups()[1].split())
            arguments.append((match_all.groups()[0], inside))

        # start of match
        elif match_start is not None:
            pcount += 1
            if pcount == 1:
                cur_cmd = match_start.groups()[0]
                inside_args = [x for x in match_start.groups()[1].split()]
                inside_args = hack_inside_group(inside_args)
            else:
                print "error too many parenthesis"
                assert False

        # close up shop
        elif match_end is not None:
            pcount -= 1
            if pcount == 0:

                for arg in hack_inside_group(match_end.groups()[0].split()):
                    inside_args.append(arg)
                arguments.append((cur_cmd, inside_args))
            else:
                print "error on close paren"
                assert False
        else:
            for arg in hack_inside_group(i.split()):
                inside_args.append(arg)
    return arguments


def do_work(directory_start):
    errs = []
    for dir_path, names, files in os.walk(directory_start):
        # if we have a cmakelists and package.xml than we are in a thing to modify
        if "CMakeLists.txt" in files and "package.xml" in files:
            print "Working on: {:s}".format(dir_path)
            cmake = dir_path + '/CMakeLists.txt'
            to_parse = []
            changed = False
            results = []
            with open(cmake) as cfile:
                lines = cfile.read().split("\n")
                # get rid of comments
                for i in lines:
                    cidx = i.find("#")
                    if cidx < 0:
                        to_parse.append(i)
                    elif cidx == 0:
                        pass
                    else:
                        to_parse.append(i[:cidx])
                cmake_args = get_groups(to_parse)
                add_targets = []
                add_instrument_code = []
                to_remove = []
                copy_locs = set()

                # loop through the received tripples to get the desired information
                in_if = False
                cxx_flags = defaultdict(str)
                global_cxx_flags = ''
                global_link_flags = ''
                link_flags = defaultdict(str)
                for i in cmake_args:
                    try:
                        if i[0] == 'if' and i[1][2] == '"indigo"':
                            in_if = True
                        if i[0] == 'else':
                            in_if = False
                        if i[0] =='endif':
                            in_if = False
                    except:
                        print 'error on: ',  i

                    # removed library call temporarily
                    if i[0] == "add_executable" or i[0] == "add_library" or i[0] == "add_install":
                        add_targets.append(i[1][0])
                        rest = i[1][1:]
                        if len(rest) == 1 and rest[0].startswith('${'):
                            to_find = rest[0][2:-1]
                            for x in cmake_args:
                                if x[0] == 'set':
                                    if x[1][0] == to_find:
                                        add_instrument_code.append(x)
                        else:
                            add_instrument_code.append(i)

                    # Check for set` CMAKE_CXX_FLAGS to save those off for later...
                    # also need to check out
                    if i[0].lstrip().strip() == 'set' and i[1][0].lstrip().rstrip() == 'CMAKE_CXX_FLAGS':
                        to_remove.append(i)
                        if in_if:
                            for flag in i[1][1:]:
                                new_flag = flag.replace('${CMAKE_CXX_FLAGS}', '')
                                global_cxx_flags += new_flag
                        else:
                            print 'hey'
                    save_next_compile = False
                    save_next_link = False
                    if i[0].lstrip().strip() == 'set_target_properties':
                        trgt = i[1][0]
                        for j in i[1]:
                            if save_next_compile:
                                new_flag = j.replace('${CMAKE_CXX_FLAGS}', '')
                                cxx_flags[trgt] += new_flag
                                save_next_compile = False
                            if save_next_link:
                                new_flag = j.replace('${CMAKE_LINK_FLAGS}', '')
                                link_flags[trgt] += new_flag
                                save_next_link = False

                            if j == 'COMPILE_FLAGS':
                                save_next_compile = True
                            if j == 'LINK_FLAGS':
                                save_next_link = True




                # strip off quots
                for trgt in cxx_flags.iterkeys():
                    cxx_flags[trgt] = cxx_flags[trgt].replace('"', '')
                    cxx_flags[trgt] += global_cxx_flags
                for trgt in link_flags.iterkeys():
                    link_flags[trgt] = link_flags[trgt].replace('"', '')
                    link_flags[trgt] += link_flags[trgt].replace('"', '')
                # set the compile and link flags needed
                for trg in add_targets:
                    print "\tadding for target: ", trg
                    changed = True
                    cmake_args.append(('set_target_properties',
                                       [trg, 'PROPERTIES', 'COMPILE_FLAGS', '"-g -flto {:s}"'.format(cxx_flags[trg]), 'LINK_FLAGS', '"-flto {:s}"'.format(link_flags[trg])]))

                # append src/instrument.cpp to the file list for all of the executables
                for arg in cmake_args:
                    if arg in add_instrument_code:
                        # get the directory to copy it into...
                        src_dir, _ = os.path.split(arg[1][-1])
                        if src_dir != '':
                            inst_file = src_dir + '/instrument.cpp'
                        else:
                            inst_file = 'instrument.cpp'

                        if inst_file not in arg[1]:
                            arg[1].append(inst_file)
                            if dir_path != '':
                                copy_locs.add(dir_path + '/' + inst_file)
                            else:
                                copy_locs.add(inst_file)

            if changed:
                # cmake_args.insert(2, ('set', ["CMAKE_CXX_COMPILER", "clang++"]))
                # back up old CMakeLists.txt
                if not os.path.exists(dir_path + '/CMakeLists.txt_backup'): shutil.copy(dir_path + '/CMakeLists.txt', dir_path + '/CMakeLists.txt_backup')

                # write out the new file
                with open(dir_path + '/CMakeLists.txt', 'w') as outf:
                    for val in cmake_args:
                        outf.write(val[0] + '(\n')
                        for x in val[1]:
                            outf.write('\t{:s}\n'.format(x))
                        outf.write(')\n')

                with open(dir_path + '/CMakeLists.txt_mod', 'w') as outf:
                    for val in cmake_args:
                        outf.write(val[0] + '(\n')
                        for x in val[1]:
                            outf.write('\t{:s}\n'.format(x))
                        outf.write(')\n')

                # copy the file over now
                for destination in copy_locs:
                    try:
                        shutil.copy(INSTRUMENT_FILE_LOCATION, destination)
                    except:
                        errs.append("Problem copying instrument file to {:s}".format(destination))
    print '\n\n Errors:'
    for i in errs:
        print i



def restore(directory_start):
    for dir_path, names, files in os.walk(directory_start):
        # if we have a cmakelists and package.xml than we are in a thing to modify
        if "CMakeLists.txt_backup" in files:
            print("Restoring backup in directory: {:s}".format(dir_path))
            shutil.copy(dir_path + '/' + "CMakeLists.txt_backup", dir_path + '/' + "CMakeLists.txt")


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description="Parse all of the python files in the directories below the one passed")
    parser.add_argument('directory', help="Directory and below to process")
    parser.add_argument("--restore", help="Revert to the CMakeLists.txt_backup files that this method creates.",
            action='store_true')
    args = parser.parse_args()

    directory = args.directory
    if not os.path.exists(directory):
        print("Invalid directory!")
        sys.exit(-1)
    if args.restore:
        restore(directory)
    else:
        if not os.path.exists(INSTRUMENT_FILE_LOCATION):
            print("INSTRUMENTATION FILE DOES NOT EXIST WOAH")
            sys.exit(-1)/home/ataylor/ros_ws/thresholds/src/thresholdanalysis
        do_work(directory)

