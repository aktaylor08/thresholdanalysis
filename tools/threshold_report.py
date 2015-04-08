__author__ = 'ataylor'

#!/usr/bin/env python

import argparse
import glob
import json

def read_data(directory):
    data = {}
    for json_file in glob.glob(directory + '*.json'):
        with open(json_file) as openf:
            json_in = json.load(openf)
            for thresh_key in json_in:
                data[thresh_key] = json_in[thresh_key]
    return data


if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(
        description=("Read in information and report information about"
                     "thresholds found during compilation in a ros package"))
    parser.add_argument('directory', help="Directory to process")
    args = parser.parse_args()
    directory = args.directory
    if directory[-1] != '/':
        directory += '/'
    data = read_data(directory)

    cpp= 0
    py = 0
    param = 0
    const = 0
    numerical = 0
    param_counts = {}
    file_counts = {}

    maxd = 0
    mind = 9999
    dtotal = 0

    for d in data.itervalues():
        # file type count
        if d['file'].endswith('.py'):
            py += 1
        if d['file'].endswith('.cpp'):
            cpp += 1

        # Type count
        if d['type'].lower() == 'parameter':
            param += 1
        if d['type'].lower() == 'code':
            const += 1
        if d['type'].lower() == 'numerical':
            numerical += 1

        # use count
        if d['type'].lower() == 'parameter':
            temp = param_counts.get(d['source'], 0) + 1
            param_counts[d['source']] = temp

        # file count
        temp = file_counts.get(d['file'], 0) + 1
        file_counts[d['file']] = temp

        # distance
        dist = d['distance']
        if dist > maxd:
            maxd = dist
        if dist < maxd:
            mind = dist
        dtotal += dist
    print d


    param_counts = sorted([(int(y),x) for x,y in param_counts.iteritems()], key=lambda asdf: asdf[0], reverse=True)
    file_counts= sorted([(int(y),x) for x,y in file_counts.iteritems()], key=lambda asdf: asdf[0], reverse=True)

    print '\n\n'
    print 'Report for directory:\t\t\t{:s}'.format(directory)
    print 'Number of threshold branches:\t{:d}'.format(len(data))
    print '\tIn python files\t\t\t\t{:d}'.format(py)
    print '\tIn c++ files\t\t\t\t{:d}'.format(cpp)
    print '\n'
    print 'From Parameters:\t\t\t\t{:d}'.format(param)
    print 'ided as  constant:\t\t\t\t{:d}'.format(const)
    print 'Numerical Comparisons:\t\t\t{:d}'.format(numerical)
    print '\n'
    print "Max Separation:\t\t\t{:d}".format(maxd)
    print "Min Separation:\t\t\t{:d}".format(mind)
    print "Average Separation:\t\t\t{:f}".format(float(dtotal) / len(data))
    print '\n'
    print "Files with Constants:"
    print "\tCount\tFile Name"
    for i in file_counts:
        print "\t{:d}\t\t{:s}".format(i[0], i[1])
    print '\n'
    print "Parameters Used:"
    print "\tCount\tParameter Name"
    for i in param_counts:
        print '\t{:d}\t\t{:s}'.format(i[0], i[1])


