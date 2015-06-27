#!/usr/bin/env python

import argparse

import glob
import json
import os


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
                     "thresholds found during compilation in a runtime package"))
    parser.add_argument('-d', '--directory', help="Directory to process", required=False)
    args = parser.parse_args()
    if args.directory is not None:
        directory = args.directory
    else:
        directory = os.getcwd()

    if directory[-1] != '/':
        directory += '/'
    data = read_data(directory)

    cpp = 0
    py = 0
    param = 0
    const = 0
    numerical = 0
    param_counts = {}
    param_files = {}
    file_counts = {}

    maxd = 0
    mind = 9999
    dtotal = 0
    count = 0

    file_params = {}
    param_to_files = {}

    for d in data.itervalues():
        # file type count
        if d['file'].endswith('.py'):
            py += 1
        if d['file'].endswith('.cpp') or d['file'].endswith('.h'):
            cpp += 1

        fset = file_params.get(d['file'], set())
        fset.add(d['source'])
        file_params[d['file']] = fset


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

            temp = param_to_files.get(d['source'], set())
            temp.add(d['file'])
            param_to_files[d['source']] = temp

        # file count
        temp = file_counts.get(d['file'], 0) + 1
        file_counts[d['file']] = temp

        # distance
        dist = d['distance']
        if dist > maxd:
            maxd = dist
        if dist < mind:
            mind = dist
        dtotal += dist
        count += 1

    param_counts = sorted([[int(y), x] for x, y in param_counts.iteritems()], key=lambda asdf: asdf[0], reverse=True)
    file_counts = sorted([[int(y), x] for x, y in file_counts.iteritems()], key=lambda asdf: asdf[0], reverse=True)
    # just tack on the other results..hack for now.
    for i in file_counts:
        i.append(len(file_params[i[1]]))

    # Total Unique
    total_unique = sum(i[2] for i in file_counts)
    for p in param_counts:
        p.append(len(param_to_files[p[1]]))
    unique_params = sum([p[2] for p in param_counts])

    print '\n\n'
    print '{:35s}\t{:s}'.format("Report for directory:", directory)
    print '{:35s}\t{:d}'.format("Number of threshold branches:", len(data))
    print '  {:33s}\t{:d}'.format("In python files:", py)
    print '  {:33s}\t{:d}'.format("In cpp files:", cpp)
    print '\n'
    print '{:35s}\t{:d}'.format("From Parameters:", param)
    print '{:35s}\t{:d}'.format("identified constant:", const)
    print '{:35s}\t{:d}'.format("Numerical Comparison:", numerical)
    print '\n'
    print '{:35s}\t{:d}'.format('Unique Thresholds:', total_unique)
    print '{:35s}\t{:d}'.format('Unique Parameters:', unique_params)
    print '\n'
    print '{:35s}\t{:d}'.format("Max Separation:", maxd)
    print '{:35s}\t{:d}'.format("Min Separation:", mind)
    if count != 0:
        print '{:35s}\t{:f}'.format("Average Separation:", float(dtotal) / count)
    else:
        print '{:35s}\t{:f}'.format("Average Separation:", 0)
    print '\n'
    print '{:35s}\t{:d}'.format("Number of files with thresholds", len(file_counts))
    print "Files with thresholds:"
    print "{:10s}{:10s}\t{:20s}".format("Total", "Unique", "File Name")
    for i in file_counts:
        print "{:<10d}{:<10d}\t{:20s}".format(i[0], i[2], i[1])
    print '\n'
    print "Parameters Used:"
    print "{:10s}{:10s}\t{:20s}".format("Uses", "Files", "Paramter")
    for i in param_counts:
        print "{:<10d}{:<10d}\t{:20s}".format(i[0], i[2], i[1])
    print '{:d},{:d},{:d},{:d},{:d},{:d},{:d}'.format(len(data), cpp, py, total_unique, param, unique_params,
                                                      len(file_counts))

    param = [x for x in data.itervalues() if x['type'] == 'Parameter']
    num = [x for x in data.itervalues() if x['type'] == 'Numerical']
    code = [x for x in data.itervalues() if x['type'] == 'code']
    param = sorted(param, key=lambda x: x['distance'])
    num = sorted(num, key=lambda x: x['distance'])
    code = sorted(code, key=lambda x: x['distance'])
    file_grouped = sorted(param, key=lambda x: x['file'])

    print '\n'
    print '{:15s}{:10s}{:30s}{:10s}{:s}'.format('Type', "Distance", "Source",  "Line No.", "File",)
    for i in param:
        print '{:15s}{:<10d}{:30s}{:<10d}{:s}'.format(
            i['type'],
            i['distance'],
            i['source'],
            i['lineno'],
            i['file'],
        )
    print '-----\n'
    for i in num:
        print '{:15s}{:<10d}{:30s}{:<10d}{:s}'.format(
            i['type'],
            i['distance'],
            i['source'],
            i['lineno'],
            i['file'],
        )
    print '-----\n'
    for i in code:
        print '{:15s}{:<10d}{:30s}{:<10d}{:s}'.format(
            i['type'],
            i['distance'],
            i['source'],
            i['lineno'],
            i['file'],
        )
    print '-----\n'
    for i in file_grouped:
        print '{:15s}{:<10d}{:30s}{:<10d}{:s}'.format(
            i['type'],
            i['distance'],
            i['source'],
            i['lineno'],
            i['file'],
        )



