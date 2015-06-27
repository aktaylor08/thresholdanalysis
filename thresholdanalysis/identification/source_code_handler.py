import sys

from thresholdanalysis.identification.backward_analysis import get_outside_calls

if __name__ == '__main__':
    fname = sys.argv[1]
    get_outside_calls(file_name=fname)
