import sys
from backward_analysis import build_import_list


if __name__ == '__main__':
    fname = sys.argv[1]
    build_import_list(file_name=fname)
