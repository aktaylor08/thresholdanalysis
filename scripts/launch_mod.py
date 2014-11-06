#!/usr/bin/env python
# encoding: utf-8

import sys
import os
import rospkg
from collections import deque
import xml.etree.ElementTree as ET 

CONST_LINE = "<!--THIS FILE HAS BEEN MODIFIED TO ANALYZE THRESHOLDS. DO NOT MODIFY-->\n"

class XmlVisit(object):

    def __init__(self):
        self.depth = 0
        self.others = []
        self.rpkg = rospkg.RosPack()

    def visit_node(self, node):
        if node.tag == 'node':
            self.handle_node(node)
        if node.tag == 'include':
            self.handle_include(node)
        for child in node:
            self.depth +=1
            self.visit_node(child)
            self.depth -=1

    def handle_include(self, node):
        val = node.attrib['file']
        #handle this here
        start = val.find('$(find')
        if start >= 0:
            end = val.find(')', start)
            package = val[start + 7:end]
            pkg_path = self.rpkg.get_path(package)
            path = val[end+1:]
            self.others.append(pkg_path + path)
            point = path.rfind('.')
            new = val[:end+1] +path[:point] + '_thresh_mon' + path[point:]
            node.attrib['file'] = new


    def handle_node(self, xml_node):
        fname = xml_node.attrib['type']
        if fname.endswith('.py'):
            package = xml_node.attrib['pkg']
            path = self.rpkg.get_path(package)
            path = self.get_relative_path(path, fname)
            xml_node.attrib['pkg'] = 'thresholdanalysis'
            xml_node.attrib['type'] = 'backward_analysis.py'
            
            #does it have command line args? 
            if 'args' in xml_node.attrib:
                xml_node.attrib['args'] = path + ' ' + xml_node.attrib['args']
            else:
                xml_node.attrib['args'] = path 


    def get_relative_path(self, directory, fname):
        '''get the aboslute path to the node'''
        files = os.listdir(directory)
        to_examine = []
        for i in files:
            if not os.path.isdir(directory+'/'+i):
                if i == fname:
                    return directory + '/' + i 
            else:
                to_examine.append(i) 
        for f in to_examine:
            val = self.get_relative_path(directory +'/' +  f, fname)
            if val is not None:
                return val
        return None



if __name__ == '__main__':
    if len(sys.argv) > 1:
        queue = deque()
        visited = set()
        fname = sys.argv[1] 
        fname = os.path.abspath(fname)
        queue.append(fname)
        while len(queue) > 0:
            fname = queue.popleft() 
            #check if previously modified...
            with open(fname) as f:
                line =f.readline()
                if line == CONST_LINE:
                    print fname, 'File already modified!'
                    visited.add(fname)
                    continue
            print fname, 'modifying'
            tree = ET.parse(fname)
            root = tree.getroot() 
            visitor = XmlVisit()
            visitor.visit_node(root)
            base, ext = os.path.splitext(fname)
            out_name = base + '_thresh_mon' + ext
            vals =ET.tostringlist(root)
            visited.add(fname)
            for i in visitor.others:
                queue.append(i)


            with open(out_name, 'w') as fout:
                fout.write(CONST_LINE)
                fout.write('\n')
                for i in vals:
                    fout.write(i)

    else:
        print 'no file'



    


