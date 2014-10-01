#!/usr/bin/env python
# encoding: utf-8


def todo(i):
    for val in range(i):
        if val > 5:
            break
        else:
            continue
        print i * 5
    else:
        print 'no greater than 5'
    print 'done doing'
    print 'i am a car VROOM'

todo(3)


# def to_do():
#     i = 10
#     while i > 0:
#         i = i - 1
#         j = i * 10
#         for k in range(j):
#             print j
#         print i
#         if i < 3:
#             print 'hello there'
#             continue
#             print 'NOT HERE'
#         elif i > 4:
#             print j
#         else:
#             print 'BY'
#             return
        
#     if 4 < 5:
#         print k
