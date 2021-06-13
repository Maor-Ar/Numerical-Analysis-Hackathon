"""
 * Authors: Maor Arnon (ID: 205974553) and Neriya Zudi (ID:207073545)
 * Emails: maorar1@ac.sce.ac.il    neriyazudi@Gmail.com
 * Department of Computer Engineering - Assignment 5 - Numeric Analytics
"""

from sympy import tan
from math import e
import sympy as sp
import math


class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[90m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    # Background colors:
    GREYBG = '\033[100m'
    REDBG = '\033[101m'
    GREENBG = '\033[102m'
    YELLOWBG = '\033[103m'
    BLUEBG = '\033[104m'
    PINKBG = '\033[105m'
    CYANBG = '\033[106m'

def Lagrange_interpolation(Table,pointToFindVal):  ##calculate by lagrange
    yp=0

    for i in range(len(Table)):
        p=1
        for j in range(len(Table)):
            if i!=j:
                p= p * (pointToFindVal - Table[j][0])/(Table[i][0]-Table[j][0])
        yp= yp + p * Table[i][1]

    print("For points -",Table ,"And x value-",pointToFindVal,"We get -",yp)
    return yp

def HackathonQuestionLottery():
    table_Points = [[2.2,2.3],[0.0,0.1],[5.8,7.1],[9.4,0.4],[7.9,7.3],[4.1,3.5],[5.8,5.6],[5.1,4.1]]
    values = [3,1,5,4]

    a = Lagrange_interpolation(table_Points[0:2],values[0]*values[1])
    b,c = Lagrange_interpolation(table_Points[0:2],values[0]) , Lagrange_interpolation(table_Points[0:2],values[1])
    d,e = Lagrange_interpolation(table_Points[2:4],values[2]) , Lagrange_interpolation(table_Points[2:4],values[3])
    f = Lagrange_interpolation(table_Points[6:8],values[2]*values[3])
    print("Returned Values ->",a,b,c,d,e,f)
    print(bcolors.OKBLUE, "Part 1 is: ", math.floor(a) % 9 + 1, bcolors.ENDC)
    print(bcolors.OKBLUE, "Part 2 is: ", math.floor(b) % 8 + 11 , "And ",math.floor(c) % 8 + 11 , bcolors.ENDC)
    print(bcolors.OKBLUE, "Part 3 is: ", math.floor(d) % 11 + 20 , "And ",math.floor(e) % 11 + 20 , bcolors.ENDC)
    print(bcolors.OKBLUE, "Part 4 is: ", math.floor(f) % 6 + 32 , bcolors.ENDC)


HackathonQuestionLottery()