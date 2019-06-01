# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import random


def parseargs():
    parser = argparse.ArgumentParser(description="evaluate accuracy according to distance")

    parser.add_argument("--ref", type=str, required=True,
                        help="input reference")
    parser.add_argument("--res", type=str, required=True,
                        help="input result")
    return parser.parse_args()


def main(args):
    acc1=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    acc2=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    acc=[0.0,0.0,0.0,0.0,0.0,0.0,0.0]
    num=[0,0,0,0,0,0,0]
    num_all = 0
    acc_all = 0.0
    ref = open(args.ref, "r").readlines()
    res = open(args.res, "r").readlines()
    for pred1,pred2 in zip(ref,res):
        num_all += 1
        pred1 = pred1.strip().split()
        pred2 = pred2.strip().split()
        a1=int(pred1[0])
        a2=int(pred2[0])
        b1=int(pred1[1])
        b2=int(pred2[1])
        index=int(abs(a1-b1)/10)
        num[index]+=1
        if a1==a2:
            acc1[index]+=1
        if b1==b2:
            acc2[index]+=1
        if (a1==a2)&(b1==b2):
            acc[index]+=1
            acc_all +=1
    print ([v/n if n else 0 for v,n in zip(acc1,num)])
    print ([v/n if n else 0 for v,n in zip(acc2,num)])
    print ([v/n if n else 0 for v,n in zip(acc,num)])
    print (acc_all/num_all)


if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
