# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import nltk


def parseargs():
    parser = argparse.ArgumentParser(description="evaluate accuracy according to distance")

    parser.add_argument("--y", type=str, required=True,
                        help="input reference")
    parser.add_argument("--ref", type=str, required=True,
                        help="input result")
    parser.add_argument("--res1", type=str, required=True,
                        help="input result")
    parser.add_argument("--res2", type=str, required=True,
                        help="input result")
    return parser.parse_args()


def main(args):
    ss1=[[],[],[],[],[],[],[]]
    ss2=[[],[],[],[],[],[],[]]
    ss=[[],[],[],[],[],[],[]]

    res1 = open(args.res1, "r").readlines()
    res2 = open(args.res2, "r").readlines()
    ref = open(args.ref, "r").readlines()

    y = open(args.y, "r").readlines()
    for r1,r2,r,pred1 in zip(res1,res2,ref,y):
        pred1 = pred1.strip().split()

        a1=int(pred1[0])
        b1=int(pred1[1])
        index=int(abs(a1-b1)/10)
        #ss1[index].append(r1)
        #ss2[index].append(r2)
        #ss[index].append([r])
        ss1[index].append(r1.strip().split())
        ss2[index].append(r2.strip().split())
        ss[index].append([r.strip().split()])

    print ([nltk.translate.bleu_score.corpus_bleu(i,j,weights=(0.25,0.25,0.25,0.25)) if i!= [] else 0.0 for i,j in zip(ss,ss1)])
    print ([nltk.translate.bleu_score.corpus_bleu(i,j,weights=(0.25,0.25,0.25,0.25)) if i!= [] else 0.0 for i,j in zip(ss,ss2)])
    #print ([(len(i,j)) if i!= [] else 0.0 for i,j in zip(ss1,ss)])
    #print ([(len(i,j)) if i!= [] else 0.0 for i,j in zip(ss2,ss)])




if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
