# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import numpy
import random


def parseargs():
    parser = argparse.ArgumentParser(description="Shuffle corpus")

    parser.add_argument("--src", type=str, required=True,
                        help="input corpora")
    parser.add_argument("--ref", type=str, required=True,
                        help="input corpora")
    parser.add_argument("--suffix", type=str, default="reorder",
                        help="Suffix of output files")
    parser.add_argument("--num", type=int, default=1, help="times")

    return parser.parse_args()


def main(args):
    suffix = "." + args.suffix
    data1 = open(args.src, "r").readlines()
    data2 = open(args.ref, "r").readlines()
    out_x = open(args.src + suffix, "w")
    out_recover = open(args.src + ".recover", "w")
    out_ref = open(args.ref + ".refer","w")
    out_y = open(args.src + ".label", "w")
    times = args.num
    for sent,ref in zip(data1,data2):
	line = sent.strip().split()
	length = len(line)
	if length <= 5:
	    continue
        if length >= 50:
            continue
	for _ in range(times):
	    line = sent.strip().split()
	    out_idx = random.randint(0,length-2)
	    in_idx = random.randint(0,length-2)
	    if out_idx==in_idx:
		continue
	    cur_word = line.pop(out_idx)
	    out_sent = line.insert(in_idx,cur_word)
	    out_x.write(' '.join(word for word in line)+'\n')
            out_recover.write(sent)
            out_ref.write(ref)
	    if out_idx > in_idx:
		out_y.write(str(in_idx) + " "+ str(out_idx+1) +"\n")
	    else:
		out_y.write(str(in_idx) + " "+ str(out_idx) +"\n")


if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
