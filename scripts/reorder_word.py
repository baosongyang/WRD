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

    parser.add_argument("--corpus", type=str, required=True,
                        help="input corpora")
    parser.add_argument("--suffix", type=str, default="reorder",
                        help="Suffix of output files")
    parser.add_argument("--num", type=int, default=1, help="times")

    return parser.parse_args()


def main(args):
    suffix = "." + args.suffix
    data = open(args.corpus, "r").readlines()
    out_x = open(args.corpus + suffix, "w")
    out_y = open(args.corpus + ".label", "w")
    times = args.num
    for sent in data:
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
	    if out_idx > in_idx:
		out_y.write(str(in_idx) + " "+ str(out_idx+1) +"\n")
	    else:
		out_y.write(str(in_idx) + " "+ str(out_idx) +"\n")


if __name__ == "__main__":
    parsed_args = parseargs()
    main(parsed_args)
