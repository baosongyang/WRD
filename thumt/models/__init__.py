# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import thumt.models.seq2seq
import thumt.models.rnnsearch
import thumt.models.transformer
import thumt.models.transformer_di
import thumt.models.transformer_local
import thumt.models.transformer_new
import thumt.models.transformer_ori
import thumt.models.transformer_ori2
import thumt.models.RNN_p
import thumt.models.Multi_Column


def get_model(name):
    name = name.lower()

    if name == "rnnsearch":
        return thumt.models.rnnsearch.RNNsearch
    elif name == "seq2seq":
        return thumt.models.seq2seq.Seq2Seq
    elif name == "transformer":
        return thumt.models.transformer.Transformer
    elif name == "transformer_di":
        return thumt.models.transformer_di.Transformer
    elif name == "transformer_ori":
        return thumt.models.transformer_ori.Transformer
    elif name == "transformer_ori2":
        return thumt.models.transformer_ori2.Transformer
    elif name == "transformer_new":
        return thumt.models.transformer_new.Transformer
    elif name == "transformer_local":
        return thumt.models.transformer_local.Transformer
    elif name == "rnnp":
        return thumt.models.RNN_p.Transformer
    elif name == "multi":
        return thumt.models.Multi_Column.Transformer
    else:
        raise LookupError("Unknown model %s" % name)
