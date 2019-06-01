#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 The THUMT Authors

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import itertools
import os

import numpy as np
import tensorflow as tf
from thumt.layers import nn
import thumt.data.dataset_wd as dataset
import thumt.data.vocab as vocabulary
import thumt.models as models
import thumt.utils.inference as inference
import thumt.utils.parallel as parallel


def parse_args():
    parser = argparse.ArgumentParser(
        description="Translate using existing NMT models",
        usage="translator.py [<args>] [-h | --help]"
    )

    # input files
    parser.add_argument("--softmax_size", type=int, required=True,
                        help="Size of softmax output")
    parser.add_argument("--input", type=str, nargs=2, required=True,
                        help="Path of input file")
    parser.add_argument("--eval", type=str, nargs=2, required=True,
                        help="Path of input file")
    parser.add_argument("--test", type=str, nargs=2, required=True,
                        help="Path of input file")
    parser.add_argument("--eval_steps", type=int, required=True,
                        help="Path of output file")
    parser.add_argument("--output", type=str, required=True,
                        help="Path of output file")
    parser.add_argument("--checkpoints", type=str, nargs="+", required=True,
                        help="Path of trained models")
    parser.add_argument("--vocabulary", type=str, nargs=2, required=True,
                        help="Path of source and target vocabulary")

    # model and configuration
    parser.add_argument("--models", type=str, required=True, nargs="+",
                        help="Name of the model")
    parser.add_argument("--parameters", type=str,
                        help="Additional hyper parameters")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose output")

    return parser.parse_args()


def default_parameters():
    params = tf.contrib.training.HParams(
        input=None,
        output=None,
        vocabulary=None,
        # vocabulary specific
        pad="<pad>",
        bos="<bos>",
        eos="<eos>",
        unk="<unk>",
        mapping=None,
        append_eos=False,
        # decoding
        top_beams=1,
        beam_size=4,
        decode_alpha=0.6,
        decode_length=50,
        decode_batch_size=64,
        device_list=[0],
        num_threads=1
    )

    return params


def merge_parameters(params1, params2):
    params = tf.contrib.training.HParams()

    for (k, v) in params1.values().iteritems():
        params.add_hparam(k, v)

    params_dict = params.values()

    for (k, v) in params2.values().iteritems():
        if k in params_dict:
            # Override
            setattr(params, k, v)
        else:
            params.add_hparam(k, v)

    return params


def import_params(model_dir, model_name, params):
    if model_name.startswith("experimental_"):
        model_name = model_name[13:]

    model_dir = os.path.abspath(model_dir)
    m_name = os.path.join(model_dir, model_name + ".json")

    if not tf.gfile.Exists(m_name):
        return params

    with tf.gfile.Open(m_name) as fd:
        tf.logging.info("Restoring model parameters from %s" % m_name)
        json_str = fd.readline()
        params.parse_json(json_str)

    return params


def override_parameters(params, args):
    if args.parameters:
        params.parse(args.parameters)

    params.vocabulary = {
        "source": vocabulary.load_vocabulary(args.vocabulary[0]),
        "target": vocabulary.load_vocabulary(args.vocabulary[1])
    }
    params.vocabulary["source"] = vocabulary.process_vocabulary(
        params.vocabulary["source"], params
    )
    params.vocabulary["target"] = vocabulary.process_vocabulary(
        params.vocabulary["target"], params
    )

    control_symbols = [params.pad, params.bos, params.eos, params.unk]

    params.mapping = {
        "source": vocabulary.get_control_mapping(
            params.vocabulary["source"],
            control_symbols
        ),
        "target": vocabulary.get_control_mapping(
            params.vocabulary["target"],
            control_symbols
        )
    }

    return params


def session_config(params):
    optimizer_options = tf.OptimizerOptions(opt_level=tf.OptimizerOptions.L1,
                                            do_function_inlining=False)
    graph_options = tf.GraphOptions(optimizer_options=optimizer_options)
    config = tf.ConfigProto(allow_soft_placement=True,
                            graph_options=graph_options)
    if params.device_list:
        device_str = ",".join([str(i) for i in params.device_list])
        config.gpu_options.visible_device_list = device_str

    return config


def set_variables(var_list, value_dict, prefix):
    ops = []
    for var in var_list:
        for name in value_dict:
            var_name = "/".join([prefix] + list(name.split("/")[1:]))

            if var.name[:-2] == var_name:
                tf.logging.debug("restoring %s -> %s" % (name, var.name))
                with tf.device("/cpu:0"):
                    op = tf.assign(var, value_dict[name])
                    ops.append(op)
                break

    return ops


def shard_features(features, placeholders, predictions):
    num_shards = len(placeholders)
    feed_dict = {}
    n = 0

    for name in features:
        feat = features[name]
        batch = feat.shape[0]

        if batch < num_shards:
            feed_dict[placeholders[0][name]] = feat
            n = 1
        else:
            shard_size = (batch + num_shards - 1) // num_shards

            for i in range(num_shards):
                shard_feat = feat[i * shard_size:(i + 1) * shard_size]
                feed_dict[placeholders[i][name]] = shard_feat
                n = num_shards

    return predictions[:n], feed_dict


def main(args):
    eval_steps = args.eval_steps
    tf.logging.set_verbosity(tf.logging.DEBUG)
    # Load configs
    model_cls_list = [models.get_model(model) for model in args.models]
    params_list = [default_parameters() for _ in range(len(model_cls_list))]
    params_list = [
        merge_parameters(params, model_cls.get_parameters())
        for params, model_cls in zip(params_list, model_cls_list)
    ]
    params_list = [
        import_params(args.checkpoints[i], args.models[i], params_list[i])
        for i in range(len(args.checkpoints))
    ]
    params_list = [
        override_parameters(params_list[i], args)
        for i in range(len(model_cls_list))
    ]

    # Build Graph
    with tf.Graph().as_default():
        model_var_lists = []

        # Load checkpoints
        for i, checkpoint in enumerate(args.checkpoints):
            tf.logging.info("Loading %s" % checkpoint)
            var_list = tf.train.list_variables(checkpoint)
            values = {}
            reader = tf.train.load_checkpoint(checkpoint)

            for (name, shape) in var_list:
                if not name.startswith(model_cls_list[i].get_name()):
                    continue

                if name.find("losses_avg") >= 0:
                    continue

                tensor = reader.get_tensor(name)
                values[name] = tensor

            model_var_lists.append(values)

        # Build models
        model_fns = []

        for i in range(len(args.checkpoints)):
            name = model_cls_list[i].get_name()
            model = model_cls_list[i](params_list[i], name + "_%d" % i)
            model_fn = model.get_inference_func()
            model_fns.append(model_fn)

        params = params_list[0]
        # Read input file
        #features = dataset.get_inference_input(args.input, params)
        #features_eval = dataset.get_inference_input(args.eval, params)
        #features_test = dataset.get_inference_input(args.test, params)

        features_train = dataset.get_inference_input(args.input, params, False,True)
        features_eval = dataset.get_inference_input(args.eval, params,True,False)
        features_test = dataset.get_inference_input(args.test, params,True,False)

        # Create placeholders
        placeholders = []

        for i in range(len(params.device_list)):
            placeholders.append({
                "source": tf.placeholder(tf.int32, [None, None],
                                         "source_%d" % i),
                "source_length": tf.placeholder(tf.int32, [None],
                                                "source_length_%d" % i)
                ,"target": tf.placeholder(tf.int32, [None,2], "target_%d" % i)
            })



        # A list of outputs
        predictions = parallel.data_parallelism(
            params.device_list,
            lambda f: inference.create_inference_graph(model_fns, f, params),
            placeholders)

        # Create assign ops
        assign_ops = []

        all_var_list = tf.trainable_variables()

        for i in range(len(args.checkpoints)):
            un_init_var_list = []
            name = model_cls_list[i].get_name()

            for v in all_var_list:
                if v.name.startswith(name + "_%d" % i):
                    un_init_var_list.append(v)

            ops = set_variables(un_init_var_list, model_var_lists[i],
                                name + "_%d" % i)
            assign_ops.extend(ops)

        assign_op = tf.group(*assign_ops)
        results = []

        tf_x = tf.placeholder(tf.float32, [None, None, 512])
        tf_y = tf.placeholder(tf.int32, [None,2])
        tf_x_len = tf.placeholder(tf.int32, [None])
        
        src_mask = -1e9*(1.0-tf.sequence_mask(tf_x_len,maxlen=tf.shape(predictions[0])[1],dtype=tf.float32))
        with tf.variable_scope("my_metric"):
            #q,k,v = tf.split(linear(tf_x, 3*512, True, True, scope="logit_transform"), [512, 512,512],axis=-1)
            q,k,v = tf.split(nn.linear(predictions[0], 3*512, True, True, scope="logit_transform"), [512, 512,512],axis=-1)
            q = nn.linear(tf.nn.tanh(q),1,True,True,scope="logit_transform2")[:,:,0]+src_mask
            # label smoothing
            ce1 = nn.smoothed_softmax_cross_entropy_with_logits(
                logits=q,
                labels=tf_y[:,:1],
                #smoothing=params.label_smoothing,
                smoothing=False,
                normalize=True
            )
            w1 = tf.nn.softmax(q)[:,None,:]
            #k = nn.linear(tf.nn.tanh(tf.matmul(w1,v)+k),1,True,True,scope="logit_transform3")[:,:,0]+src_mask
            k = tf.matmul(k,tf.matmul(w1,v)*(512**-0.5),False,True)[:,:,0] +src_mask
            # label smoothing
            ce2 = nn.smoothed_softmax_cross_entropy_with_logits(
                logits=k,
                labels=tf_y[:,1:],
                #smoothing=params.label_smoothing,
                smoothing=False,
                normalize=True
            )
            w2 = tf.nn.softmax(k)[:,None,:]
            weights = tf.concat([w1,w2],axis=1)
        loss = tf.reduce_mean(ce1+ce2)
        
        #tf_x = tf.placeholder(tf.float32, [None, 512])
        #tf_y = tf.placeholder(tf.int32, [None])

        #l1 = tf.layers.dense(tf.squeeze(predictions[0], axis=-2), 64, tf.nn.sigmoid)
        #output = tf.layers.dense(l1, int(args.softmax_size))

        #loss = tf.losses.sparse_softmax_cross_entropy(labels=tf_y, logits=output)
        o1 = tf.argmax(w1,axis = -1)
        o2 = tf.argmax(w2,axis = -1)
        a1, a1_update = tf.metrics.accuracy(labels=tf.squeeze(tf_y[:,0]), predictions=tf.argmax(w1, axis = -1),name='a1')
        a2, a2_update = tf.metrics.accuracy(labels=tf.squeeze(tf_y[:,1]), predictions=tf.argmax(w2, axis = -1),name='a2')
        accuracy, accuracy_update = tf.metrics.accuracy(labels=tf.squeeze(tf_y), predictions=tf.argmax(weights, axis = -1),name='a_all')

        running_vars = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="my_metric")
        #running_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="my_metric")
        running_vars_initializer = tf.variables_initializer(var_list=running_vars)

        variables_to_train = tf.trainable_variables()
        #print (len(variables_to_train), (variables_to_train[0]), variables_to_train[1])
        variables_to_train.remove(variables_to_train[0])
        variables_to_train.remove(variables_to_train[0])
        #print (len(variables_to_train))
        
        optimizer = tf.train.AdamOptimizer(learning_rate=0.00001)
        train_op = optimizer.minimize(loss, var_list=variables_to_train)
        #train_op = optimizer.minimize(loss, var_list=running_vars)
    

        # Create session
        with tf.Session(config=session_config(params)) as sess:
            init_op = tf.group(tf.global_variables_initializer(),
                    tf.local_variables_initializer())
            sess.run(init_op)
            # Restore variables
            sess.run(assign_op)
            sess.run(tf.tables_initializer())

            current_step = 0

            best_validate_acc = 0
            last_test_acc = 0

            train_x_set = []
            train_y_set = []
            valid_x_set = []
            valid_y_set = []
            test_x_set = []
            test_y_set = []
            train_x_len_set = []
            valid_x_len_set = []
            test_x_len_set = []

            while current_step < eval_steps:
                print('=======current step ' + str(current_step))
                batch_num = 0
                while True:
                    try:
                        feats = sess.run(features_train)
                        op, feed_dict = shard_features(feats, placeholders,
                                                predictions) 
                        #x = (np.squeeze(sess.run(predictions, feed_dict=feed_dict), axis = -2))
                        y =  feed_dict.values()[2]
                        x_len = feed_dict.values()[1]

                        feed_dict.update({tf_y:y})
                        feed_dict.update({tf_x_len:x_len})

                        los, __, pred = sess.run([loss, train_op, weights], feed_dict = feed_dict)
                        #print ("current_step", current_step, "batch_num", batch_num, "loss", los)
                        
                        batch_num += 1
                        if batch_num%100 == 0:

                        # eval
                            b_total = 0
                            a_total = 0
                            a1_total = 0
                            a2_total = 0
                            validate_acc = 0     
                            batch_num_eval = 0               
    
                            while True:
                                try:
                                    feats_eval = sess.run(features_eval)
                                    op, feed_dict_eval = shard_features(feats_eval, placeholders, predictions)
                                    #x = (np.squeeze(sess.run(predictions, feed_dict=feed_dict), axis = -2))
                                    y =  feed_dict_eval.values()[2]
                                    x_len =  feed_dict_eval.values()[1]
                                    feed_dict_eval.update({tf_y:y})
                                    feed_dict_eval.update({tf_x_len:x_len})
                        
                                    sess.run(running_vars_initializer)
                                    acc = 0
                                    #acc, pred = sess.run([accuracy, output], feed_dict = {tf_x : x, tf_y : y})
                                    sess.run([a1_update, a2_update, accuracy_update, weights], feed_dict = feed_dict_eval)
                                    acc1,acc2,acc = sess.run([a1,a2,accuracy])
                                    batch_size = len(y)
                                    #print(acc)
                                    a1_total += round(batch_size*acc1)
                                    a2_total += round(batch_size*acc2)
                                    a_total += round(batch_size*acc)
                                    b_total += batch_size
                                    batch_num_eval += 1

                                    if batch_num_eval == 20:
                                        break

                                except tf.errors.OutOfRangeError:
                                    print ("eval out of range")
                                    break
                            if b_total: 
                                validate_acc = a_total/b_total
                                print("eval acc : "  + str(validate_acc) + "( "+str(a1_total/b_total)+ ", "+ str(a2_total/b_total) + " )")
                            #print("last test acc : " + str(last_test_acc))

                            if validate_acc > best_validate_acc:
                                best_validate_acc = validate_acc
                                
                            # test
                            b_total = 0
                            a1_total = 0
                            a2_total = 0
                            a_total = 0
                            batch_num_test = 0
                            with open(args.output, "w") as outfile:
                                while True:
                                    try:
                                        feats_test = sess.run(features_test)
                                        op, feed_dict_test = shard_features(feats_test, placeholders,predictions)
                                    
                                        #x = (np.squeeze(sess.run(predictions, feed_dict=feed_dict), axis = -2))
                                        y =  feed_dict_test.values()[2]
                                        x_len =  feed_dict_test.values()[1]
                                        feed_dict_test.update({tf_y:y})
                                        feed_dict_test.update({tf_x_len:x_len})

                                        sess.run(running_vars_initializer)
                                        acc = 0
                                        #acc, pred = sess.run([accuracy, output], feed_dict = {tf_x : x, tf_y : y})
                                        __,__,__,out1,out2=sess.run([a1_update,a2_update,accuracy_update, o1,o2], feed_dict = feed_dict_test)
                                        acc1,acc2,acc = sess.run([a1,a2,accuracy])
                                        for pred1,pred2 in zip(out1,out2):
                                            outfile.write("%s " % pred1[0])
                                            outfile.write("%s\n" % pred2[0])
                                        batch_size = len(y)
                                        a_total += round(batch_size*acc)
                                        a1_total += round(batch_size*acc1)
                                        a2_total += round(batch_size*acc2)
                                        b_total += batch_size
                                        batch_num_test += 1

                                        if batch_num_test==20:
                                            break
                                    except tf.errors.OutOfRangeError:
                                        print ("test out of range")
                                        break
                                if b_total:
                                    last_test_acc = a_total/b_total
                                    #print("new test acc : " + str(last_test_acc)+ "( "+str(a1_total/b_total)+ ", "+ str(a2_total/b_total) + " )")

                        if batch_num == 25000:
                            break
                    except tf.errors.OutOfRangeError:
                        print ("train out of range")
                        break


                # eval
#                b_total = 0
#                a_total = 0
#                a1_total = 0
#                a2_total = 0
#                validate_acc = 0     
#                batch_num = 0               
    
#                while True:
#                    try:
#                        feats_eval = sess.run(features_eval)
#                        op, feed_dict = shard_features(feats_eval, placeholders, predictions)
#                        #x = (np.squeeze(sess.run(predictions, feed_dict=feed_dict), axis = -2))
#                        y =  feed_dict.values()[2]
#                        x_len =  feed_dict.values()[1]
#                        feed_dict.update({tf_y:y})
#                        feed_dict.update({tf_x_len:x_len})
                        
#                        sess.run(running_vars_initializer)
#                        acc = 0
                        #acc, pred = sess.run([accuracy, output], feed_dict = {tf_x : x, tf_y : y})
#                        sess.run([a1_update, a2_update, accuracy_update, weights], feed_dict = feed_dict)
#                        acc1,acc2,acc = sess.run([a1,a2,accuracy])
#                        batch_size = len(y)
                        #print(acc)
#                        a1_total += round(batch_size*acc1)
#                        a2_total += round(batch_size*acc2)
#                        a_total += round(batch_size*acc)
#                        b_total += batch_size
#                        batch_num += 1

#                        if batch_num == 10:
#                            break

#                    except tf.errors.OutOfRangeError:
#                        print ("eval out of range")
#                        break
                            
#                validate_acc = a_total/b_total
#                print("eval acc : "  + str(validate_acc) + "( "+str(a1_total/b_total)+ ", "+ str(a2_total/b_total) + " )")
#                print("last test acc : " + str(last_test_acc))

#                if validate_acc > best_validate_acc:
#                    best_validate_acc = validate_acc
                                
                    # test
#                    b_total = 0
#                    a1_total = 0
#                    a2_total = 0
#                    a_total = 0
#                    batch_num = 0

#                    while True:
#                        try:
#                            feats_test = sess.run(features_test)
#                            op, feed_dict = shard_features(feats_test, placeholders,
#                                                             predictions)
                                    
                            #x = (np.squeeze(sess.run(predictions, feed_dict=feed_dict), axis = -2))
#                            y =  feed_dict.values()[2]
#                            x_len =  feed_dict.values()[1]
#                            feed_dict.update({tf_y:y})
#                            feed_dict.update({tf_x_len:x_len})

#                            sess.run(running_vars_initializer)
#                            acc = 0
                            #acc, pred = sess.run([accuracy, output], feed_dict = {tf_x : x, tf_y : y})
#                            sess.run([a1_update,a2_update,accuracy_update, weights], feed_dict = feed_dict)
#                            acc1,acc2,acc = sess.run([a1,a2,accuracy])

#                            batch_size = len(y)
#                            a_total += round(batch_size*acc)
#                            a1_total += round(batch_size*acc1)
#                            a2_total += round(batch_size*acc2)
#                            b_total += batch_size
#                            batch_num += 1

#                            if batch_num==10:
#                                break
#                        except tf.errors.OutOfRangeError:
#                            print ("test out of range")
#                            break         
#                    last_test_acc = a_total/b_total
#                    print("new test acc : " + str(last_test_acc)+ "( "+str(a1_total/b_total)+ ", "+ str(a2_total/b_total) + " )")

                current_step += 1    
                print("")
        print("Final test acc " + str(last_test_acc))

        return


if __name__ == "__main__":
    main(parse_args())
