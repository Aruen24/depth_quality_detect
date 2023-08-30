from datetime import datetime
import os.path
import time
import sys
import random
import tensorflow as tf
import numpy as np
import importlib
import argparse
import time
import json
import h5py
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
import pickle
# from queue import Queue
import random,threading,time
import utli
import train_function
import dataPath
from multiprocessing import Process,Queue
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def softmax_loss(logits,label_batches):
    with tf.name_scope('loss'):
        gamm = 2
        eps = 1e-7
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=label_batches)
        p = tf.exp(-cross_entropy)
        loss = (1 - p) ** gamm * cross_entropy
        total_loss = tf.reduce_mean(loss)
        tf.summary.scalar('cross entropy', total_loss)
        # regularization_loss = ops.get_collection(ops.GraphKeys.REGULARIZATION_LOSSES)
        # total_loss = tf.add_n([cost] + regularization_loss, name='total_loss')
        # tf.summary.scalar('total_loss', total_loss)
    return total_loss

def main(args):
    # lrrate = utli.learnrate("learn_rate/learnrate.txt", 10)
    true_data_dirs = dataPath.trueFilePaths
    false_data_dirs = dataPath.falseFilePaths

    true_paths_raw = utli.get_dataset_path_from_list(true_data_dirs)
    false_paths_raw =utli.get_dataset_path_from_list(false_data_dirs)
    num_false = len(false_paths_raw)
    num_true = len(true_paths_raw)
    print("[train] true num:%d  false num:%d" % (num_true, num_false))


    true_data_dirs = dataPath.trueValidFilePaths
    false_data_dirs = dataPath.falseValidFilePaths
    true_paths_raw_valid = utli.get_dataset_path_from_list(true_data_dirs)
    false_paths_raw_valid = utli.get_dataset_path_from_list(false_data_dirs)
    num_false_valid = len(false_paths_raw_valid)
    num_true_valid = len(true_paths_raw_valid)
    print("[Valid] true num:%d  false num:%d" % (num_true_valid, num_false_valid))


    subdir = datetime.strftime(datetime.now(), '%Y%m%d_%H%M%S')
    log_dir = os.path.join(args.log_base_dir, subdir)
    if not os.path.exists(log_dir):  # Create the log directory if it doesn't exist
        os.makedirs(log_dir)
    utli.write_arguments_to_file(args, os.path.join(log_dir, 'arguments.txt'))


    img_placeholder = tf.placeholder(tf.float32, [None, args.image_h_size, args.image_w_size, 1])
    phase_train_placeholder = tf.placeholder(tf.bool)
    label_placeholder = tf.placeholder(tf.int32, [None, 2], name='label')
    lr = tf.placeholder(tf.float32, name="learningrate")
    global_step = tf.Variable(0, trainable=False)

    network = importlib.import_module(args.model_def)
    logits, Predictions = network.inference(img_placeholder, args.keep_probability, phase_train=phase_train_placeholder, weight_decay=args.weight_decay)
    #logits, Predictions = network.interface(img_placeholder,phase_train_placeholder,2,depth_mult=1.0,Selayer=False,avgdown=False,scope='FeatherNet')
    #total_loss, t, cost = train_function.loss(logits, label_placeholder)
    total_loss = tf.losses.softmax_cross_entropy(onehot_labels=label_placeholder, logits=Predictions)
    #total_loss = tf.losses.softmax_cross_entropy(onehot_labels=label_placeholder, logits=logits)
    train_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label_placeholder, 1)), tf.float32))#train_function.get_accuracy(Predictions, label_placeholder)
    valid_acc = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(label_placeholder, 1)), tf.float32))#train_function.get_accuracy_val(Predictions, label_placeholder)
    train_op = train_function.training(total_loss, lr, global_step)

    tf.summary.scalar('learning_rate', lr)
    summary_op = tf.summary.merge_all()
    #saver = tf.train.Saver(max_to_keep = 65)
    saver = tf.train.Saver(max_to_keep = 150)


    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        summary_writer = tf.summary.FileWriter(log_dir, sess.graph)

        if args.doflip:
            dataRaw = np.zeros((num_false*2 + num_true*2, args.image_h_size, args.image_w_size, 1)).astype(np.float32)
            print("start read true data..")
            dataRaw[0:num_true, :, :, :] = utli.get_batch_data_process(true_paths_raw, num_process=25)
            dataRaw[num_true:num_true*2, :, :, :] = np.flip(dataRaw[0:num_true, :, :, :],2)
            print("start read false data..")
            dataRaw[num_true*2:num_true*2+num_false, :, :, :] = utli.get_batch_data_process(false_paths_raw,num_process=25)
            dataRaw[num_true*2+num_false:, :, :, :] = np.flip(dataRaw[num_true*2:num_true*2+num_false, :, :, :],2)
            num_true = num_true*2
            num_false = num_false*2
            labelRaw = np.zeros((num_false + num_true))
            labelRaw[0:num_true] = 1
        #if args.doflip: 
        #    dataRaw = np.zeros((num_false + num_true*2, args.image_h_size, args.image_w_size, 1)).astype(np.float32)
        #    print("start read true data..")
        #    dataRaw[0:num_true, :, :, :] = utli.get_batch_data_process(true_paths_raw, num_process=25)
        #    dataRaw[num_true:num_true*2, :, :, :] = np.flip(dataRaw[0:num_true, :, :, :],2)
        #    print("start read false data..")
        #    dataRaw[num_true*2:num_true*2+num_false, :, :, :] = utli.get_batch_data_process(false_paths_raw,num_process=25)
        #    #dataRaw[num_true*2+num_false:, :, :, :] = np.flip(dataRaw[num_true*2:num_true*2+num_false, :, :, :],2)
        #    num_true = num_true*2
        #    #num_false = num_false*2
        #    labelRaw = np.zeros((num_false + num_true))
        #    labelRaw[0:num_true] = 1
        else:
            dataRaw = np.zeros((num_false + num_true, args.image_h_size, args.image_w_size, 1)).astype(np.float32)
            print("start read true data..")
            dataRaw[0:num_true, :, :, :] = utli.get_batch_data_process(true_paths_raw, num_process=25)
            print("start read false data..")
            dataRaw[num_true:, :, :, :] = utli.get_batch_data_process(false_paths_raw, num_process=25)
            labelRaw = np.zeros((num_false + num_true))
            labelRaw[0:num_true] = 1


        # read valid dataset
        dataValid = np.zeros((num_false_valid + num_true_valid, args.image_h_size, args.image_w_size, 1)).astype(np.float32)
        print("start read valid true data..")
        dataValid[0:num_true_valid, :, :, :] = utli.get_batch_data_process(true_paths_raw_valid, num_process=25)
        print("start read valid false data..")
        dataValid[num_true_valid:, :, :, :] = utli.get_batch_data_process(false_paths_raw_valid, num_process=25)
        labelValid = np.zeros((num_false_valid + num_true_valid))
        labelValid[0:num_true_valid] = 1
        labelValid = labelValid.astype(int)
        labelValid = np.eye(2)[labelValid.reshape(-1)].reshape(len(labelValid), -1)

        num_total = num_false + num_true
        index = np.linspace(0, num_total - 1, num_total).astype(np.int)
        for epoch in range(args.max_nrof_epochs):
            random.shuffle(index)
            lrrate = utli.learnrate(args.learning_rate_schedule_file, epoch)
            for i in range(num_total//args.batch_size):
                start_index = i*args.batch_size
                end_index = start_index+args.batch_size
                temp_index = index[start_index:end_index]
                data = dataRaw[temp_index,:,:,:]

                label = labelRaw[temp_index].astype(int)
                label = np.eye(2)[label.reshape(-1)].reshape(args.batch_size, -1)


                feed_dict = {img_placeholder: data,
                             label_placeholder: label,
                             lr: lrrate,
                             phase_train_placeholder: True}

                all, trainAcc, loss, summary_str, step = sess.run([train_op,train_acc, total_loss, summary_op,global_step], feed_dict=feed_dict)

                summary_writer.add_summary(summary_str, step)

                if i%10==0:
                    print("[%d/%d] trainAcc:%.6f    loss:%.6f        lr:%.6f"%(i,epoch, trainAcc,loss,lrrate))


            saver.save(sess, log_dir + '/', global_step=step)

            feed_dict = {img_placeholder: dataValid,
                         label_placeholder: labelValid,
                         phase_train_placeholder: False}

            validAcc,predict_valid = sess.run([valid_acc, Predictions],feed_dict=feed_dict)
            FRR_all, FAR_all, acc= utli.get_roc(labelValid,predict_valid)
            print("validAcc:%.8f    FAR:%.8f    FRR:%.8f" % (validAcc,FAR_all,FRR_all))
            with open(os.path.join(log_dir, 'valid_result.txt'), 'at') as f:
                f.write('%d\tacc=%.5f\tFAR=%.5f\tFRR=%.5f\n' % (step, validAcc,FAR_all,FRR_all))

            summary = tf.Summary()
            summary.value.add(tag='accuracy/val_accuracy', simple_value=validAcc)
            summary.value.add(tag='accuracy/val_FAR', simple_value=FAR_all)
            summary.value.add(tag='accuracy/val_FRR', simple_value=FRR_all)
            summary_writer.add_summary(summary, step)





def parse_arguments(argv):

    parser = argparse.ArgumentParser()
    parser.add_argument('--log_base_dir', type=str, help='Directory where to write event logs.', default='log/faceFlat3')
    parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.',
                        default='models.faceFlat3')
    #parser.add_argument('--model_def', type=str, help='Model definition. Points to a module containing the definition of the inference graph.',
    #                    default='models.Feathernet_slim')
    #parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run.', default=105)
    parser.add_argument('--max_nrof_epochs', type=int, help='Number of epochs to run.', default=105)
    #parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=256)
    parser.add_argument('--batch_size', type=int, help='Number of images to process in a batch.', default=16)
    parser.add_argument('--doflip', type=bool, help='do flip or not.', default=True)
    parser.add_argument('--learning_rate_schedule_file', type=str, help='File containing the learning rate schedule that is used when learning_rate is set to to -1.',
                        default='learn_rate/learnrate.txt')
    parser.add_argument('--image_w_size', type=int, help='Image size (height, width) in pixels.', default=96)
    #parser.add_argument('--image_w_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--image_h_size', type=int, help='Image size (height, width) in pixels.', default=112)
    parser.add_argument('--keep_probability', type=float, help='Keep probability of dropout for the fully connected layer(s).', default=0.6)
    #parser.add_argument('--keep_probability', type=float, help='Keep probability of dropout for the fully connected layer(s).', default=0.5)
    parser.add_argument('--weight_decay', type=float, help='L2 weight regularization.', default=1e-5)

    return parser.parse_args(argv)

if __name__ == '__main__':

    main(parse_arguments(sys.argv[1:]))
