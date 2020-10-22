import tensorflow as tf
import numpy as np

class lossObj:

    #define possible loss functions like dice score, cross entropy, weighted cross entropy

    def __init__(self):
        print('loss init')

    def dice_loss_with_backgrnd(self, logits, labels, epsilon=1e-10):
        '''
        Calculate a dice loss defined as `1-foreground_dice`. Default mode assumes that the 0 label
         denotes background and the remaining labels are foreground.
        input params:
            logits: Network output before softmax
            labels: ground truth label masks
            epsilon: A small constant to avoid division by 0
        returns:
            loss: Dice loss with background
        '''

        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

            loss = 1 - tf.reduce_mean(dices_per_subj)
        return loss

    def dice_loss_without_backgrnd(self, logits, labels, epsilon=1e-10, from_label=1, to_label=-1):
        '''
        Calculate a dice loss of only foreground labels without considering background class.
        Here, label 0 is background and the remaining labels are foreground.
        input params:
            logits: Network output before softmax
            labels: ground truth label masks
            epsilon: A small constant to avoid division by 0
            from_label: First label to evaluate
            to_label: Last label to evaluate
        returns:
            loss: Dice loss without background
        '''

        with tf.name_scope('dice_loss'):

            prediction = tf.nn.softmax(logits)

            intersection = tf.multiply(prediction, labels)
            intersec_per_img_per_lab = tf.reduce_sum(intersection, axis=[1, 2])

            l = tf.reduce_sum(prediction, axis=[1, 2])
            r = tf.reduce_sum(labels, axis=[1, 2])

            dices_per_subj = 2 * intersec_per_img_per_lab / (l + r + epsilon)

            loss = 1 - tf.reduce_mean(tf.slice(dices_per_subj, (0, from_label), (-1, to_label)))
        return loss

    def pixel_wise_cross_entropy_loss(self, logits, labels):
        '''
        Simple wrapper for the normal tensorflow cross entropy loss
        input params:
            logits: Network output before softmax
            labels: Ground truth masks
        returns:
            loss:  weighted cross entropy loss
        '''

        loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
        return loss

    def pixel_wise_cross_entropy_loss_weighted(self, logits, labels, class_weights):
        '''
        Weighted cross entropy loss, with a weight per class
        input params:
            logits: Network output before softmax
            labels: Ground truth masks
            class_weights: A list of the weights for each class
        returns:
            loss:  weighted cross entropy loss
        '''

        # deduce weights for batch samples based on their true label
        weights = tf.reduce_sum(class_weights * labels, axis=3)

        # For weighted error
        # compute your (unweighted) softmax cross entropy loss
        unweighted_losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits,labels=labels)
        # apply the weights, relying on broadcasting of the multiplication
        weighted_losses = unweighted_losses * weights
        # reduce the result to get your final loss
        loss = tf.reduce_mean(weighted_losses)

        return loss

