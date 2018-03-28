### Mengchu Cheng
### BUGGY!!!

import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import preprocess

n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
num_classes = 10 # total classes (0-99 years old)
img_size = 32

num_steps = 20000
learning_rate = 0.01

def neural_network_model_fn(features, labels, mode):
    """Model function for neural network."""

    print("features shape: " + str(features["x"].get_shape()))

    # Input Layer
    # [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, img_size * img_size * 3])
    print("input_layer shape: " + str(input_layer.get_shape()))

    # first layer
    layer_1 = tf.layers.dense(inputs=input_layer, units=n_hidden_1)
    print("layer 1 shape: " + str(layer_1.get_shape()))

    # second layer
    layer_2 = tf.layers.dense(inputs=layer_1, units=n_hidden_2)
    print("layer 2 shape: " + str(layer_2.get_shape()))

    #layer_2_flat = tf.reshape(layer_2, [-1,img_size*img_size*n_hidden_2])
    #print("layer 2 flat shape: " + str(layer_2_flat.get_shape()))

    # Logits layer
    logits = tf.layers.dense(inputs=layer_2, units=num_classes)
    print("logits shape: " + str(logits))

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Calculate Loss (for both TRAIN and EVAL modes)
    #onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=num_classes, dtype=tf.int32)

    print("labels shape: " + str(labels))

    #loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
    loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=logits, labels=tf.cast(labels, dtype=tf.int32)))


    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        train_op = optimizer.minimize(
            loss=loss,
            global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {
        "accuracy": tf.metrics.accuracy(
            labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(
        mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def main():

    # 10,000
    num_total_data = 5
    total_data = preprocess.getData(img_size,num_total_data)

    # first 80% rows
    train_data = total_data[:int(num_total_data * 0.8)]
    train_feature = np.array([row[0] for row in train_data]).astype(np.float32)
    print("train feature shape: " + str(train_feature.shape))
    train_labels = np.array([row[2] for row in train_data]).astype(np.int).flatten() # age
    print("train feature shape: " + str(train_labels.shape))
    print(train_labels)

    # the last 20% rows
    eval_data = total_data[int(num_total_data * 0.8):]
    eval_feature = np.array([row[0] for row in eval_data]).astype(np.float32)
    eval_labels = np.array([row[2] for row in eval_data]).astype(np.int).flatten() # age

    # build the classifier
    age_classifier = tf.estimator.Estimator(neural_network_model_fn)

    # train the model
    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_feature},
        y=train_labels,
        batch_size=1,
        num_epochs=None,
        shuffle=True)
    age_classifier.train(input_fn=train_input_fn,
        steps=num_steps)

    # evaluate the model
    # Evaluate the model and print results
    eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_feature},
        y=eval_labels,
        num_epochs=1,
        shuffle=False)
    eval_results = age_classifier.evaluate(input_fn=eval_input_fn)
    print(eval_results)

if __name__ == "__main__":
  main()