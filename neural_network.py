import cv2
import glob
import numpy as np
import os
import tensorflow as tf
import preprocess

# tf.logging.set_verbosity(tf.logging.INFO)

n_hidden_1 = 256 # 1st layer number of neurons
n_hidden_2 = 256 # 2nd layer number of neurons
learning_rate = 0.00005
num_steps = 20000

age_num_classes = 10 # total classes (0-99 years old)
gender_num_classes = 2
img_size = 32
num_total_data = 10000

age_model_dir = "/Users/MC/Documents/class/AIProject/final_project/tf_saved_models/age_classifier"
gender_model_dir = "/Users/MC/Documents/class/AIProject/final_project/tf_saved_models/gender_classifier"

def neural_network_age_model_fn(features, labels, mode):
    """Model function for neural network."""

    # Input Layer
    # [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, img_size * img_size * 3])

    # first layer
    layer_1 = tf.layers.dense(inputs=input_layer, units=n_hidden_1)

    # second layer
    layer_2 = tf.layers.dense(inputs=layer_1, units=n_hidden_2)

    # Logits layer
    logits = tf.layers.dense(inputs=layer_2, units=age_num_classes)

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

def neural_network_gender_model_fn(features, labels, mode):
    """Model function for neural network."""

    # Input Layer
    # [batch_size, image_width, image_height, channels]
    input_layer = tf.reshape(features["x"], [-1, img_size * img_size * 3])

    # first layer
    layer_1 = tf.layers.dense(inputs=input_layer, units=n_hidden_1)

    # second layer
    layer_2 = tf.layers.dense(inputs=layer_1, units=n_hidden_2)

    # Logits layer
    logits = tf.layers.dense(inputs=layer_2, units=gender_num_classes)

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


# def train_eval_plot(num_steps, train_feature, train_label, eval_feature, eval_label, num_classes, model_fn):

#     print("hidden layer 1: " + str(n_hidden_1))
#     print("hidden layer 2: " + str(n_hidden_2))
#     # print("number of steps:" + str(num_steps))
#     print("learning rate" + str(learning_rate))

#     # build the classifier
#     classifier = tf.estimator.Estimator(model_fn)

#     # train the model
#     train_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": train_feature},
#         y=train_label,
#         batch_size=100,
#         num_epochs=None,
#         shuffle=True)
#     classifier.train(input_fn=train_input_fn,
#         steps=num_steps)

#     # evaluate the model
#     eval_input_fn = tf.estimator.inputs.numpy_input_fn(
#         x={"x": eval_feature},
#         y=eval_label,
#         num_epochs=1,
#         shuffle=False)
#     eval_results = classifier.evaluate(input_fn=eval_input_fn)
#     print("evaluation results:" + str(eval_results))

#     # # print confusion matrix
#     # predict_input_fn = tf.estimator.inputs.numpy_input_fn(
#     #     x={"x": eval_feature},
#     #     num_epochs=1,
#     #     shuffle=False)
#     # predictions = list(classifier.predict(predict_input_fn))
#     # predicted_classes = [p["classes"] for p in predictions]
#     # confusion_matrix = tf.confusion_matrix(labels=eval_label, predictions=predicted_classes, num_classes=num_classes)
#     # with tf.Session():
#     #     print("Confusion Matrix: \n\n", tf.Tensor.eval(confusion_matrix,feed_dict=None, session=None))

def main():

    total_data = preprocess.getData(img_size,num_total_data)

    # first 80% rows
    train_data = total_data[:int(num_total_data * 0.8)]
    train_feature = np.array([row[0] for row in train_data]).astype(np.float32)
    train_age_label = np.array([row[2] for row in train_data]).astype(np.int).flatten() # age
    train_gender_label = np.array([row[3] for row in train_data]).astype(np.int).flatten() # gender

    # the last 20% rows
    eval_data = total_data[int(num_total_data * 0.8):]
    eval_feature = np.array([row[0] for row in eval_data]).astype(np.float32)
    eval_age_label = np.array([row[2] for row in eval_data]).astype(np.int).flatten() # age
    eval_gender_label = np.array([row[3] for row in eval_data]).astype(np.int).flatten() # age

    # build the classifier
    age_classifier = tf.estimator.Estimator(neural_network_age_model_fn, model_dir=age_model_dir)
    gender_classifier = tf.estimator.Estimator(neural_network_gender_model_fn, model_dir=gender_model_dir)

    # train the model
    age_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_feature},
        y=train_age_label,
        batch_size=125,
        num_epochs=None,
        shuffle=True)
    age_classifier.train(input_fn=age_train_input_fn,
        steps=num_steps)

    gender_train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": train_feature},
        y=train_gender_label,
        batch_size=125,
        num_epochs=None,
        shuffle=True)
    gender_classifier.train(input_fn=gender_train_input_fn,
        steps=num_steps)

    # evaluate the model
    age_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_feature},
        y=eval_age_label,
        num_epochs=1,
        shuffle=False)
    age_eval_results = age_classifier.evaluate(input_fn=age_eval_input_fn)
    print("age evaluation results:" + str(age_eval_results))

    gender_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_feature},
        y=eval_gender_label,
        num_epochs=1,
        shuffle=False)
    gender_eval_results = gender_classifier.evaluate(input_fn=gender_eval_input_fn)
    print("gender evaluation results:" + str(gender_eval_results))


    # print confusion matrix for age
    age_predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_feature},
        num_epochs=1,
        shuffle=False)
    age_predictions = list(age_classifier.predict(age_predict_input_fn))
    predicted_ages = [p["classes"] for p in age_predictions]
    age_confusion_matrix = tf.confusion_matrix(labels=eval_age_label, predictions=predicted_ages, num_classes=age_num_classes)
    with tf.Session():
        print('Age Confusion Matrix: \n\n', tf.Tensor.eval(age_confusion_matrix,feed_dict=None, session=None))

    # print confusion matrix for gender
    gender_predict_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={"x": eval_feature},
        num_epochs=1,
        shuffle=False)
    gender_predictions = list(gender_classifier.predict(gender_predict_input_fn))
    predicted_genders = [p["classes"] for p in gender_predictions]
    gender_confusion_matrix = tf.confusion_matrix(labels=eval_gender_label, predictions=predicted_genders, num_classes=gender_num_classes)
    with tf.Session():
        print('Gender Confusion Matrix: \n\n', tf.Tensor.eval(gender_confusion_matrix,feed_dict=None, session=None))


if __name__ == "__main__":
  main()