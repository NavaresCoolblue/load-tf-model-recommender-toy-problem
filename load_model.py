import numpy as np
import pickle
import tensorflow as tf


test_inputs = pickle.load(open("./unit_test_data/inputs.dat", "rb"))
test_predictions = pickle.load(open("./unit_test_data/predictions.dat", "rb"))
test_rev_dict = pickle.load(open("./unit_test_data/reverse_dict.dat", "rb"))

sess = tf.Session()
# 1. [LOAD]: import graph
saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
# 2. [LOAD]: restore session
saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))

# 3. [LOAD]: initialise graph
graph = tf.get_default_graph()
# 4. [LOAD]: load the input tensor
input = graph.get_tensor_by_name("inputs:0")
# 5. [LOAD]: load the dropout tensor
dropout = graph.get_tensor_by_name("dropout:0")
# 6. [LOAD]: build the prediction input doct
feed_dict = {input: test_inputs, dropout: 1.0}

# 7. [LOAD]: get prediction layer tensor
pred_layer_restored = graph.get_tensor_by_name("prediction:0")

# 8. [LOAD]: run the session with the data and the tensor
predict_with_load = sess.run(pred_layer_restored, feed_dict)

# check predictions resultsresults
total_sum = 0
for i in range(predict_with_load.shape[0]):
    total_sum += sum(predict_with_load[i] == test_predictions[i])

total_sum == predict_with_load.shape[0] * predict_with_load.shape[1]
