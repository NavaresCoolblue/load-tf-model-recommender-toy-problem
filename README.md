# load-tf-model-recommender-toy-problem
How to save TF models and make predictions using loaded model: first draft recommender systems toy problem 

A recommended read can be found [here](https://towardsdatascience.com/deploy-tensorflow-models-9813b5a705d5)

## Requirements

```
import sys
import tensorflow as tf
import numpy as np
import pandas as pd
import itertools
from itertools import islice
import collections
import multiprocessing
import functools
from functools import partial
import math
import ml_metrics
import h5py
import os
import pickle
```

## Execution

Query and save some data from the recommender system table and save it in the *toy_input* folder as *recommender_test.csv*

```
#standardSQL
SELECT * 
FROM `coolblue-bi-data-science-exp.recommender_systems.ga_product_sequence`
limit 1000
```

### Train & save the model
 ```
 $python test_recommender.py
 ```

### Load saved model & test
 
 ```
 $python load_model.py
 ```

## TensorFlow model and session saving

In the code follow the **[DEPLOY]** tags

1. Make sure you provide a *name* to each placeholder & tensor you will want to recover once you load the session. Otherwise, it will be hard to find them

```
data = tf.placeholder(tf.float32, [None, MOVING_WINDOW, TOP_POPULAR_PRODUCTS], name="inputs")
target = tf.placeholder(tf.float32, [None, TOP_POPULAR_PRODUCTS], name="targets")
dropout = tf.placeholder(tf.float32, name="dropout")
```

2. In order to make predictions when you load the model, you want to identify the output layer tensor so label it as well with the argument *name*
```
tf.nn.softmax(tf.matmul(last, weight) + bias, name="prediction")
```

3. Once you start TF session, instantiate the Saver
```
saver = tf.train.Saver()
```

4. Do your stuff: train, cv, predict.... 

5. Save the session 

This will create a model graph and checkpoints along with tensor indices. During (4), several checkpoints can be also created in order to, for instance, stop training and continuing with it in another session. 

```
saver.save(sess, "./tmp/model.ckpt")

```

**[NOTE]*** in the toy prblem the full training is run and predictions are saved in the *unit_test* folder to compare the output with the future prediction in another session

## Tensorflow model loading and prediction

1. Instantiate another session, import the meta graph and restore the session

```
sess = tf.Session()
saver = tf.train.import_meta_graph('./tmp/model.ckpt.meta')
saver.restore(sess, tf.train.latest_checkpoint('./tmp/'))
```

2. Get the default graph already loaded (NB: several models (graphs) can be saved in one session. In this case, label them before saving and call them by the labels)
```
graph = tf.get_default_graph()
```

3. Initialize the input parameters and assing them values using a *feed_dictionary*. Remember to invoke them using the same names provided in the saved session
```
input = graph.get_tensor_by_name("inputs:0")
dropout = graph.get_tensor_by_name("dropout:0")
feed_dict = {input: test_inputs, dropout: 1.0}
```

4. Feed the graph (model) with the already trained weights (tensors) at the last model layer

```
pred_layer_restored = graph.get_tensor_by_name("prediction:0")
```

5. Run the predictions 
```
predict_with_load = sess.run(pred_layer_restored, feed_dict)
```
