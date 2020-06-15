#%% Setup
import os
import tempfile
import apache_beam as beam
import numpy as np
import pandas as pd
from datetime import datetime

import tensorflow_hub as hub
import tensorflow as tf
import tensorflow_model_analysis as tfma
import tensorflow_data_validation as tfdv
from tensorflow_model_analysis.addons.fairness.post_export_metrics import fairness_indicators
from tensorflow_model_analysis.addons.fairness.view import widget_view
from fairness_indicators.examples import util

from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget


#%% Load the data
download_original_data = False #@param {type:"boolean"}

if download_original_data:
  train_tf_file = tf.keras.utils.get_file('train_tf.tfrecord',
                                          'https://storage.googleapis.com/civil_comments_dataset/train_tf.tfrecord')
  validate_tf_file = tf.keras.utils.get_file('validate_tf.tfrecord',
                                             'https://storage.googleapis.com/civil_comments_dataset/validate_tf.tfrecord')

  # The identity terms list will be grouped together by their categories
  # (see 'IDENTITY_COLUMNS') on threshould 0.5. Only the identity term column,
  # text column and label column will be kept after processing.
  train_tf_file = util.convert_comments_data(train_tf_file)
  validate_tf_file = util.convert_comments_data(validate_tf_file)

else:
  train_tf_file = tf.keras.utils.get_file('train_tf_processed.tfrecord',
                                          'https://storage.googleapis.com/civil_comments_dataset/train_tf_processed.tfrecord')
  validate_tf_file = tf.keras.utils.get_file('validate_tf_processed.tfrecord',
                                             'https://storage.googleapis.com/civil_comments_dataset/validate_tf_processed.tfrecord')


#%% Explore the data distribution in TFDV
stats = tfdv.generate_statistics_from_tfrecord(data_location=train_tf_file)
tfdv.visualize_statistics(stats)


#%% Calculate label distribution for gender-related examples
raw_dataset = tf.data.TFRecordDataset(train_tf_file)

toxic_gender_examples = 0
nontoxic_gender_examples = 0

# There are 1,082,924 examples in the dataset
for raw_record in raw_dataset.take(1082924):
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    if str(example.features.feature["gender"].bytes_list.value) != "[]":
        if str(example.features.feature["toxicity"].float_list.value) == "[1.0]":
          toxic_gender_examples += 1
        else:
          nontoxic_gender_examples += 1

print("Toxic Gender Examples: %s" % toxic_gender_examples)
print("Nontoxic Gender Examples: %s" % nontoxic_gender_examples)


# %% Train the model
TEXT_FEATURE = 'comment_text'
LABEL = 'toxicity'

FEATURE_MAP = {
    # Label:
    LABEL: tf.io.FixedLenFeature([], tf.float32),
    # Text:
    TEXT_FEATURE:  tf.io.FixedLenFeature([], tf.string),

    # Identities:
    'sexual_orientation':tf.io.VarLenFeature(tf.string),
    'gender':tf.io.VarLenFeature(tf.string),
    'religion':tf.io.VarLenFeature(tf.string),
    'race':tf.io.VarLenFeature(tf.string),
    'disability':tf.io.VarLenFeature(tf.string),
}


#%% 
def train_input_fn():
  def parse_function(serialized):
    parsed_example = tf.io.parse_single_example(
        serialized=serialized, features=FEATURE_MAP)
    # Adds a weight column to deal with unbalanced classes.
    parsed_example['weight'] = tf.add(parsed_example[LABEL], 0.1)
    return (parsed_example,
            parsed_example[LABEL])
  train_dataset = tf.data.TFRecordDataset(
      filenames=[train_tf_file]).map(parse_function).batch(512)
  return train_dataset


#%%
BASE_DIR = tempfile.gettempdir()

model_dir = os.path.join(BASE_DIR, 'train', datetime.now().strftime(
    "%Y%m%d-%H%M%S"))

embedded_text_feature_column = hub.text_embedding_column(
    key=TEXT_FEATURE,
    module_spec='https://tfhub.dev/google/nnlm-en-dim128/1')

classifier = tf.estimator.DNNClassifier(
    hidden_units=[500, 100],
    weight_column='weight',
    feature_columns=[embedded_text_feature_column],
    optimizer=tf.optimizers.Adagrad(learning_rate=0.003),
    loss_reduction=tf.losses.Reduction.SUM,
    n_classes=2,
    model_dir=model_dir)

classifier.train(input_fn=train_input_fn, steps=1000)


#%% Export model
def eval_input_receiver_fn():
  serialized_tf_example = tf.compat.v1.placeholder(
      dtype=tf.string, shape=[None], name='input_example_placeholder')

  # This *must* be a dictionary containing a single key 'examples', which
  # points to the input placeholder.
  receiver_tensors = {'examples': serialized_tf_example}

  features = tf.io.parse_example(serialized_tf_example, FEATURE_MAP)
  features['weight'] = tf.ones_like(features[LABEL])

  return tfma.export.EvalInputReceiver(
    features=features,
    receiver_tensors=receiver_tensors,
    labels=features[LABEL])

tfma_export_dir = tfma.export.export_eval_savedmodel(
  estimator=classifier,
  export_dir_base=os.path.join(BASE_DIR, 'tfma_eval_model'),
  eval_input_receiver_fn=eval_input_receiver_fn)


# %% Compute fairness metrics
tfma_eval_result_path = os.path.join(BASE_DIR, 'tfma_eval_result')

# NOTE: If you want to explore slicing by other categories, you can change
# the slice_section value to "sexual_orientation", "religion", "race", 
# or "disability"
slice_selection = 'gender' 

# Computing confidence intervals can help you make better decisions 
# regarding your data, but it requires computing multiple resamples, 
# so it takes significantly longer to run, particularly in Colab 
# (which cannot take advantage of parallelization), 
# so we leave it disabled here.
compute_confidence_intervals = False

# Define slices that you want the evaluation to run on.
slice_spec = [
    tfma.slicer.SingleSliceSpec(), # Overall slice
    tfma.slicer.SingleSliceSpec(columns=[slice_selection]),
]

# Add the fairness metrics.
add_metrics_callbacks = [
  tfma.post_export_metrics.fairness_indicators(
      thresholds=[0.1, 0.3, 0.5, 0.7, 0.9],
      labels_key=LABEL
      )
]

eval_shared_model = tfma.default_eval_shared_model(
    eval_saved_model_path=tfma_export_dir,
    add_metrics_callbacks=add_metrics_callbacks)

# Run the fairness evaluation.
with beam.Pipeline() as pipeline:
  _ = (
      pipeline
      | 'ReadData' >> beam.io.ReadFromTFRecord(validate_tf_file)
      | 'ExtractEvaluateAndWriteResults' >>
       tfma.ExtractEvaluateAndWriteResults(
                 eval_shared_model=eval_shared_model,
                 slice_spec=slice_spec,
                 compute_confidence_intervals=compute_confidence_intervals,
                 output_path=tfma_eval_result_path)
  )

eval_result = tfma.load_eval_result(output_path=tfma_eval_result_path)


# %% Part IV: Dig Deeper into the Data
# Limit the number of examples to 1000, so that data loads successfully
# for most browser/machine configurations. 
DEFAULT_MAX_EXAMPLES = 1000

# Load 100000 examples in memory.
def wit_dataset(file, num_examples=100000):
  dataset = tf.data.TFRecordDataset(
      filenames=[train_tf_file]).take(num_examples)
  return [tf.train.Example.FromString(d.numpy()) for d in dataset]

wit_data = wit_dataset(train_tf_file)

# Configure WIT with 1000 examples, the FEATURE_MAP we defined above, and
# a label of 1 for positive (toxic) examples and 0 for negative (nontoxic)
# examples
config_builder = WitConfigBuilder(wit_data[:DEFAULT_MAX_EXAMPLES]).set_estimator_and_feature_spec(
    classifier, FEATURE_MAP).set_label_vocab(['0', '1']).set_target_feature(LABEL)
wit = WitWidget(config_builder)
