import tensorflow as tf
import math
from sklearn.metrics import roc_auc_score
import numpy as np

EPOCHS = 5
BATCH_SIZE = 2048


def train_test_model_demo(model, train_label_path, train_idx_path, train_value_path,
                          test_label_path, test_idx_path, test_value_path):
    train_batch_dataset = get_batch_dataset(train_label_path, train_idx_path, train_value_path)
    test_batch_dataset = get_batch_dataset(test_label_path, test_idx_path, test_value_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

    for epoch in range(5):
        train_model(model, train_batch_dataset, optimizer, epoch)
        test_model(model, test_batch_dataset)


def get_batch_dataset(label_path, idx_path, value_path):
    label = tf.data.TextLineDataset(label_path)
    idx = tf.data.TextLineDataset(idx_path)
    value = tf.data.TextLineDataset(value_path)

    label = label.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
    idx = idx.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
    value = value.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)

    batch_dataset = tf.data.Dataset.zip((label, idx, value))
    batch_dataset = batch_dataset.shuffle(buffer_size=20480)
    batch_dataset = batch_dataset.batch(BATCH_SIZE)
    batch_dataset = batch_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return batch_dataset


""" ************************************************************************************ """
"""                      Using Criteo DataSet to train/test Model                        """
""" ************************************************************************************ """
@tf.function
def cross_entropy_loss(y_true, y_pred):
    return tf.reduce_mean(tf.losses.binary_crossentropy(y_true, y_pred))


@tf.function
def train_one_step(model, optimizer, idx, value, label):
    with tf.GradientTape() as tape:
        output = model(idx, value)
        loss = cross_entropy_loss(y_true=label, y_pred=output)

        reg_loss = []
        for p in model.trainable_variables:
            reg_loss.append(tf.nn.l2_loss(p))
        reg_loss = tf.reduce_sum(tf.stack(reg_loss))
        loss = loss + model.reg_l2 * reg_loss

    grads = tape.gradient(loss, model.trainable_variables)
    grads = [tf.clip_by_norm(g, 100) for g in grads]
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
    return loss


def train_model(model, train_batch_dataset, optimizer, epoch):
    train_item_count = 41120555
    for batch_idx, (label, idx, value) in enumerate(train_batch_dataset):
        if len(label) == 0:
            break

        loss = train_one_step(model, optimizer, idx, value, label)

        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{} / {} ({:.2f}%)]\tLoss:{:.6f}'.format(
                epoch, batch_idx * len(idx), train_item_count,
                100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), loss.numpy()))


def test_model(model, test_batch_dataset):
    pred_y, true_y = [], []
    binaryloss = tf.keras.metrics.BinaryCrossentropy()
    for batch_idx, (label, idx, value) in enumerate(test_batch_dataset):
        if len(label) == 0:
            break

        output = model(idx, value)
        binaryloss.update_state(y_true=label, y_pred=output)
        pred_y.extend(list(output.numpy()))
        true_y.extend(list(label.numpy()))
    print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
    print('LogLoss: %.5f' % binaryloss.result())

# import tensorflow as tf
# import math
# from sklearn.metrics import roc_auc_score
# import numpy as np
#
# EPOCHS = 5
# BATCH_SIZE = 2048
#
#
# def get_batch_dataset(label_path, idx_path, value_path):
#     label = tf.data.TextLineDataset(label_path)
#     idx = tf.data.TextLineDataset(idx_path)
#     value = tf.data.TextLineDataset(value_path)
#
#     label = label.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
#     idx = idx.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
#     value = value.map(lambda x: tf.strings.to_number(tf.strings.split(x, sep='\t')), num_parallel_calls=12)
#
#     batch_dataset = tf.data.Dataset.zip((label, idx, value))
#     batch_dataset = batch_dataset.shuffle(buffer_size=20480)
#     batch_dataset = batch_dataset.batch(BATCH_SIZE)
#     batch_dataset = batch_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
#     return batch_dataset
#
#
# def train_test_model_demo(model, train_label_path, train_idx_path, train_value_path, test_label_path, test_idx_path, test_value_path):
#     train_batch_dataset = get_batch_dataset(train_label_path, train_idx_path, train_value_path)
#     test_batch_dataset = get_batch_dataset(test_label_path, test_idx_path, test_value_path)
#
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#
#     for epoch in range(5):
#         train_model(model, train_batch_dataset, optimizer, epoch)
#         test_model(model, test_batch_dataset)
#
#
# """ ************************************************************************************ """
# """                      Using Criteo DataSet to train/test Model                        """
# """ ************************************************************************************ """
# def train_model(model, train_batch_dataset, optimizer, epoch):
#     binaryloss = tf.keras.metrics.BinaryCrossentropy()
#     train_item_count = 41120555
#     for batch_idx, (label, idx, value) in enumerate(train_batch_dataset):
#         if len(label) == 0:
#             break
#
#         with tf.GradientTape() as tape:
#             output = model(idx, value)
#             loss = tf.keras.losses.binary_crossentropy(y_true=label, y_pred=output)
#             binaryloss.update_state(y_true=label, y_pred=output)
#
#         grads = tape.gradient(loss, model.trainable_variables)
#         grads = [tf.clip_by_norm(g, 100) for g in grads]
#         optimizer.apply_gradients(grads_and_vars=zip(grads, model.trainable_variables))
#         if batch_idx % 1000 == 0:
#             print('Train Epoch: {} [{} / {} ({:.2f}%)]\tLoss:{:.6f}'.format(epoch, batch_idx * len(idx), train_item_count,
#                   100. * batch_idx / math.ceil(int(train_item_count / BATCH_SIZE)), binaryloss.result()))
#
#
# def test_model(model, test_batch_dataset):
#     pred_y, true_y = [], []
#     binaryloss = tf.keras.metrics.BinaryCrossentropy()
#     for batch_idx, (label, idx, value) in enumerate(test_batch_dataset):
#         if len(label) == 0:
#             break
#
#         output = model(idx, value)
#         binaryloss.update_state(y_true=label, y_pred=output)
#         pred_y.extend(list(output.numpy()))
#         true_y.extend(list(label.numpy()))
#     print('Roc AUC: %.5f' % roc_auc_score(y_true=np.array(true_y), y_score=np.array(pred_y)))
#     print('LogLoss: %.5f' % binaryloss.result())
