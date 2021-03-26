import tensorflow as tf


class CategoricalMetric(tf.keras.metrics.Metric):

    def __init__(self, metric, name="custom_metric", **kwargs):
        super(CategoricalMetric, self).__init__(name = name, **kwargs)
        self.m = metric

    def reset_states(self):
        self.m.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis = -1)
        y_pred = y_pred[..., tf.newaxis]
        self.m.update_state(tf.maximum(y_true, 0), y_pred, sample_weight)

    def result(self):
        return self.m.result()


class CategoricalF1(tf.keras.metrics.Metric):

    def __init__(self, name="custom_f1", **kwargs):
        super(CategoricalF1, self).__init__(name = name, **kwargs)
        self.precision = tf.keras.metrics.Precision()
        self.recall = tf.keras.metrics.Recall()

    def reset_states(self):
        self.precision.reset_states()
        self.recall.reset_states()

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.argmax(y_pred, axis = -1)
        y_pred = y_pred[..., tf.newaxis]
        self.precision.update_state(tf.maximum(y_true, 0), y_pred, sample_weight)
        self.recall.update_state(tf.maximum(y_true, 0), y_pred, sample_weight)

    def result(self):
        __prec = self.precision.result()
        __recall = self.recall.result()
        return 2 * __prec * __recall / (__prec + __recall)
