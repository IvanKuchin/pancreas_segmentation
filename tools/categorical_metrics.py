import tensorflow as tf


class CustomCounter(tf.keras.metrics.Metric):

    def __init__(self, name="custom_counter", **kwargs):
        super(CustomCounter, self).__init__(name = name, **kwargs)
        self.m = self.add_variable(name = "m", initializer = "zeros", shape = ())

    def reset_states(self):
        self.m.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.m.assign(self.m + 1)

    def result(self):
        return self.m

class CustomReduceMetric(tf.keras.metrics.Metric):

    def __init__(self, what="y_true", reduce="max", name="custom_max_y_true", **kwargs):
        super(CustomReduceMetric, self).__init__(name = name, **kwargs)
        self.what = what
        self.reduce = reduce
        self.m = self.add_variable(name = "m", initializer = "zeros", shape = ())

    def reset_states(self):
        self.m.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        if self.what == "y_true":
            subj = y_true
        elif self.what == "y_pred":
            subj = tf.argmax(y_pred, axis = -1)

        if self.reduce == "max":
            self.m.assign(tf.reduce_max(subj))
        elif self.reduce == "min":
            self.m.assign(tf.reduce_min(subj))
        elif self.reduce == "mean":
            self.m.assign(tf.reduce_mean(subj))
        elif self.reduce == "sum":
            self.m.assign(tf.reduce_sum(subj))

    def result(self):
        return self.m


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
        return 2 * __prec * __recall / (__prec + __recall + tf.keras.backend.epsilon())
