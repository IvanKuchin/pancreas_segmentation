import tensorflow as tf

from . import categorical_metrics
from . import binary_metrics

class PrintMetricsInput(tf.keras.metrics.Metric):

    def __init__(self, name="custom_counter", **kwargs):
        super(PrintMetricsInput, self).__init__(name = name, **kwargs)
        self.m = self.add_variable(name = "m", initializer = "zeros", shape = ())

    def reset_states(self):
        self.m.assign(0)

    def update_state(self, y_true, y_pred, sample_weight=None):
        print("y_true:", y_true, "\t y_pred:", y_pred)

    def result(self):
        return self.m

def __classification_metrics(config: dict) -> list:
    metrics = [
        tf.keras.metrics.BinaryAccuracy(name='binary_accuracy'),
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.F1Score(name="f1"),
        tf.keras.metrics.TruePositives(name="tp"),
        tf.keras.metrics.TrueNegatives(name="tn"),
        binary_metrics.Binary_MCC(name = "custom_mcc"),
        # PrintMetricsInput(),
        # binary_metrics.precision,
        # binary_metrics.recall,
        # binary_metrics.f1,
        # binary_metrics.fbeta,
        # binary_metrics.specificity,
        # binary_metrics.negative_predictive_value,
        # binary_metrics.matthews_correlation_coefficient,
        # binary_metrics.equal_error_rate
    ]
    return metrics

def __segmentation_metrics(config: dict) -> list:
    metrics = [
        'accuracy',
        categorical_metrics.CategoricalMetric(tf.keras.metrics.TruePositives(), name = 'custom_tp'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.FalsePositives(), name = 'custom_fp'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.TrueNegatives(), name = 'custom_tn'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.FalseNegatives(), name = 'custom_fn'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.Accuracy(), name = 'custom_accuracy'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.Precision(), name = 'custom_precision'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.Recall(), name = 'custom_recall'),
        categorical_metrics.CategoricalMetric(tf.keras.metrics.MeanIoU(num_classes=2), name = 'custom_MeanIoU'),
        categorical_metrics.CategoricalF1(name = 'custom_f1'),
        categorical_metrics.CustomReduceMetric(what = "y_true", reduce = "max", name = 'custom_max_y_true'),
        categorical_metrics.CustomReduceMetric(what = "y_pred", reduce = "max", name = 'custom_max_y_pred'),
        categorical_metrics.CustomReduceMetric(what = "y_true", reduce = "min", name = 'custom_min_y_true'),
        categorical_metrics.CustomReduceMetric(what = "y_pred", reduce = "min", name = 'custom_min_y_pred'),
        categorical_metrics.CustomReduceMetric(what = "y_true", reduce = "sum", name = 'custom_sum_y_true'),
        categorical_metrics.CustomReduceMetric(what = "y_pred", reduce = "sum", name = 'custom_sum_y_pred'),
        categorical_metrics.CustomCounter(name = 'custom_counter'),
    ]
    return metrics

def metrics_factory(config: dict):
    if config.TASK_TYPE == "segmentation":
        return __segmentation_metrics(config)
    elif config.TASK_TYPE == "classification":
        return __classification_metrics(config)
    else:
        raise ValueError("Unknown loss function")