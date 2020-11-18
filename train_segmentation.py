import tensorflow as tf
import numpy as np
import time
from craft_datasets import craft_datasets, py_read_data_and_label, crop_to_shape


TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/"


def generator_downsample(filters, size, apply_batchnorm=True):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv3D(filters, kernel_size = size, strides = 2, padding = "same")
    )
    if (apply_batchnorm):
        model.add(
            tf.keras.layers.BatchNormalization()
        )
    model.add(
        tf.keras.layers.ReLU()
    )
    return model


def generator_upsample(filters, size, apply_dropout=False):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv3DTranspose(filters, kernel_size = size, strides = 2, padding = "same")
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )
    if (apply_dropout):
        model.add(
            tf.keras.layers.Dropout(0.5)
        )
    model.add(
        tf.keras.layers.ReLU()
    )
    return model


def craft_network():
    downsample_steps = [
        generator_downsample(16, 3),  # (?, 128, 128, 16)
        generator_downsample(32, 3),  # (?,  64,  64, 32)
        generator_downsample(64, 3),  # (?,  32,  32, 64)
        generator_downsample(128, 3),  # (?,  16,  16, 128)
        generator_downsample(256, 3),  # (?,   8,   8, 256)
        generator_downsample(512, 3),  # (?,   4,   4, 512)
        #     generator_downsample(512, 3), # (?, 2, 2, 512)
        #         generator_downsample(512, 3), # (?, 1, 1, 512)
    ]

    upsample_steps = [
        #         generator_upsample(512, 4, apply_dropout=True), # (?, 2, 2, 512)
        #         generator_upsample(512, 4, apply_dropout=True), # (?, 4, 4, 512)
        generator_upsample(256, 3),  # (?,   8,   8, 256)
        generator_upsample(128, 3),  # (?,  16,  16, 128)
        generator_upsample(64, 3),  # (?,  32,  32,  64)
        generator_upsample(32, 3),  # (?,  64,  64,  32)
        generator_upsample(16, 3),  # (?, 128, 128,  16)
    ]

    inputs = tf.keras.layers.Input(shape = [256, 256, 256, 1])

    x = inputs
    generator_steps_otput = []
    for step in downsample_steps:
        x = step(x)
        generator_steps_otput.append(x)

    skip_conns = reversed(generator_steps_otput[:-1])
    for step, skip_conn in zip(upsample_steps, skip_conns):
        x = step(x)
        x = tf.keras.layers.Concatenate(name = "add_" + step.name)([x, skip_conn])

    output_layer = tf.keras.layers.Conv3DTranspose(2, kernel_size = 3, strides = 2, padding = "same")(x)

    return tf.keras.models.Model(inputs = [inputs], outputs = [output_layer])


def run_through_data_wo_any_action(ds_train, ds_valid):
    print("Train ds:")
    for idx, (data, label) in enumerate(ds_train):
        print("---", idx)
        print("data shape:", data.shape, "\tdata mean:", tf.reduce_mean(data))
        print("label shape:", label.shape, "\tlabel mean:", tf.reduce_mean(tf.cast(label, dtype = tf.float32)))
        # print(label)
        # break

    print("Valid ds:")
    for idx, (data, label) in enumerate(ds_valid):
        print("---", idx)
        print("data shape:", data.shape, "\tdata mean:", tf.reduce_mean(data))
        print("label shape:", label.shape, "\tlabel mean:", tf.reduce_mean(tf.cast(label, dtype = tf.float32)))


def predict_on_random_data(model):
    for i in range(1, 32):
        t0 = time.time()
        pred = model.predict(tf.random.normal([i, 256, 256, 256, 1]))
        t1 = time.time()
        print("prediction shape/time:", pred.shape, "/", np.round(t1 - t0))


def main():
    ds_train, ds_valid = craft_datasets(TFRECORD_FOLDER)
    # run_through_data_wo_any_action(ds_train, ds_valid)

    model = craft_network()
    # predict_on_random_data()

    model.compile(optimizer = "adam", loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True),
                  metrics = ['accuracy', 'sparse_categorical_crossentropy'])

    history = model.fit(ds_train, epochs = 1, validation_data = ds_valid)
    print("")
    print("---------------")
    print(history.history)



def main1():

    arr1, arr2 = py_read_data_and_label(tf.constant("c:\\docs\\src\\kt\\datasets\\ct-150\\tfrecords\\0001_data.npy"),
                                        tf.constant("c:\\docs\\src\\kt\\datasets\\ct-150\\tfrecords\\0001_label.npy"))
    print(arr1.shape)
    print(arr2.shape)


def main2():
    arr1 = tf.random.uniform([2, 270, 270, 270])
    random_offset = crop_to_shape(arr1, arr1)
    print(tf.shape(random_offset[0]))


if __name__ == "__main__":
    main()
    # main1()
    # main2()


# 25/Unknown - 438s 18s/step - loss: 0.3738 - accuracy: 0.9191 - sparse_categorical_crossentropy: 0.53792020-11-17 23:25:56.334274: W tensorflow/core/framework/op_kernel.cc:1622] OP_REQUIRES failed at sparse_xent_op.cc:90 : Invalid argument: Received a label value of -1 which is outside the valid range of [0, 2).  Label values: 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
