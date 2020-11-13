import os
import tensorflow as tf
import numpy as np

TFRECORD_FOLDER = "/docs/src/kt/datasets/ct-150/tfrecords/"
INPUT_DIMS = tf.constant([256, 256, 256])


def tfrecord_fname_to_patientid(fname_src):
    # print(fname_src)
    fname = tf.strings.split(fname_src, sep = "\\")[-1]
    patient_id = tf.strings.split(fname, sep = ".")[0]
    return patient_id


def py_read_data_and_label(data_fname, data_label):
    data_array = np.load(data_fname.numpy())
    label_array = np.load(data_label.numpy())
    return (data_array.astype(np.float32), label_array.astype(np.int32))
    # return data_array.astype(np.float32)


def read_data_and_label(patient_id):
    # print(patient_id)
    data_fname = TFRECORD_FOLDER + patient_id + "_data.npy"
    label_fname = TFRECORD_FOLDER + patient_id + "_label.npy"
    data_array, label_array = tf.py_function(py_read_data_and_label, [data_fname, label_fname],
                                             Tout = (tf.float32, tf.int32))

    return data_array, label_array


def crop_to_shape(data, label):
    target_shape = INPUT_DIMS
    data_shape = tf.shape(data)
    random_range = data_shape - target_shape
    random_offset = tf.cast(
        tf.math.multiply(tf.random.uniform(shape = random_range.shape), tf.cast(random_range, dtype = tf.float32)),
        dtype = tf.int32)
    _data = data[random_offset[0]:random_offset[0] + target_shape[0],
            random_offset[1]:random_offset[1] + target_shape[1], random_offset[2]:random_offset[2] + target_shape[2],
            ...]
    _label = label[random_offset[0]:random_offset[0] + target_shape[0],
             random_offset[1]:random_offset[1] + target_shape[1], random_offset[2]:random_offset[2] + target_shape[2],
             ...]
    return _data, _label


def random_flip_along_axis(tensor1, tensor2, _axis):
    if tf.cast(tf.round(tf.random.uniform([1])), tf.bool):
        tensor1 = tf.reverse(tensor1, axis = [_axis])
        tensor2 = tf.reverse(tensor2, axis = [_axis])
    return tensor1, tensor2


def random_flip(data, label):
    data, label = random_flip_along_axis(data, label, 0)
    data, label = random_flip_along_axis(data, label, 1)
    data, label = random_flip_along_axis(data, label, 2)
    return data, label

def expand_dimension(data, label):
    return data[..., tf.newaxis], label[..., tf.newaxis]


def craft_datasets(src_folder, ratio=0.2):
    list_ds = tf.data.Dataset.list_files(TFRECORD_FOLDER + "*.tfrecord").map(tfrecord_fname_to_patientid).map(
        read_data_and_label).map(crop_to_shape).map(random_flip).map(expand_dimension).batch(5)
    total_number_of_entries = tf.data.experimental.cardinality(list_ds).numpy()

    return list_ds.skip(total_number_of_entries * ratio), list_ds.take(total_number_of_entries * ratio)


def generator_downsample(filters, size, apply_batchnorm = True):
    model = tf.keras.models.Sequential()
    model.add(
        tf.keras.layers.Conv3D(filters, kernel_size=size, strides=2, padding="same")
    )
    if(apply_batchnorm):
        model.add(
            tf.keras.layers.BatchNormalization()
        )
    model.add(
        tf.keras.layers.LeakyReLU()
    )
    return model

def generator_upsample(filters, size, apply_dropout = False):
    model = tf.keras.Sequential()
    model.add(
        tf.keras.layers.Conv3DTranspose(filters, kernel_size=size, strides=2, padding="same")
    )
    model.add(
        tf.keras.layers.BatchNormalization()
    )
    if(apply_dropout):
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
        generator_downsample(128, 3), # (?,  16,  16, 128)
        generator_downsample(256, 3), # (?,   8,   8, 256)
        generator_downsample(512, 3), # (?,   4,   4, 512)
#         generator_downsample(512, 3), # (?, 2, 2, 512)
#         generator_downsample(512, 3), # (?, 1, 1, 512)
    ]

    upsample_steps = [
#         generator_upsample(512, 4, apply_dropout=True), # (?, 2, 2, 512)
#         generator_upsample(512, 4, apply_dropout=True), # (?, 4, 4, 512)
        generator_upsample(256, 3), # (?,   8,   8, 256)
        generator_upsample(128, 3), # (?,  16,  16, 128)
        generator_upsample( 64, 3), # (?,  32,  32,  64)
        generator_upsample( 32, 3), # (?,  64,  64,  32)
        generator_upsample( 16, 3), # (?, 128, 128,  16)
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

    output_layer = tf.keras.layers.Conv3DTranspose(1, kernel_size=4, strides=2, padding="same")(x)

    return tf.keras.models.Model(inputs=[inputs], outputs = [output_layer])



def main():
    # ds_train, ds_valid = craft_datasets(TFRECORD_FOLDER)
    # data, label = next(iter(ds_train))

    model = craft_network()
    # tf.keras.utils.plot_model(model, show_shapes = True, expand_nested = True)
    # print(model.summary())

    # print(data.shape)
    # pred = tf.keras.layers.Conv3D(10, kernel_size = 3, strides = 1, padding = "same")(data)
    # print(pred.shape)

    # print("Train ds:")
    # for idx, (data, label) in enumerate(ds_train):
    #     print("---", idx)
    #     print("data shape:", data.shape, "\tdata mean:", tf.reduce_mean(data))
    #     print("label shape:", label.shape, "\tlabel mean:", tf.reduce_mean(tf.cast(label, dtype=tf.float32)))
    #
    # print("Valid ds:")
    # for idx, (data, label) in enumerate(ds_valid):
    #     print("---", idx)
    #     print("data shape:", data.shape, "\tdata mean:", tf.reduce_mean(data))
    #     print("label shape:", label.shape, "\tlabel mean:", tf.reduce_mean(tf.cast(label, dtype=tf.float32)))

def main1():
    arr1, arr2 = py_read_data_and_label(["c:\\docs\\src\\kt\\datasets\\ct-150\\tfrecords\\0001_data.npy",
                                         "c:\\docs\\src\\kt\\datasets\\ct-150\\tfrecords\\0001_label.npy"])
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
