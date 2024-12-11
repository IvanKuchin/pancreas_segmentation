import time
import glob
import os
import sys
import re
import numpy.typing as npt
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai.dataset.pomc_reader.factory import reader_factory
from pancreas_ai.dataset.savers.factory import saver_factory
from pancreas_ai.dataset.pomc_reader.preprocess import preprocess_data
import pancreas_ai.config as config

# INPUT_DIMS = np.array([config.IMAGE_DIMENSION_X, config.IMAGE_DIMENSION_Y, config.IMAGE_DIMENSION_Z])
# AUGMENT_SCALE_FACTOR = 0.1
# AUGMENT_SCALED_DIMS = tf.cast(tf.constant(INPUT_DIMS, dtype = tf.float32) * (1 + AUGMENT_SCALE_FACTOR),
#                               dtype = tf.int32).numpy()

################### Common parameters ###################

PATIENTS_SRC_FOLDER = config.POMC_PATIENTS_SRC_FOLDER
LABELS_SRC_FOLDER   = config.POMC_LABELS_SRC_FOLDER

DEBUG = True

class POMCDataset:
    def __init__(self, patients_src_folder, labels_src_folder, TFRECORD_FOLDER):
        self.patients_src_folder = patients_src_folder
        self.labels_src_folder = labels_src_folder
        self.TFRECORD_FOLDER = TFRECORD_FOLDER

        self.saver = saver_factory(config)
        self.reader = reader_factory(config.TASK_TYPE, config)


    def get_patient_id_from_folder(self, folder):
        result = None
        m = re.search("(\\w+)$", folder)
        if m:
            result = m.group(1)
        return result

    # def _body_up_side_down(self, min):
    #     return min[2] > 0

    def print_statistic(self, tensor_old, tensor_new):
        print("\tshape:\t\t{} -> {}".format(tensor_old.shape, tensor_new.shape))
        print("\tmin/max:\t{}/{} -> {}/{}".format(np.min(tensor_old), np.max(tensor_old), np.min(tensor_new),
                                                np.max(tensor_new)))
        print("\tmean std:\t{:.3f} {:.3f} -> {:.3f} {:.3f}".format(np.mean(tensor_old), np.std(tensor_old), np.mean(tensor_new), np.std(tensor_new)))
        print("\tsum:\t\t{} -> {}".format(np.sum(tensor_old), np.sum(tensor_new)))

    def preprocess_data(self, data: npt.NDArray[np.float32], label: npt.NDArray[np.float32]) -> tuple[npt.NDArray[np.float32], npt.NDArray[np.float32]]:
        return preprocess_data(data, label, config)

    def save_npz(self, subfolder: str, patient_id:str, percentage: int, original_data, original_label, scaled_data, scaled_label):
        result = True
        scaled_data = np.cast[np.float32](scaled_data)
        scaled_label = np.cast[np.int8](scaled_label)
        np.savez_compressed(os.path.join(self.TFRECORD_FOLDER, subfolder, patient_id + f"_{percentage}.npz", ), [scaled_data, scaled_label])

        return result

    def pickle_src_data(self, train_valid_percentage=config.VALIDATION_PERCENTAGE):
        if not os.path.exists(self.TFRECORD_FOLDER):
            print("ERROR: can't find TFRecord folder:", self.TFRECORD_FOLDER)
            return
        if not os.path.exists(os.path.join(self.TFRECORD_FOLDER, "train")):
            os.makedirs(os.path.join(self.TFRECORD_FOLDER, "train"))
        if not os.path.exists(os.path.join(self.TFRECORD_FOLDER, "valid")):
            os.makedirs(os.path.join(self.TFRECORD_FOLDER, "valid"))

        folder_list = glob.glob(os.path.join(self.patients_src_folder, "*"))

        for folder in folder_list:
            if not os.path.isdir(folder):
                continue

            subfolder = "train" if np.random.rand() > train_valid_percentage else "valid"

            patient_id = self.get_patient_id_from_folder(folder)
            if len(patient_id) == 0:
                print("ERROR: identifying patient_id from folder:", folder)
                continue

            # --- read data about patient
            print("Read data about patient", patient_id)

            start_ts = time.time()
            src_data, src_metadata = self.reader.read_data(folder)
            if len(src_data.shape) == 0:
                print("ERROR: can't read DICOM data from folder:", folder)
                continue
            print("\tRead DICOM data in {:.2f} sec".format(time.time() - start_ts))

            start_ts = time.time()
            label_data, label_metadata = self.reader.read_label(folder)
            if len(label_data.shape) == 0:
                print("ERROR: can't find nifti labels:", patient_id)
                continue
            print("\tRead NRRD labels in {:.2f} sec".format(time.time() - start_ts))

            if self.reader.check_before_processing(src_data, label_data, src_metadata, label_metadata) == False:
                print("ERROR: data & labels are not consistent patient_id:", patient_id)
                continue

            for percentage in config.CUTOUT_BORDER_FROM_PANCREAS:
                print(f"\n\tPreprocess data for {percentage}%")

                scaled_data, scaled_label = self.reader.rescale_if_needed(src_data, label_data, percentage)

                start_ts = time.time()
                scaled_src_data, scaled_label_data = self.preprocess_data(scaled_data, scaled_label)
                print("\tPreprocess data in {:.2f} sec".format(time.time() - start_ts))

                if DEBUG:
                    print("\tData statistics:")
                    self.print_statistic(src_data, scaled_src_data)
                    print("\tLabel statistics:")
                    self.print_statistic(label_data, scaled_label_data)

                if self.reader.check_after_preprocessing(scaled_src_data, scaled_label_data) == False:
                    print("ERROR: data or label failed sanity check")
                    continue

                # if self.save_npz(subfolder, patient_id, percentage, src_data, label_data, scaled_src_data, scaled_label_data) == False:
                #     print("ERROR: can't save TFRecord patient id:", patient_id)
                #     continue

                print(f"\tSave patientID: {patient_id} to {subfolder} with border cut out around pancreas at {percentage}%")
                saver_obj = self.saver(self.TFRECORD_FOLDER, subfolder, patient_id, percentage, config)
                if saver_obj.save(scaled_src_data, scaled_label_data) == False:
                    print("ERROR: can't save sliced CT of patientID:", patient_id)
                    continue

def main():
    pomc = POMCDataset(PATIENTS_SRC_FOLDER, LABELS_SRC_FOLDER, config.TFRECORD_FOLDER)
    pomc.pickle_src_data()
    return

if __name__ == "__main__":
    main()
