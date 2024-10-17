import dataset.pomc_reader.classification
import dataset.pomc_reader.segmentation

def reader_factory(type:str):
        if type == "segmentation":
            return dataset.pomc_reader.segmentation.Reader
        elif type == "classification":
            return dataset.pomc_reader.classification.Reader
        else:
            raise ValueError("Unknown reader factory type: {}".format(type))