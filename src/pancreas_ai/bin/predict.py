import sys
import os

from totalsegmentator.python_api import totalsegmentator
import nibabel as nib

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from pancreas_ai import config

class Predict:
    def __init__(self, ct_folder):
        self.ct_folder = os.path.join(os.getcwd(), ct_folder)

    def is_ct_folder_exists(self):        
        if not(os.path.exists(self.ct_folder)):
            return False
        

        # check if the folder contains CT files
        for root, dirs, files in os.walk(self.ct_folder):
            for file in files:
                if file.endswith(".dcm"):
                    return True

        return True
    
    def segment(self):
        segmentation = totalsegmentator(self.ct_folder, fastest=False)
        nib.save(segmentation, "segmentation.nii.gz")

def main():
    predict = Predict(config.PREDICTION_CT_FOLDER)

    print("Searching for a folder with CT-scan", end="")
    if predict.is_ct_folder_exists():
        print("\tok") 
    else:
        print("\tfail")
        return

    print("Segmentation")
    predict.segment()


if __name__ == "__main__":
    main()