#!/bin/bash

chmod +x /opt/venv_totalsegmentator/bin/
chmod +x /opt/venv_tensorflow/bin/

. /opt/venv_totalsegmentator/bin/activate
TotalSegmentator -i ./predict/ -o ./predict/segmentation.nii.gz -ta total -m

. /opt/venv_tensorflow/bin/activate
predict

