import dataset.thickness.strange as strange
import dataset.thickness.nibabel as nibabel

def ThicknessFactory(thickness_type):
    if thickness_type == 'strange':
        return strange.thickness
    elif thickness_type == 'nibabel':
        return nibabel.thickness
    else:
        raise ValueError(f'Unknown thickness class: {thickness_type}')
