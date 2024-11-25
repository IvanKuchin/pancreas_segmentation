def thickness(slices, axis):
    dcm_patient_pos_1_1 = slices[0][0x0020, 0x0032].value[axis]
    dcm_patient_pos_2_1 = slices[-1][0x0020, 0x0032].value[axis]

    slice_thickness = (dcm_patient_pos_1_1 - dcm_patient_pos_2_1) / (1 - len(slices))

    return slice_thickness
