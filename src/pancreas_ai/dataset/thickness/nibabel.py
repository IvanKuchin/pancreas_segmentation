def thickness(slices, axis):
    dcm_patient_pos_1_1 = slices[0][0x0020, 0x0032].value[axis]
    dcm_patient_pos_2_1 = slices[-1][0x0020, 0x0032].value[axis]

    dcm_instance_number_1 = slices[0][0x0020, 0x0013].value
    dcm_instance_number_2 = slices[-1][0x0020, 0x0013].value

    slice_thickness = (dcm_patient_pos_2_1 - dcm_patient_pos_1_1) / (dcm_instance_number_2 - dcm_instance_number_1)

    return slice_thickness
