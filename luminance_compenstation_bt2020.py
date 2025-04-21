# Modified from script shared by Sakari Kapanen, posted in this BA thread:
# https://blenderartists.org/t/feedback-development-filmic-baby-step-to-a-v2/1361663/1761

import colour
import numpy as np
import mpmath as mp

mp.prec = 500
mp.dps = 500
mp.trap_complex = True
np.set_printoptions(precision=50, floatmode='maxprec_equal')

rgb_src = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
rgb_dst = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
src_to_dst = rgb_dst.matrix_XYZ_to_RGB @ rgb_src.matrix_RGB_to_XYZ
luminance_coeffs = np.array([0.2658180370250449, 0.59846986045365, 0.1357121025213052])

# define some transform matrices
bt2020_id65_to_xyz_id65 = np.array([[0.6369535067850740, 0.1446191846692331, 0.1688558539228734],
                                    [0.2626983389565560, 0.6780087657728165, 0.0592928952706273],
                                    [0.0000000000000000, 0.0280731358475570, 1.0608272349505707]])

xyz_id65_to_bt2020_id65 = np.array([[1.7166634277958805, -0.3556733197301399, -0.2533680878902478],
                                    [-0.6666738361988869, 1.6164557398246981, 0.0157682970961337],
                                    [0.0176424817849772, -0.0427769763827532, 0.9422432810184308]])

e_gamut_to_xyz_id65 = np.array([[0.7053968501, 0.1640413283, 0.08101774865],
                                [0.2801307241, 0.8202066415, -0.1003373656],
                                [-0.1037815116, -0.07290725703, 1.265746519]])

xyz_id65_to_e_gamut = np.array([[1.52505277, -0.3159135109, -0.1226582646],
                                [-0.50915256, 1.333327409, 0.1382843651],
                                [0.09571534537, 0.05089744387, 0.7879557705]])


def compensate_low_side(rgb):
    # Calculate original luminance
    Y = np.dot(rgb, luminance_coeffs)

    # Calculate luminance of the opponent color, and use it to compensate for negative luminance values
    inverse_rgb = max(rgb) - rgb
    max_inverse = max(inverse_rgb)
    Y_inverse_RGB = np.dot(inverse_rgb, luminance_coeffs)
    y_compensate_negative = (max_inverse - Y_inverse_RGB + Y)

    # Offset the input tristimulus such that there are no negatives
    min_rgb = np.amin(rgb)
    offset = np.maximum(-min_rgb, 0.0)
    rgb_offset = rgb + offset[np.newaxis]

    # Calculate luminance of the opponent color, and use it to compensate for negative luminance values
    inverse_rgb_offset = max(rgb_offset) - rgb_offset
    max_inverse_rgb_offset = max(inverse_rgb_offset)
    Y_inverse_RGB_offset = np.dot(inverse_rgb_offset, luminance_coeffs)
    Y_new = np.dot(rgb_offset, luminance_coeffs)
    Y_new = (max_inverse_rgb_offset - Y_inverse_RGB_offset + Y_new)

    # Compensate the intensity to match the original luminance
    luminance_ratio = np.where(Y_new > y_compensate_negative, y_compensate_negative / Y_new, 1.0)
    rgb_out = luminance_ratio[np.newaxis] * rgb_offset
    return rgb_out


def main():
    # resolution of the 3D LUT
    LUT_res = 37
    LUT = colour.LUT3D(name=f'Offset the negative values in BT.2020 and compensate for luminance',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [f'Lower Side of Guard Rail Formation for BT.2020 medium.',
                    f'Modified from python script shared by Sakari Kapanen',
                    f'This LUT assumes input and output to be E Gamut encoded in Log2 (-10 stop to +15 stop)',
                    f'LUT resolution {LUT_res}']

    x, y, z, _ = LUT.table.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                col = np.array(LUT.table[i][j][k], dtype=np.longdouble)

                # decode the input transfer function
                col = colour.log_decoding(col,
                                          function='Log2',
                                          min_exposure=-10,
                                          max_exposure=+15,
                                          middle_grey=0.18)

                # decode the input primaries from E-Gamut to Rec.2020
                col = np.tensordot(col, e_gamut_to_xyz_id65, axes=(0, 1))

                col = np.tensordot(col, xyz_id65_to_bt2020_id65, axes=(0, 1))

                # Apply lower guard rail
                col = compensate_low_side(col)

                # re-encode the primaries to E-Gamut
                col = np.tensordot(col, bt2020_id65_to_xyz_id65, axes=(0, 1))

                col = np.tensordot(col, xyz_id65_to_e_gamut, axes=(0, 1))

                # re-encode the log2 transfer function
                col = colour.log_encoding(col,
                                          function='Log2',
                                          min_exposure=-10,
                                          max_exposure=+15,
                                          middle_grey=0.18)

                LUT.table[i][j][k] = np.array(col, dtype=LUT.table.dtype)

    colour.write_LUT(
        LUT,
        f"luminance_compensation_bt2020.cube")
    print(LUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
