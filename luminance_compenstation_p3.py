# Modified from script shared by Sakari Kapanen, posted in this BA thread:
# https://blenderartists.org/t/feedback-development-filmic-baby-step-to-a-v2/1361663/1761

import colour
import numpy as np
from numpy import ndarray
import mpmath as mp

mp.prec = 500
mp.dps = 500
mp.trap_complex = True
np.set_printoptions(precision=50, floatmode='maxprec_equal')

rgb_src = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
rgb_dst = colour.RGB_COLOURSPACES["P3-D65"]
src_to_dst = rgb_dst.matrix_XYZ_to_RGB @ rgb_src.matrix_RGB_to_XYZ
dst_to_src = rgb_src.matrix_XYZ_to_RGB @ rgb_dst.matrix_RGB_to_XYZ
# luminance_coeffs = np.array([0.2658180370250449, 0.59846986045365, 0.1357121025213052])
# the coeffs above was Rec.2020 calculated using CIE 2015 CMFs and Blackbody spectra at 6504k.
# But since recently (June 22, 2025) I realized D65 spectra is not actually blackbody, I re-calculated it and produce the numbers below
# The visual result is effectively the same as before though. Though there is numerical difference, it is not a big deal.
# Check the `BT2020_2015.py` for detail
luminance_coeffs = np.array([0.2589235355689848, 0.6104985346066525, 0.13057792982436284])

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

xyz_id65_to_bt709_i65 = np.array(
    [[3.24100323297635872776822907326277, -1.53739896948878551619088739244035, -0.49861588199636291962590917137277],
     [-0.96922425220251640087809619217296, 1.87592998369517593992839010752505, 0.04155422634008471699518239006466],
     [0.05563941985197547179797794569822, -0.20401120612390993835916219723003, 1.05714897718753331190555400098674]])

xyz_id65_to_p3_id65 = np.array([[2.4935091239346101, -0.9313881794047790, -0.4027127567416516],
                                [-0.8294732139295544, 1.7626305796003032, 0.0236242371055886],
                                [0.0358512644339181, -0.0761839369220759, 0.9570295866943110]])

bt709_id65_to_xyz_id65 = np.linalg.inv(xyz_id65_to_bt709_i65)

p3_id65_to_xyz_id65 = np.linalg.inv(xyz_id65_to_p3_id65)


def compensate_low_side(rgb):
    Y = np.dot(np.matmul(dst_to_src, rgb), luminance_coeffs)

    # Calculate luminance of the opponent color, and use it to compensate for negative luminance values
    inverse_rgb = max(rgb) - rgb
    max_inverse = max(inverse_rgb)
    Y_inverse_RGB = np.dot(np.matmul(dst_to_src, inverse_rgb), luminance_coeffs)
    y_compensate_negative = (max_inverse - Y_inverse_RGB + Y)
    Y = colour.algebra.lerp(np.clip(np.power(Y, 0.08), a_min=0, a_max=1), y_compensate_negative, Y, False)
    # the lerp was because unlike in the Rec.2020 version, if we use the compensate_negative value as-is the Rec.2020-
    # green will be offset upwards too much, so lerp it to limit the compensate_negative to small values

    # Offset the input tristimulus such that there are no negatives
    min_rgb = np.amin(rgb)
    offset = np.maximum(-min_rgb, 0.0)
    rgb_offset = rgb + offset[np.newaxis]

    # Calculate luminance of the opponent color, and use it to compensate for negative luminance values
    inverse_rgb_offset = max(rgb_offset) - rgb_offset
    max_inverse_rgb_offset = max(inverse_rgb_offset)
    Y_inverse_RGB_offset = np.dot(np.matmul(dst_to_src, inverse_rgb_offset), luminance_coeffs)
    Y_new = np.dot(np.matmul(dst_to_src, rgb_offset), luminance_coeffs)
    Y_new_compensate_negative = (max_inverse_rgb_offset - Y_inverse_RGB_offset + Y_new)
    Y_new = colour.algebra.lerp(np.clip(np.power(Y_new, 0.08), a_min=0, a_max=1), Y_new_compensate_negative, Y_new,
                                False)
    # the lerp was because unlike in the Rec.2020 version, if we use the compensate_negative value as-is the Rec.2020-
    # green will be offset upwards too much, so lerp it to limit the compensate_negative to small values

    # Compensate the intensity to match the original luminance
    luminance_ratio = np.where(Y_new > Y, Y / np.clip(Y_new, a_min=1.e-100, a_max=None), 1.0)
    rgb_out = luminance_ratio[np.newaxis] * rgb_offset
    return rgb_out


def main():
    # resolution of the 3DLUT
    LUT_res = 37
    LUT = colour.LUT3D(name=f'Offset the negative values in P3 and compensate for luminance',
                       size=LUT_res)

    LUT.domain = ([[0, 0, 0], [1.0, 1.0, 1.0]])
    LUT.comments = [
        f'Lower Side of Guard Rail Formation for P3 medium.',
        f'Modified from python script shared by Sakari Kapanen',
        f'This LUT assumes E Gamut encoded in ST2084 (100) curve and then power 0.6',
        f'LUT resolution {LUT_res}']

    x, y, z, _ = LUT.table.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                col: ndarray = np.array(LUT.table[i][j][k], dtype=np.clongdouble)

                # record the original input value
                original = col

                # decode the input transfer function
                col = colour.models.exponent_function_basic(col, 0.6, "basicRev")

                col = colour.models.eotf_ST2084(col, 100)

                # decode the input primaries from E-Gamut to P3
                col = np.matmul(xyz_id65_to_p3_id65 @ e_gamut_to_xyz_id65, col)

                # record values that are outside of Rec.709
                invalid = (col < 0).any()

                # Apply lower guard rail
                col = compensate_low_side(col)

                # re-encode the primaries to E-Gamut
                col = np.matmul(xyz_id65_to_e_gamut @ p3_id65_to_xyz_id65, col)

                # re-encode transfer function
                col = colour.models.eotf_inverse_ST2084(col, 100)

                col = colour.models.exponent_function_basic(col, 0.6, "basicFwd")

                # if within Rec.709, output untouched original
                col = np.where(invalid, col, original)

                LUT.table[i][j][k] = np.array(col, dtype=LUT.table.dtype)

    colour.write_LUT(
        LUT,
        f"luminance_compensation_p3.cube")
    print(LUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
