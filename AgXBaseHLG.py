import math
import colour
import numpy
import re
import sigmoid
import argparse
import luminance_compenstation_bt2020 as lu2020
import luminance_compenstation_p3 as lup3

# Log range parameters
midgrey = 0.18
normalized_log2_minimum = -10
normalized_log2_maximum = +6.5

# define color space matrices
bt2020_id65_to_xyz_id65 = numpy.array([[0.6369535067850740, 0.1446191846692331, 0.1688558539228734],
                                       [0.2626983389565560, 0.6780087657728165, 0.0592928952706273],
                                       [0.0000000000000000, 0.0280731358475570, 1.0608272349505707]])

xyz_id65_to_bt2020_id65 = numpy.array([[1.7166634277958805, -0.3556733197301399, -0.2533680878902478],
                                       [-0.6666738361988869, 1.6164557398246981, 0.0157682970961337],
                                       [0.0176424817849772, -0.0427769763827532, 0.9422432810184308]])

# inset matrix from Troy's SB2383 script, setting is rotate = [3.0, -1, -2.0], inset = [0.4, 0.22, 0.13]
# link to the script: https://github.com/sobotka/SB2383-Configuration-Generation/blob/main/generate_config.py
# the relevant part is at line 88 and 89
inset_matrix = numpy.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                            [0.137318972929847, 0.761241990602591, 0.101439036467562],
                            [0.11189821299995, 0.0767994186031903, 0.811302368396859]])

# outset matrix from Troy's SB2383 script, setting is rotate = [0, 0, 0] inset = [0.4, 0.22, 0.04], used on inverse
# link to the script: https://github.com/sobotka/SB2383-Configuration-Generation/blob/main/generate_config.py
# the relevant part is at line 88 and 89
outset_matrix = numpy.linalg.inv(numpy.array([[0.899796955911611, 0.0871996192028351, 0.013003424885555],
                                              [0.11142098895748, 0.875575586156966, 0.0130034248855548],
                                              [0.11142098895748, 0.0871996192028349, 0.801379391839686]]))

# these lines are dependencies from Troy's AgX script
x_pivot = numpy.abs(normalized_log2_minimum) / (
        normalized_log2_maximum - normalized_log2_minimum
)

# define SDR max nits
SDRMax = 100
HDRMax = 1000
HDR_SDR_Ratio = HDRMax / SDRMax
midgrey_offset_power = math.log(0.18 / HDR_SDR_Ratio, 0.18)

# parameters used for compensating for midgrey offset power curve's per-channel result
# larger power value will result in more chroma laden image, lower value would result in less chroma landen result
# increase the lower domain limit will limit the upper bound of chroma level, decrease the upper domain limit will limit the lower bound of the chroma level
# todo: Use an actual HDR capable device, test in DaVinci Resolve, and find the setting that matches SDR AgX Base Rec.2020 the most
# I (Eary) don't have an HDR capable device so I probably won't be the one doing it. Blender's HDR/EDR support in 4.0 seems to be broken, so maybe test this in Resolve.
chroma_mix_power_of_value = 1.3
chroma_mix_value_domain = [0, 1]

# define middle grey
y_pivot = colour.models.eotf_inverse_BT2100_HLG(
    colour.models.exponent_function_basic(midgrey, midgrey_offset_power, 'basicFwd') * HDRMax)

exponent = [0.4, 0.4]
slope = 2.4

argparser = argparse.ArgumentParser(
    description="Generates an OpenColorIO configuration",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)
argparser.add_argument(
    "-et",
    "--exponent_toe",
    help="Set toe curve rate of change as an exponential power, hello Sean Cooper",
    type=float,
    default=exponent[0],
)
argparser.add_argument(
    "-ps",
    "--exponent_shoulder",
    help="Set shoulder curve rate of change as an exponential power",
    type=float,
    default=exponent[1],
)
argparser.add_argument(
    "-fs",
    "--fulcrum_slope",
    help="Set central section rate of change as rise over run slope",
    type=float,
    default=slope,
)
argparser.add_argument(
    "-fi",
    "--fulcrum_input",
    help="Input fulcrum point relative to the normalized log2 range",
    type=float,
    default=x_pivot,
)
argparser.add_argument(
    "-fo",
    "--fulcrum_output",
    help="Output fulcrum point relative to the normalized log2 range",
    type=float,
    default=y_pivot,
)
argparser.add_argument(
    "-ll",
    "--limit_low",
    help="Lowest value of the normalized log2 range",
    type=float,
    default=normalized_log2_minimum,
)
argparser.add_argument(
    "-lh",
    "--limit_high",
    help="Highest value of the normalized log2 range",
    type=float,
    default=normalized_log2_maximum,
)

args = argparser.parse_args()


# these lines are dependencies from Troy's AgX script


def apply_sigmoid(x):
    sig_x_input = x

    col = sigmoid.calculate_sigmoid(
        sig_x_input,
        pivots=[args.fulcrum_input, args.fulcrum_output],
        slope=args.fulcrum_slope,
        powers=[args.exponent_toe, args.exponent_shoulder],
    )

    return col


def AgX_Base_Rec2020(col, mix_percent):
    # apply lower guard rail
    col = lu2020.compensate_low_side(col)

    # apply inset matrix
    col = numpy.tensordot(col, inset_matrix, axes=(0, 1))

    # record current chromaticity angle
    pre_form_hsv = colour.RGB_to_HSV(col)

    # apply Log2 curve to prepare for sigmoid
    log = colour.log_encoding(col,
                              function='Log2',
                              min_exposure=normalized_log2_minimum,
                              max_exposure=normalized_log2_maximum,
                              middle_grey=midgrey)

    # apply sigmoid
    col = apply_sigmoid(log)

    # Linearize
    col = colour.models.exponent_function_basic(col, 1 / math.log(y_pivot, 0.18), 'basicFwd')

    pre_middel_grey_lowering_hsv = colour.models.RGB_to_HSV(col)

    # lower the middle grey, so upon end encoding, the middle grey matches the common "SDR 1.0=100nits" HLG implementation
    # This SDR 1.0=100nits HDR implementation is weird since middle grey end up being 1.8% of the max emission, instead of 18%.
    # But this is how it is done in Davinci Resolve and OCIO's builtin transform for HLG etc.
    col = colour.models.exponent_function_basic(col, midgrey_offset_power, 'basicFwd')

    # a hack trying to match HDR and SDR, compensating the per-channel nature of the additional power curve (that we applied to match the middle grey of SDR=100nits assumption)
    col = colour.models.RGB_to_HSV(col)

    col[1] = colour.algebra.lerp(
        numpy.clip(pre_middel_grey_lowering_hsv[2] ** chroma_mix_power_of_value, a_min=chroma_mix_value_domain[0],
                   a_max=chroma_mix_value_domain[1]), col[1], pre_middel_grey_lowering_hsv[1], False)
    col = colour.models.HSV_to_RGB(col)

    # record post-sigmoid chroma angle
    col = colour.RGB_to_HSV(col)

    # mix pre-formation chroma angle with post formation chroma angle.
    col[0] = colour.algebra.lerp(mix_percent / 100, pre_form_hsv[0], col[0], False)

    col = colour.HSV_to_RGB(col)

    # apply outset to make the result more chroma-laden
    col = numpy.tensordot(col, outset_matrix, axes=(0, 1))

    col = numpy.clip(col, a_min=0, a_max=1)
    return col


colour.utilities.filter_warnings(python_warnings=True)


def main():
    # resolution of the 3D LUT
    LUT_res = 45

    # The mix_percent here is the mixing factor of the pre- and post-formation chroma angle. Specifically, a simple HSV here was used.
    # Mixing, or lerp-ing the H is a hack here that does not fit a first-principle design.
    # I tried other methods but this seems to be the most straight forward way.
    # I just can't bare to see our rotation of primaries, the "flourish", is messed up with a per-channel notorious six hue shift.
    # This means if we rotate red a bit towards orange for countering abney effect, the orange will then be skewed to yellow.
    # Then we apply the rotation in different primaries, like in BT.2020, where BT.709 red is already more orangish in the first place,
    # this gets magnified. Troy's original version has outset that also includes the inverse rotation, but because the original rotation
    # has already been skewed by the per-channel N6, the outset matrix in his version didn't cancel the rotation. This seems like such a
    # mess to me, so I decided to take this hacky approach at least to get the flourish rotation somewhat in control.
    # The result is also that my outset matrix now doesn't contain any rotation, otherwise the original rotation can actually be cancelled.
    # The number of 40% here is based on personal testing, you can try to test which number works better if you would like to change it.
    mix_percent = 40

    LUT = colour.LUT3D(name=f'AgX_Formation_Rec2100HLG',
                       # LUT = colour.LUT3D(name=f'AgX_Formation_Rec2100HLG_P3_Limited',
                       # LUT = colour.LUT3D(name=f'No_Guard_Rail_AgX_Formation_Rec2100HLG',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [
        f'AgX Base Rec.2100 Formation LUT designed to target the {HDRMax} nits HLG medium with assumption of SDR = {SDRMax} nits',
        f'per-channel chroma offset compensation power value = {chroma_mix_power_of_value}, domain for that mix factor is {chroma_mix_value_domain}',
        f'This LUT expects input to be E Gamut Log2 encoding from -10 stops to +15 stops',

        # f'AgX Base Rec.2020 Formation LUT designed to be used on Inverse',
        # f'This LUT expects input (output if inverse) to be Rec.2020 Log2 encoding from -10 stops to +6.5 stops',

        f'But the end image formation will be Rec2100-HLG',
        # f'But the end image formation will be Rec2100-HLG with gamut limited to Display P3',
        f' rotate = [3.0, -1, -2.0], inset = [0.4, 0.22, 0.13], outset = [0.4, 0.22, 0.04]',
        f'The image formed has {mix_percent}% per-channel shifts',
                    f'DOMAIN_MIN 0 0 0',
                    f'DOMAIN_MAX 1 1 1']

    x, y, z, _ = LUT.table.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                col = numpy.array(LUT.table[i][j][k], dtype=numpy.longdouble)

                # decode LUT input transfer function (change max to 6.5 when generating no guard rail version)
                col = colour.log_decoding(col,
                                          function='Log2',
                                          min_exposure=-10,
                                          max_exposure=+15,
                                          middle_grey=midgrey)

                # decode LUT input primaries from E-Gamut to Rec.2020 (mute when generating no guard rail version)
                col = numpy.tensordot(col, lu2020.e_gamut_to_xyz_id65, axes=(0, 1))

                col = numpy.tensordot(col, lu2020.xyz_id65_to_bt2020_id65, axes=(0, 1))

                col = AgX_Base_Rec2020(col, mix_percent)

                # Apply P3 Lower Rail for P3 limited output, mute these lines below for full Rec.2020 gamut output
                # or unmute the lines below if you want to limit output to P3 gamut.
                # col = numpy.tensordot(col, lu2020.bt2020_id65_to_xyz_id65, axes=(0, 1))
                # col = numpy.tensordot(col, lup3.xyz_id65_to_p3_id65, axes=(0, 1))
                # col = lup3.compensate_low_side(col)
                # col = numpy.tensordot(col, lup3.p3_id65_to_xyz_id65, axes=(0, 1))
                # col = numpy.tensordot(col, lu2020.xyz_id65_to_bt2020_id65, axes=(0, 1))

                # re-encode transfer function
                col = colour.models.eotf_inverse_BT2100_HLG(col * HDRMax)

                col = numpy.clip(col, a_min=0, a_max=1)

                LUT.table[i][j][k] = numpy.array(col, dtype=LUT.table.dtype)

    LUT_name = f"AgX_Base_Rec2100-HLG.cube"
    # LUT_name = f"AgX_Base_Rec2100-HLG_P3_Limited.cube")
    # LUT_name = f"No_GR_AgX_Base_Rec2100-HLG.cube")
    colour.write_LUT(
        LUT,
        LUT_name)
    print(LUT)
    written_lut = open(LUT_name).read()
    written_lut = written_lut.replace('# DOMAIN_', 'DOMAIN_')
    written_lut = written_lut.replace('nan', '0')

    def remove_trailing_zeros(text):
        # Regular expression to find numbers in the text
        pattern = r'\b(\d+\.\d*?)(0+)(?=\b|\D)'

        # Replace each found number with trailing zeros removed
        def replace_zeros(match):
            # Remove trailing zeros and, if there are no digits after the decimal point, remove the point as well
            after_decimal = match.group(1).rstrip('0')
            if after_decimal.endswith('.'):
                after_decimal = after_decimal.rstrip('.')
            return after_decimal

        # Split the text into lines and process each line
        lines = text.split('\n')
        modified_lines = []

        for line in lines:
            if not line.startswith('#'):
                modified_lines.append(re.sub(pattern, replace_zeros, line))
            else:
                modified_lines.append(line)  # Keep lines starting with #

        # Join the modified lines back into text
        result = '\n'.join(modified_lines)
        return result

    written_lut = remove_trailing_zeros(written_lut)

    open(LUT_name, 'w').write(written_lut)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
