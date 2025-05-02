import colour
import numpy
import sigmoid
import argparse
import luminance_compenstation_bt2020 as lu2020
import luminance_compenstation_p3 as lup3
import Guard_Rail_Upper as high_rail
import working_space
import math
import re

# Log range parameters
midgrey = 0.18
normalized_log2_minimum = -10
normalized_log2_maximum = +6.5

# Define HDR related parameters
HDR_max_nits = 1000
SDR_max_nits = 203
# In a dynamically tunable protocol, HDR max may be tunable by users from 400 nits to 1000 nits, SDR max may be from 100
# to 400 nits
HDR_SDR_ratio = HDR_max_nits / SDR_max_nits

# This would be a checkbox in a dynamically tunable GUI interface
Use_HDR = True

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
# Apr 4, 2025: The parameters in the above comments were calculated with Rec.709 primaries in the source script but got used in Rec.2020
#              To achieve the same matrix calculating in Rec.2020 in the working_space.py script, use this parameter in Rec.2020: 
#              Rotations: [ 2.13976149 -1.22827335 -3.05174246], Inset: [ 0.32965205  0.28051336  0.12475368]
#              These parameters calculating in Rec.2020 should result in almost, if not exactly, the same matrix as the old parameters calculated in Rec.709
inset_matrix = numpy.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                            [0.137318972929847, 0.761241990602591, 0.101439036467562],
                            [0.11189821299995, 0.0767994186031903, 0.811302368396859]])

additional_inset_scaling = math.log(HDR_SDR_ratio, 10) * 0.15
additional_inset_space = working_space.create_workingspace(primaries_rotate=[0, 0, 0],
                                                           primaries_scale=[additional_inset_scaling, additional_inset_scaling, additional_inset_scaling],
                                                           colourspace_in=colour.RGB_COLOURSPACES["ITU-R BT.2020"])
additional_inset_for_hdr_appearance_matching = colour.RGB_COLOURSPACES["ITU-R BT.2020"].matrix_XYZ_to_RGB @ additional_inset_space.matrix_RGB_to_XYZ

# outset matrix from Troy's SB2383 script, setting is rotate = [0, 0, 0] inset = [0.4, 0.22, 0.04], used on inverse
# link to the script: https://github.com/sobotka/SB2383-Configuration-Generation/blob/main/generate_config.py
# the relevant part is at line 88 and 89
# Apr 4, 2025: The parameters in the above comments were calculated with Rec.709 primaries in the source script but got used in Rec.2020
#              To achieve the same matrix calculating in Rec.2020 in the working_space.py script, use this parameter in Rec.2020: 
#              Rotations: [ 0 0 0], Inset: [ 0.32317438  0.28325605  0.0374326 ]
#              These parameters calculating in Rec.2020 should result in almost, if not exactly, the same matrix as the old parameters calculated in Rec.709
outset_matrix = numpy.linalg.inv(numpy.array([[0.899796955911611, 0.0871996192028351, 0.013003424885555],
                                              [0.11142098895748, 0.875575586156966, 0.0130034248855548],
                                              [0.11142098895748, 0.0871996192028349, 0.801379391839686]]))

# these lines are dependencies from Troy's AgX script
x_pivot = numpy.abs(normalized_log2_minimum) / (
        normalized_log2_maximum - normalized_log2_minimum
)


# define middle grey
y_pivot = 0.18 ** (1.0 / 2.4)

exponent = [1.5, 1.5]
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

def lerp_chromaticity_angle(h1: float, h2: float, t: float) -> float:
    """Circular h interpolation using shortest path
    Args:
        h1: Start h (0-1)
        h2: End h (0-1)
        t: Mixing ratio (0-1)
    Returns:
        Interpolated h (0-1)
    """
    delta = h2 - h1
    if delta > 0.5:
        h2 -= 1.0  # Go the reverse direction
    elif delta < -0.5:
        h2 += 1.0  # Go the reverse direction
    lerped = h1 + t * (h2 - h1)
    return lerped % 1.0  # Wrap around at 1.0

def darken_middle_grey(col):
    temp_log = colour.log_encoding(col,
                              function='Log2',
                              min_exposure=-20,
                              max_exposure=math.log(1/0.18, 2),
                              middle_grey=0.18)

    original_middle_grey = colour.log_encoding(0.18,
                              function='Log2',
                              min_exposure=-20,
                              max_exposure=math.log(1/0.18, 2),
                              middle_grey=0.18)

    darkened_middle_grey = colour.log_encoding(0.18/HDR_SDR_ratio,
                              function='Log2',
                              min_exposure=-20,
                              max_exposure=math.log(1/0.18, 2),
                              middle_grey=0.18)

    darkened = sigmoid.calculate_sigmoid(
        temp_log,
        pivots=[original_middle_grey, darkened_middle_grey],
        slope=1,
        powers=[1, 1],
    )

    darkened_linear_image = colour.log_decoding(darkened,
                              function='Log2',
                              min_exposure=-20,
                              max_exposure=math.log(1/0.18, 2),
                              middle_grey=0.18)
    return darkened_linear_image

def AgX_Base_Rec2020(col, mix_percent):
    # apply lower guard rail
    col = lu2020.compensate_low_side(col)

    # apply inset matrix
    col = numpy.tensordot(col, inset_matrix, axes=(0, 1))

    if Use_HDR == True:
        col = numpy.tensordot(col, additional_inset_for_hdr_appearance_matching, axes=(0, 1))

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
    col = colour.models.exponent_function_basic(col, 2.4, 'basicFwd')

    if Use_HDR == True:
        col = darken_middle_grey(col)

    # record post-sigmoid chroma angle
    col = colour.RGB_to_HSV(col)

    col[0] = lerp_chromaticity_angle(pre_form_hsv[0], col[0], mix_percent / 100)

    col = colour.HSV_to_RGB(col)

    # apply outset to make the result more chroma-laden
    col = numpy.tensordot(col, outset_matrix, axes=(0, 1))

    return col

colour.utilities.filter_warnings(python_warnings=True)


def main():
    # resolution of the 3D LUT
    LUT_res = 57

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

    LUT = colour.LUT3D(name=f'AgX_Formation Rec.2100 HLG P3 limited',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [f'AgX Base Rec.2100 HLG Formation P3 Limited LUT',
                    f'This LUT expects input to be E Gamut Log2 encoding from -10 stops to +15 stops',
                    f'But the end image formation will be from {normalized_log2_minimum} to {normalized_log2_maximum} encoded in power 2.4',
                    f'Rec.709 generated parameters rotate = [3.0, -1, -2.0], inset = [0.4, 0.22, 0.13], outset = [0.4, 0.22, 0.04]',
                    f' Same inset matrix can be generated in Rec.2020 with rotation [2.13976149, -1.22827335, -3.05174246], Inset: [0.32965205, 0.28051336, 0.12475368], outset = [0.32317438, 0.28325605, 0.0374326]',
                    f'The image formed has {mix_percent}% per-channel shifts',
                    f'HDR max nits is {HDR_max_nits}, SDR max nits is {SDR_max_nits}']

    x, y, z, _ = LUT.table.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                col = numpy.array(LUT.table[i][j][k], dtype=numpy.longdouble)

                # decode LUT input transfer function
                col = colour.log_decoding(col,
                                          function='Log2',
                                          min_exposure=-10,
                                          max_exposure=+15,
                                          middle_grey=midgrey)

                # decode LUT input primaries from E-Gamut to Rec.2020
                col = numpy.tensordot(col, lu2020.e_gamut_to_xyz_id65, axes=(0, 1))

                col = numpy.tensordot(col, lu2020.xyz_id65_to_bt2020_id65, axes=(0, 1))

                col = AgX_Base_Rec2020(col, mix_percent)

                # Apply P3 Lower Rail for P3 limited output, mute these lines below for full Rec.2020 gamut output
                # or unmute the lines below if you want to limit output to P3 gamut.
                col = numpy.tensordot(col, lu2020.bt2020_id65_to_xyz_id65, axes=(0, 1))
                col = numpy.tensordot(col, lup3.xyz_id65_to_p3_id65, axes=(0, 1))
                col = lup3.compensate_low_side(col)
                col = numpy.tensordot(col, lup3.p3_id65_to_xyz_id65, axes=(0, 1))
                col = numpy.tensordot(col, lu2020.xyz_id65_to_bt2020_id65, axes=(0, 1))

                # re-encode transfer function
                col = colour.models.eotf_inverse_BT2100_HLG(col * HDR_max_nits)
                col = numpy.clip(col, a_min=0, a_max=1)
                LUT.table[i][j][k] = numpy.array(col, dtype=LUT.table.dtype)

    LUT_name = f"AgX_Base_Rec2100-HLG_p3_lim.cube"
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
