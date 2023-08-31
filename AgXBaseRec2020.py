import colour
import numpy
import sigmoid
import argparse
import luminance_compenstation_bt2020 as lu2020
import Guard_Rail_Upper as high_rail

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
inset_matrix = numpy.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                            [0.137318972929847, 0.761241990602591, 0.101439036467562],
                            [0.11189821299995, 0.0767994186031903, 0.811302368396859]])

# outset matrix from Troy's SB2383 script, setting is rotate = [0, 0, 0] inset = [0.4, 0.22, 0.04], used on inverse
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
    col = colour.models.exponent_function_basic(col, 2.4, 'basicFwd')

    # record post-sigmoid chroma angle
    col = colour.RGB_to_HSV(col)

    # mix pre-formation chroma angle with post formation chroma angle.
    col[0] = colour.algebra.lerp(mix_percent / 100, pre_form_hsv[0], col[0], False)

    col = colour.HSV_to_RGB(col)

    # apply outset to make the result more chroma-laden
    col = numpy.tensordot(col, outset_matrix, axes=(0, 1))

    return col


colour.utilities.filter_warnings(python_warnings=True)


def main():
    # resolution of the 3D LUT
    LUT_res = 37

    mix_percent = 40

    LUT = colour.LUT3D(name=f'AgX_Formation Rec.2020',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [f'AgX Base Rec.2020 Formation LUT',
                    f'This LUT expects input to be E Gamut Log2 encoding from -10 stops to +15 stops',
                    f'But the end image formation will be from {normalized_log2_minimum} to {normalized_log2_maximum} encoded in power 2.4',
                    f' rotate = [3.0, -1, -2.0], inset = [0.4, 0.22, 0.13], outset = [0.4, 0.22, 0.04]',
                    f'The image formed has {mix_percent}% per-channel shifts']

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

                # re-encode transfer function
                col = colour.models.exponent_function_basic(col, 2.4, 'basicRev')

                LUT.table[i][j][k] = numpy.array(col, dtype=LUT.table.dtype)

    colour.write_LUT(
        LUT,
        f"AgX_Base_Rec2020.cube")
    print(LUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
