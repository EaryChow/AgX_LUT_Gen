import colour
import numpy
import luminance_compenstation_bt2020 as lu2020
import luminance_compenstation_p3 as lup3
import AgXBaseRec2020
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

inset_matrix = numpy.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                            [0.137318972929847, 0.761241990602591, 0.101439036467562],
                            [0.11189821299995, 0.0767994186031903, 0.811302368396859]])

outset_matrix = numpy.linalg.inv(numpy.array([[0.899796955911611, 0.0871996192028351, 0.013003424885555],
                                              [0.11142098895748, 0.875575586156966, 0.0130034248855548],
                                              [0.11142098895748, 0.0871996192028349, 0.801379391839686]]))


colour.utilities.filter_warnings(python_warnings=True)


def main():
    # resolution of the 3D LUT
    LUT_res = 37

    mix_percent = 40

    LUT = colour.LUT3D(name=f'AgX_Formation P3',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [f'AgX Base P3 Formation LUT',
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

                # form the image in Rec.2020
                col = AgXBaseRec2020.AgX_Base_Rec2020(col, mix_percent)

                # convert formed image to P3
                col = numpy.tensordot(col, lu2020.bt2020_id65_to_xyz_id65, axes=(0, 1))

                col = numpy.tensordot(col, lup3.xyz_id65_to_p3_id65, axes=(0, 1))

                # apply P3's lower Guard Rail
                col = lup3.compensate_low_side(col)

                # should have applied the higher guard rail here, but the higher guard rail actually caused an artifact
                # in higher exposure, a test file can be found here: https://discuss.pixls.us/t/sunflower-sagas-and-solutions/33292
                # download the "black woman with orange paint on their face" CR2 file, use Darktable to convert to EXR
                # import to Blender to test, set exposure to +8, you should see the relative figure-ground luminance relationship
                # gets flipped after applying higher guard rail here.

                # col = high_rail.compensate_high_side(col, high_rail.luminance_coeffs, high_rail.high_side_compensation)

                # encode transfer function, having an output function helps with precision, so use the same one as the Rec.2020 version
                col = colour.models.exponent_function_basic(col, 2.4, 'basicRev')

                LUT.table[i][j][k] = numpy.array(col, dtype=LUT.table.dtype)

    colour.write_LUT(
        LUT,
        f"AgX_Base_P3.cube")
    print(LUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
