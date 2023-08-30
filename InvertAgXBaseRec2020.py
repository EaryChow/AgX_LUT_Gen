import colour
import numpy

# Log range parameters
midgrey = 0.18
normalized_log2_minimum = -10
normalized_log2_maximum = +6.5

# define color space matrices
inset_matrix = numpy.array([[0.856627153315983, 0.0951212405381588, 0.0482516061458583],
                            [0.137318972929847, 0.761241990602591, 0.101439036467562],
                            [0.11189821299995, 0.0767994186031903, 0.811302368396859]])

outset_matrix = numpy.linalg.inv(numpy.array([[0.899796955911611, 0.0871996192028351, 0.013003424885555],
                                              [0.11142098895748, 0.875575586156966, 0.0130034248855548],
                                              [0.11142098895748, 0.0871996192028349, 0.801379391839686]]))

default_contrast = colour.io.read_LUT_SonySPI1D('AgX_Default_Contrast.spi1d')
inverse = default_contrast.invert(extrapolate=True, interpolator='Tetrahedral')


def apply_inverse_sigmoid(x):
    col = inverse.apply(x)

    return col


colour.utilities.filter_warnings(python_warnings=True)


def main():
    # resolution of the 3DLUT
    LUT_res = 36

    mix_percent = 40

    LUT = colour.LUT3D(name=f'Inverse AgX_Formation Rec.2020',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [f'Inverse of AgX Base Rec.2020 Formation LUT',
                    f'This LUT expects input to be Rec.2020 power 2.4, output to be Rec.2020 Log2 from -10 stops to +6.5 stops']

    x, y, z, _ = LUT.table.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                col = numpy.array(LUT.table[i][j][k], dtype=numpy.longdouble)

                # decode Rec.2020 image's power 2.4 transfer function
                col = colour.models.exponent_function_basic(col, 2.4, 'basicFwd')

                # inverse the outset matrix
                col = numpy.tensordot(col, numpy.linalg.inv(outset_matrix), axes=(0, 1))

                # re-encode transfer function
                col = colour.models.exponent_function_basic(col, 2.4, 'basicRev')

                # record formed image's chroma angle
                formed_hsv = colour.RGB_to_HSV(col)

                # inverse sigmoid
                col = apply_inverse_sigmoid(col)

                # decode Log2 curve
                col = colour.log_decoding(col,
                                          function='Log2',
                                          min_exposure=-10,
                                          max_exposure=+6.5,
                                          middle_grey=midgrey)

                # re-encode transfer function
                col = colour.models.exponent_function_basic(col, 2.4, 'basicRev')

                # record inverted result's chroma angle
                inverted_hsv = colour.RGB_to_HSV(col)

                # mix pre-invert and post-invert chroma angle
                inverted_hsv[0] = colour.algebra.lerp(mix_percent/100, formed_hsv[0], inverted_hsv[0], False)

                col = colour.HSV_to_RGB(inverted_hsv)

                # decode power 2.4 transfer function
                col = colour.models.exponent_function_basic(col, 2.4, 'basicFwd')

                # invert the inset matrix
                col = numpy.tensordot(col, numpy.linalg.inv(inset_matrix), axes=(0, 1))

                # re-encode log2 curve
                col = colour.log_encoding(col,
                                          function='Log2',
                                          min_exposure=-10,
                                          max_exposure=+6.5,
                                          middle_grey=midgrey)

                LUT.table[i][j][k] = numpy.array(col, dtype=LUT.table.dtype)

    colour.write_LUT(
        LUT,
        f"Inverse_AgX_Base_Rec2020.cube")
    print(LUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
