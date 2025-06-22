import colour
import numpy as np
import numpy

numpy.set_printoptions(suppress=True, precision=128)
def new_colourspace(base_colourspace, primaries, whitepoint, name, whitepoint_name):
    colourspace = colour.RGB_COLOURSPACES[base_colourspace].copy()
    colourspace.name = name
    colourspace.primaries = primaries
    colourspace.whitepoint = whitepoint
    colourspace.whitepoint_name = whitepoint_name
    colourspace.use_derived_matrix_RGB_to_XYZ = True
    colourspace.use_derived_matrix_XYZ_to_RGB = True
    colourspace.use_derived_transformation_matrices = True
    colourspace.matrix_RGB_to_XYZ = colourspace.matrix_RGB_to_XYZ
    colourspace.matrix_XYZ_to_RGB = colourspace.matrix_XYZ_to_RGB
    return colourspace


def main():
    D65sd = colour.SDS_ILLUMINANTS['D65']
    D65_blackbody6504k_sd = colour.sd_blackbody(6504, shape=colour.SpectralShape(300,900,1))
    cmfs = colour.MSDS_CMFS['CIE 2015 10 Degree Standard Observer']
    D65_XYZ_2015_10d = colour.sd_to_XYZ(D65sd, cmfs)
    D65_xy_2015_10d = D65_XYZ_2015_10d[:2] / np.sum(D65_XYZ_2015_10d)
    D65_xy_2015_10d_rounded = np.round(D65_xy_2015_10d, 4)
    # BT.2020 uses 630nm red, 532nm green, and 467nm blue
    primaries = colour.wavelength_to_XYZ(np.array([630, 532, 467]), cmfs)
    primaries /= np.sum(primaries, axis=1, keepdims=True)
    primaries = primaries[:, :2]
    whitepoint = D65_xy_2015_10d_rounded
    colourspacename = 'BT.2020 I-D65 2015 10Deg'
    whitepoint_name = 'D65'

    colourspace_2015_BT2020_ID65 = new_colourspace('ITU-R BT.2020', primaries, whitepoint, colourspacename, whitepoint_name)
    print(colourspace_2015_BT2020_ID65)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
