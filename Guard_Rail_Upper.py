import colour
import sys
import numpy as np

rgb_src = colour.RGB_COLOURSPACES["ITU-R BT.2020"]
rgb_dst = colour.RGB_COLOURSPACES["ITU-R BT.709"]
src_to_dst = rgb_dst.matrix_XYZ_to_RGB @ rgb_src.matrix_RGB_to_XYZ
dst_to_src = rgb_src.matrix_XYZ_to_RGB @ rgb_dst.matrix_RGB_to_XYZ
luminance_coeffs = np.array([0.2658180370250449, 0.59846986045365, 0.1357121025213052])


def compensate_high_side_lum(rgb, luminance_coeffs, compensation_factor):
    upper_bound = 1.0
    luminance = np.dot(np.matmul(dst_to_src, rgb), luminance_coeffs)
    # zero-luminance "colour" part of input signal
    rgb_chrominance = rgb - luminance[np.newaxis]
    # Chrominance is max(rgb) - luminance
    chrominance = np.amax(rgb_chrominance)

    # define relative_luminance to be luminance based
    relative_luminance = luminance * compensation_factor

    # Coefficient by how much the chrominance deviates from
    # the line relative_luminance = relative_chrominance
    chrominance_coefficient = np.where(
        relative_luminance > upper_bound,
        (upper_bound / relative_luminance) ** compensation_factor, 1.0)

    # Adjust chrominance by the calculated coefficient and calculate the max RGB
    # that would result from this.
    new_max_rgb = luminance + chrominance_coefficient * chrominance

    # use max rgb to make sure we can do another higher rail later
    return np.amax(rgb)[np.newaxis] * (
            luminance[np.newaxis]
            + rgb_chrominance * chrominance_coefficient[np.newaxis]
    )


def compensate_high_side_inten(rgb, luminance_coeffs, compensation_factor):
    upper_bound = 1.0
    luminance = np.dot(np.matmul(dst_to_src, rgb), luminance_coeffs)
    # zero-luminance "colour" part of input signal
    rgb_chrominance = rgb - luminance[np.newaxis]
    # Chrominance is max(rgb) - luminance
    chrominance = np.amax(rgb_chrominance)

    # define relative_luminance to be intensity based
    relative_luminance = np.amax(rgb)

    # Coefficient by how much the chrominance deviates from
    # the line relative_luminance = relative_chrominance
    chrominance_coefficient = np.where(
        relative_luminance > upper_bound,
        (upper_bound / relative_luminance) ** compensation_factor, 1.0)

    # Adjust chrominance by the calculated coefficient and calculate the max RGB
    # that would result from this.
    new_max_rgb = luminance + chrominance_coefficient * chrominance

    # Calculate scaling of the RGB triplet that must be done to
    # bring the greatest component to upper_bound.
    scale = np.where(new_max_rgb > (upper_bound + 0),
                     (upper_bound + 0) / new_max_rgb, 1.0)

    # use scale here to bring the greatest component to upper_bound.
    return scale[np.newaxis] * (
            luminance[np.newaxis]
            + rgb_chrominance * chrominance_coefficient[np.newaxis]
    )


def main():
    # resolution of the 3DLUT
    LUT_res = 37
    LUT = colour.LUT3D(name=f'Upper Guard Rail',
                       size=LUT_res)

    LUT.domain = ([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    LUT.comments = [
        f'Higher Side of Guard Rail Formation',
        f'The Guard Rail is designed to leaves valid value as-is, and only touch out of domain values',
        f'Modified from python script shared by Sakari Kapanen',
        f'This LUT assumes target medium primaries encoded in ST2084 (100) curve and then power 0.6',
        f'LUT resolution {LUT_res}']

    x, y, z, _ = LUT.table.shape

    for i in range(x):
        for j in range(y):
            for k in range(z):
                col = np.array(LUT.table[i][j][k], dtype=np.longdouble)

                if len(sys.argv) > 2:
                    high_side_compensation = float(sys.argv[2])
                else:
                    # define compensate factor
                    high_side_compensation = 1.2

                # record input original
                original = col

                # decode the input transfer function
                col = colour.models.exponent_function_basic(col, 0.6, "basicRev")

                col = colour.models.eotf_ST2084(col, 100)

                # record values that are outside the target's [0, 1] range
                invalid = (np.max(col) < 0.0) or (np.max(col) > 1.0)

                # apply luminance based attenuation, but the region between max(rgb) = 1.0 and luminance = 1.0 will be untouched
                col = compensate_high_side_lum(col, luminance_coeffs, high_side_compensation)
                # apply intensity based attenuation, to make sure the region between max(rgb) = 1.0 and luminance = 1.0 will be touched
                col = compensate_high_side_inten(col, luminance_coeffs, high_side_compensation / 6)

                # re-encode transfer function
                col = colour.models.eotf_inverse_ST2084(col, 100)

                col = colour.models.exponent_function_basic(col, 0.6, "basicFwd")

                # if within valid [0, 1] range, output untouched original
                col = np.where(invalid, col, original)

                LUT.table[i][j][k] = np.array(col, dtype=LUT.table.dtype)
    colour.write_LUT(
        LUT,
        f"guard_rail_higher.cube")
    print(LUT)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
