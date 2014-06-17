"""
Test image pre-processing functions
"""
from nose.tools import assert_true, assert_false

import nibabel
import numpy as np
from numpy.testing import assert_array_equal

from .. import image
from .. import resampling
from ..._utils import testing

def test_high_variance_confounds():
    # See also test_signals.test_high_variance_confounds()
    # There is only tests on what is added by image.high_variance_confounds()
    # compared to signal.high_variance_confounds()

    shape = (40, 41, 42)
    length = 17
    n_confounds = 10

    img, mask_img = testing.generate_fake_fmri(shape=shape, length=length)

    confounds1 = image.high_variance_confounds(img, mask_img=mask_img,
                                               percentile=10.,
                                               n_confounds=n_confounds)
    assert_true(confounds1.shape == (length, n_confounds))

    # No mask.
    confounds2 = image.high_variance_confounds(img, percentile=10.,
                                               n_confounds=n_confounds)
    assert_true(confounds2.shape == (length, n_confounds))


def test__smooth_array():
    """Test smoothing of images: _smooth_array()"""
    # Impulse in 3D
    data = np.zeros((40, 41, 42))
    data[20, 20, 20] = 1

    # fwhm divided by any test affine must be odd. Otherwise assertion below
    # will fail. ( 9 / 0.6 = 15 is fine)
    fwhm = 9
    test_affines = (np.eye(4), np.diag((1, 1, -1, 1)),
                    np.diag((.6, 1, .6, 1)))
    for affine in test_affines:
        filtered = image._smooth_array(data, affine,
                                         fwhm=fwhm, copy=True)
        assert_false(np.may_share_memory(filtered, data))

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = filtered.max()
        above_half_max = filtered > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                          axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                                    fwhm / np.abs(affine[axis, axis]))

    # Check that NaNs in the data do not propagate
    data[10, 10, 10] = np.NaN
    filtered = image._smooth_array(data, affine, fwhm=fwhm,
                                   ensure_finite=True, copy=True)
    assert_true(np.all(np.isfinite(filtered)))

    # Check copy=False.
    for affine in test_affines:
        data = np.zeros((40, 41, 42))
        data[20, 20, 20] = 1
        image._smooth_array(data, affine, fwhm=fwhm, copy=False)

        # We are expecting a full-width at half maximum of
        # fwhm / voxel_size:
        vmax = data.max()
        above_half_max = data > .5 * vmax
        for axis in (0, 1, 2):
            proj = np.any(np.any(np.rollaxis(above_half_max,
                          axis=axis), axis=-1), axis=-1)
            np.testing.assert_equal(proj.sum(),
                                    fwhm / np.abs(affine[axis, axis]))


def test_smooth_img():
    # This function only checks added functionalities compared
    # to _smooth_array()
    shapes = ((10, 11, 12), (13, 14, 15))
    lengths = (17, 18)
    fwhm = (1., 2., 3.)

    img1, mask1 = testing.generate_fake_fmri(shape=shapes[0],
                                             length=lengths[0])
    img2, mask2 = testing.generate_fake_fmri(shape=shapes[1],
                                             length=lengths[1])

    for create_files in (False, True):
        with testing.write_tmp_imgs(img1, img2,
                                    create_files=create_files) as imgs:
            # List of images as input
            out = image.smooth_img(imgs, fwhm)
            assert_true(isinstance(out, list))
            assert_true(len(out) == 2)
            for o, s, l in zip(out, shapes, lengths):
                assert_true(o.shape == (s + (l,)))

            # Single image as input
            out = image.smooth_img(imgs[0], fwhm)
            assert_true(isinstance(out, nibabel.Nifti1Image))
            assert_true(out.shape == (shapes[0] + (lengths[0],)))


def test__crop_img_to():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    niimg = nibabel.Nifti1Image(data, affine=affine)

    slices = [slice(2, 4), slice(1, 5), slice(3, 6)]
    cropped_niimg = image._crop_img_to(niimg, slices, copy=False)

    new_origin = np.array((4, 3, 2)) * np.array((2, 1, 3))

    # check that correct part was extracted:
    assert_true((cropped_niimg.get_data() == 1).all())
    assert_true(cropped_niimg.shape == (2, 4, 3))

    # check that affine was adjusted correctly
    assert_true((cropped_niimg.get_affine()[:3, 3] == new_origin).all())

    # check that data was really not copied
    data[2:4, 1:5, 3:6] = 2
    assert_true((cropped_niimg.get_data() == 2).all())

    # check that copying works
    copied_cropped_niimg = image._crop_img_to(niimg, slices)
    data[2:4, 1:5, 3:6] = 1
    assert_true((copied_cropped_niimg.get_data() == 2).all())


def test_crop_img():
    data = np.zeros((5, 6, 7))
    data[2:4, 1:5, 3:6] = 1
    affine = np.diag((4, 3, 2, 1))
    niimg = nibabel.Nifti1Image(data, affine=affine)

    cropped_niimg = image.crop_img(niimg)

    # correction for padding with "-1"
    new_origin = np.array((4, 3, 2)) * np.array((2 - 1, 1 - 1, 3 - 1))

    # check that correct part was extracted:
    # This also corrects for padding
    assert_true((cropped_niimg.get_data()[1:-1, 1:-1, 1:-1] == 1).all())
    assert_true(cropped_niimg.shape == (2 + 2, 4 + 2, 3 + 2))


def test_crop_threshold_tolerance():
    """Check to see whether crop can skip values that are extremely
    close to zero in a relative sense and will crop them away"""

    data = np.zeros([10, 14, 12])
    data[3:7, 3:7, 5:9] = 1.
    active_shape = (4 + 2, 4 + 2, 4 + 2)  # add padding

    # add an infinitesimal outside this block
    data[3, 3, 3] = 1e-12
    affine = np.eye(4)
    niimg = nibabel.Nifti1Image(data, affine=affine)

    cropped_niimg = image.crop_img(niimg)
    assert_true(cropped_niimg.shape == active_shape)


def test_mean_img():
    rng = np.random.RandomState(42)
    data1 = np.zeros((5, 6, 7))
    data2 = rng.rand(5, 6, 7)
    data3 = rng.rand(5, 6, 7, 3)
    affine = np.diag((4, 3, 2, 1))
    niimg1 = nibabel.Nifti1Image(data1, affine=affine)
    niimg2 = nibabel.Nifti1Image(data2, affine=affine)
    niimg3 = nibabel.Nifti1Image(data3, affine=affine)
    for niimgs in ([niimg1, ],
                   [niimg1, niimg2],
                   [niimg2, niimg1, niimg2],
                   [niimg3, niimg1, niimg2],  # Mixture of 4D and 3D images
                  ):

        arrays = list()
        # Ground-truth:
        for niimg in niimgs:
            niimg = niimg.get_data()
            if niimg.ndim == 4:
                niimg = np.mean(niimg, axis=-1)
            arrays.append(niimg)
        truth = np.mean(arrays, axis=0)

        mean_img = image.mean_img(niimgs)
        assert_array_equal(mean_img.get_affine(), affine)
        assert_array_equal(mean_img.get_data(), truth)

        # Test with files
        with testing.write_tmp_imgs(*niimgs) as imgs:
            mean_img = image.mean_img(imgs)
            assert_array_equal(mean_img.get_affine(), affine)
            assert_array_equal(mean_img.get_data(), truth)


def test_mean_img_resample():
    # Test resampling in mean_img with a permutation of the axes
    rng = np.random.RandomState(42)
    data = rng.rand(5, 6, 7, 40)
    affine = np.diag((4, 3, 2, 1))
    niimg = nibabel.Nifti1Image(data, affine=affine)
    mean_img = nibabel.Nifti1Image(data.mean(axis=-1), affine=affine)

    target_affine = affine[:, [1, 0, 2, 3]]  # permutation of axes
    mean_img_with_resampling = image.mean_img(niimg,
                                              target_affine=target_affine)
    resampled_mean_image = resampling.resample_img(mean_img,
                                              target_affine=target_affine)
    assert_array_equal(resampled_mean_image.get_data(),
                       mean_img_with_resampling.get_data())
    assert_array_equal(resampled_mean_image.get_affine(),
                       mean_img_with_resampling.get_affine())
    assert_array_equal(mean_img_with_resampling.get_affine(), target_affine)

