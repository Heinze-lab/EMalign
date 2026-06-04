import logging
import cv2
import numpy as np

from emalign.io.process.transform import rotate_image
from emalign.io.process.mask import mask_to_bbox

from ..io.process.mask import compute_greyscale_mask  
from .sift import estimate_transform_sift
from .utils import compute_laplacian_var_diff, homogenize_arrays_shape, resample, transform_bbox, xy_offset_to_pad


def get_overlap(img1, img2, offset, rotation=0, pad=0, homogenize_shapes=False):
    '''
    Extract overlapping parts of two images based on an offset and rotation from img2 to img1.

    Parameters:
    -----------
    img1 : np.ndarray
        Reference image
    img2 : np.ndarray
        Moving image to align with img1
    offset : tuple or array-like
        (x, y) displacement of img2 relative to img1
    rotation : float, optional
        Rotation angle in degrees to apply to img2 before computing overlap. Default: 0
    pad : int, optional
        Additional padding to extend the overlap region. Default: 0
    homogenize_shapes : bool, optional
        If True, ensures both cropped regions have identical shapes by trimming to minimum.
        Default: False

    Returns:
    --------
    tuple of np.ndarray or None
        (crop1, crop2) - Overlapping regions from img1 and img2 respectively.
        Returns None if images don't overlap.

    Notes:
    ------
    The offset describes how img2 is positioned relative to img1:
    - Positive x offset: img2 is to the right of img1
    - Negative x offset: img2 is to the left of img1
    - Positive y offset: img2 is below img1
    - Negative y offset: img2 is above img1
    '''
        
    if rotation != 0:
        if img2.dtype == bool:
            img2 = rotate_image(img2.astype(np.uint8), rotation).astype(bool)
        else:
            img2 = rotate_image(img2, rotation)
    
    offset = offset[::-1]
    if offset[1] > 0:
        ox = img2.shape[1] - int(abs(offset[1])) + pad
        crop2 = img2[:, -ox:]
        crop1 = img1[:, :ox]
    else:
        ox = img1.shape[1] - int(abs(offset[1])) + pad
        crop1 = img1[:, -ox:]
        crop2 = img2[:, :ox]

    if offset[0] < 0:
        oy = img1.shape[0] - int(abs(offset[0])) + pad
        crop1 = crop1[-oy:, :]
        crop2 = crop2[:oy, :]
    else:
        oy = img2.shape[0] - int(abs(offset[0])) + pad
        crop1 = crop1[:oy, :]
        crop2 = crop2[-oy:, :]

    if homogenize_shapes:
        y, x = np.min([crop1.shape, crop2.shape], axis=0)
        crop1 = crop1[:y,:x]
        crop2 = crop2[:y,:x]

    return crop1, crop2


def get_overlap_warp(ref_img, mov_img, ref_mask, mov_mask, M, mov_img_shape, ref_img_offset):
    '''
    Extract overlapping regions after applying affine transformation and alignment.

    This function warps images and masks using an affine transformation matrix,
    pads them to match alignment offsets, homogenizes shapes, and extracts the
    overlapping region based on the intersection of masks.

    Parameters:
    -----------
    ref_img : np.ndarray
        Reference image
    mov_img : np.ndarray
        Moving image to be warped and aligned
    ref_mask : np.ndarray (bool)
        Boolean mask indicating valid regions in reference image
    mov_mask : np.ndarray (bool)
        Boolean mask indicating valid regions in moving image
    M : np.ndarray
        2x3 affine transformation matrix for warping
    mov_img_shape : tuple
        Target shape (height, width) for the warped moving image
    ref_img_offset : tuple or array-like
        (x, y) offset to pad the reference image for alignment

    Returns:
    --------
    tuple of np.ndarray
        (ref_crop, mov_crop) - Cropped overlapping regions from reference and moving images,
        bounded by the intersection of both masks.

    Notes:
    ------
    - The moving image is warped using the affine transformation M
    - The reference mask is also warped to match the moving image coordinate system
    - Both images are padded and resized to have identical shapes
    - The final overlap is determined by the bounding box of (ref_mask & mov_mask)
    '''

    mov_img = cv2.warpAffine(mov_img, M, mov_img_shape[::-1])
    ref_mask = cv2.warpAffine(ref_mask.astype(np.uint8), M, mov_img_shape[::-1]).astype(bool)

    # Pad moving image so it matches the reference
    ref_img = np.pad(ref_img, xy_offset_to_pad(ref_img_offset))
    mov_mask = np.pad(mov_mask, xy_offset_to_pad(ref_img_offset))

    # Make sure that images have the same shape for sofima
    mov_img, ref_img = homogenize_arrays_shape([mov_img, ref_img])
    ref_mask, mov_mask = homogenize_arrays_shape([ref_mask, mov_mask])

    mask = ref_mask & mov_mask
    y1,y2,x1,x2 = mask_to_bbox(mask)

    return ref_img[y1:y2, x1:x2], mov_img[y1:y2, x1:x2]


def check_overlap(img1,
                  img2,
                  xy_offset,
                  theta,
                  threshold=0.5,
                  scale=(0.3, 0.5),
                  refine=True):

    '''
    Compute a quality metric describing how well two images overlap based on edge content similarity.

    This function uses Laplacian variance to assess the similarity of edge content in the
    overlapping regions of two images. Optionally, it can refine the alignment using SIFT
    feature matching if the initial overlap quality is below a threshold.

    Parameters:
    -----------
    img1 : np.ndarray
        Reference image
    img2 : np.ndarray
        Moving image to compare with img1
    xy_offset : tuple or array-like
        (x, y) offset describing the displacement of img2 relative to img1
    theta : float
        Rotation angle in degrees to apply to img2
    threshold : float, optional
        Quality threshold below which refinement is triggered if refine is True. Range [0,1].
        Default: 0.5
    scale : tuple of float, optional
        Two scale factors to try for SIFT-based refinement. Default: (0.3, 0.5)
    refine : bool, optional
        If True, attempts to refine the overlap using SIFT when quality is below threshold.
        Default: True

    Returns:
    --------
    float
        Overlap quality metric in range [0,1]:
        - 1.0: Perfect overlap, identical edge content
        - 0.0: No overlap or completely different edge content
        - Values closer to 1 indicate better alignment

    Notes:
    ------
    - Uses Laplacian variance to measure edge content similarity
    - If refinement is enabled and quality < threshold, attempts SIFT-based realignment
    - Tries two different scale factors for SIFT refinement
    - Returns 0 if images don't overlap geometrically
    '''

    # Index of sharpness using Laplacian
    overlap = get_overlap(img1, img2, xy_offset, theta)

    if overlap is not None:
        overlap1, overlap2 = overlap

        lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask=None)

        if refine and lap_variance_diff < threshold:
            logging.debug('Refining overlap estimation...')
            # Retry the overlap, it can often get better
            try:
                refined_offset, refined_theta = estimate_transform_sift(overlap1, overlap2, scale=scale[0])[:2]
            except Exception as e:
                logging.debug(f'First scale failed: {e}, trying second scale')
                try:
                    refined_offset, refined_theta = estimate_transform_sift(overlap1, overlap2, scale=scale[1])[:2]
                except Exception as e:
                    logging.debug(f'Second scale refinement failed: {e}')
                    return lap_variance_diff

            res = get_overlap(overlap1, overlap2, refined_offset, refined_theta)

            if res is not None:
                overlap1, overlap2 = res
                lap_variance_diff = compute_laplacian_var_diff(overlap1, overlap2, mask=None)
            else:
                lap_variance_diff = 0
    else:
        # Images do not overlap (displacement is larger than image itself)
        lap_variance_diff = 0

    return lap_variance_diff


def get_overlap_z(img, bbox, mask=None, compute_mask=False):

    if np.any(np.array(bbox) < 0):
        # Pad the image
        pad_y = np.min(bbox[:2])
        pad_x = np.min(bbox[2:])
        pad = [[pad_y, 0], [pad_x, 0]]
        img = np.pad(img, pad)
        y1, y2, x1, x2 = np.array(bbox) - np.array(pad)
    else:
        y1, y2, x1, x2 = bbox

    overlap_img = img[y1:y2, x1:x2]

    if mask is None and compute_mask:
        # Compute mask if necessary (from cropped to save time)
        overlap_mask = compute_greyscale_mask(overlap_img)
    else:
        overlap_mask = mask[y1:y2, x1:x2]

    return overlap_img, overlap_mask


def get_overlap_ref(ref_img, 
                    mov_img, 
                    ref_mask=None, 
                    mov_mask=None,
                    bbox_ref=None,
                    pad_overlap=100,
                    return_sift=False):

    if bbox_ref is None:
        # Estimate transform between ref and mov to figure out where the overlap is
        M, output_shape, ref_offset, valid_estimate, stats = estimate_transform_sift(ref_img, mov_img, 0.1, refine_estimate=True)
        if not valid_estimate:
            M, output_shape, ref_offset, valid_estimate, stats = estimate_transform_sift(ref_img, mov_img, 0.3, refine_estimate=True)

        # Get the overlapping region in the reference image
        if mov_mask is None:
            mov_mask = compute_greyscale_mask(mov_img)

        bbox_mask = np.array(mask_to_bbox(mov_mask))
        bbox_ref = transform_bbox(bbox_mask, M, ref_offset, dilate=pad_overlap)
    
    overlap_ref, overlap_ref_mask = get_overlap_z(ref_img, bbox_ref, ref_mask, compute_mask=True)

    if return_sift:
        return overlap_ref, overlap_ref_mask, bbox_ref, (M, output_shape, ref_offset, valid_estimate, stats)
    else:
        return overlap_ref, overlap_ref_mask, bbox_ref