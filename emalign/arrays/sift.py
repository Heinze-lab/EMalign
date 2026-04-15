import cv2
import numpy as np

from emalign.arrays.utils import resample


def adjust_matrix_to_shape(mov_img, M):

    y, x = mov_img.shape[:2]
    corners = np.array([[0, 0], [x, 0], [x, y], [0, y]], dtype=np.float32).reshape(-1, 1, 2)
    transformed_corners = cv2.transform(corners, M)
    
    min_x = np.floor(np.min(transformed_corners[:, 0, 0])).astype(int)
    min_y = np.floor(np.min(transformed_corners[:, 0, 1])).astype(int)
    max_x = np.ceil(np.max(transformed_corners[:, 0, 0])).astype(int)
    max_y = np.ceil(np.max(transformed_corners[:, 0, 1])).astype(int)
    
    # Create adjusted transformation matrix that accounts for shifts
    translation_adj = np.array([
        [1, 0, -min(min_x, 0)],
        [0, 1, -min(min_y, 0)]
    ], dtype=np.float32)
    
    adjusted_M = M.copy()
    adjusted_M[:, 2] += translation_adj[:, 2]

    # Get output shape
    output_w = np.ceil(max_x-min(min_x, 0))
    output_h = np.ceil(max_y-min(min_y, 0))
    mov_shape = np.array([output_h, output_w])

    # Get transformation for the other image to match
    ref_offset = M[:, 2] - adjusted_M[:, 2]

    return adjusted_M, mov_shape, ref_offset


def calculate_sift_robustness_index(good_matches, inliers, M, src_pts, dst_pts, 
                                   pixel_tolerance=10):
    """
    Calculate a robustness index (0-1) for SIFT registration results.
    
    Args:
        good_matches: List of good matches from SIFT
        inliers: Boolean array from cv2.estimateAffinePartial2D 
        M: 2x3 transformation matrix
        src_pts: Source keypoints (N, 1, 2)
        dst_pts: Destination keypoints (N, 1, 2)
        pixel_tolerance: Acceptable residual error in pixels (default: 3.0)
    
    Returns:
        float: Robustness index between 0 (poor) and 1 (excellent)
        dict: Detailed metrics used in calculation
    """
    
    # Handle edge cases
    if len(good_matches) == 0 or M is None:
        return 0.0, {'reason': 'No matches or invalid transformation'}
    
    n_matches = len(good_matches)
    n_inliers = inliers.sum() if inliers is not None else 0
    
    if n_inliers == 0:
        return 0.0, {'reason': 'No inliers found'}
    
    # Calculate residuals for inlier matches
    inlier_mask = inliers.flatten() if inliers is not None else np.ones(n_matches, dtype=bool)
    
    # Transform source points using estimated matrix
    src_2d = src_pts.reshape(-1, 2)
    dst_2d = dst_pts.reshape(-1, 2)
    src_homogeneous = np.column_stack([src_2d, np.ones(len(src_2d))])
    transformed_src = (M @ src_homogeneous.T).T[:, :2]
    
    # Calculate residuals for all matches
    all_residuals = np.linalg.norm(transformed_src - dst_2d, axis=1)
    
    # Trim 25% worse residuals so that they don't bias mean and std
    inlier_residuals = np.sort(all_residuals[inlier_mask])
    cutoff = int(len(inlier_residuals) * 0.75)
    inlier_residuals = inlier_residuals[:cutoff]
    
    # Component 1: Match quantity score (0-1)
    # Uses log scale to give diminishing returns for very high match counts
    if n_matches <= 5:
        quantity_score = n_matches / 10.0  # Very low score for few matches
    elif n_matches <= 20:
        quantity_score = 0.5 + (n_matches - 5) / 30.0  # Linear increase
    else:
        # Logarithmic scaling for high match counts
        quantity_score = 0.8 + 0.2 * min(1.0, np.log(n_matches - 19) / np.log(50))
    
    # Component 2: Inlier proportion score (0-1)
    inlier_ratio = n_inliers / n_matches
    if inlier_ratio >= 0.7:
        proportion_score = 1.0
    elif inlier_ratio >= 0.4:
        proportion_score = 0.6 + 0.4 * (inlier_ratio - 0.4) / 0.3
    elif inlier_ratio >= 0.2:
        proportion_score = 0.3 + 0.3 * (inlier_ratio - 0.2) / 0.2
    else:
        proportion_score = 0.3 * (inlier_ratio / 0.2)
    
    # Component 3: Match quality score based on residuals (0-1)
    mean_residual = np.mean(inlier_residuals)
    std_residual = np.std(inlier_residuals)
    
    # Quality based on mean residual relative to tolerance
    if mean_residual <= pixel_tolerance * 0.5:
        quality_score = 1.0
    elif mean_residual <= pixel_tolerance:
        quality_score = 0.8 - 0.3 * (mean_residual - pixel_tolerance * 0.5) / (pixel_tolerance * 0.5)
    elif mean_residual <= pixel_tolerance * 2:
        quality_score = 0.5 - 0.3 * (mean_residual - pixel_tolerance) / pixel_tolerance
    else:
        quality_score = max(0.1, 0.2 - 0.1 * (mean_residual - 2 * pixel_tolerance) / pixel_tolerance)
    
    # Component 4: Consistency score based on residual distribution (0-1)
    # Good matches should have consistent, low residuals
    acceptable_residuals = np.sum(inlier_residuals <= pixel_tolerance)
    consistency_ratio = acceptable_residuals / len(inlier_residuals)
    
    if consistency_ratio >= 0.9:
        consistency_score = 1.0
    elif consistency_ratio >= 0.7:
        consistency_score = 0.8 + 0.2 * (consistency_ratio - 0.7) / 0.2
    elif consistency_ratio >= 0.5:
        consistency_score = 0.5 + 0.3 * (consistency_ratio - 0.5) / 0.2
    else:
        consistency_score = 0.5 * (consistency_ratio / 0.5)
    
    # Component 5: Absolute minimum requirements
    # Penalize severely if basic requirements aren't met
    # Ignore inliers ratio because it's not super informative for this
    min_requirements_met = (n_matches >= 10 and n_inliers >= 6 and mean_residual <= pixel_tolerance * 3)
    
    if not min_requirements_met:
        penalty_factor = 0.3  # Severe penalty
    else:
        penalty_factor = 1.0
    
    # Weighted combination of components
    weights = {
        'quantity': 0.25,      # Match count contribution
        'proportion': 0.15,    # Inlier ratio contribution. This is not that useful as it tends to be very low even for good matches
        'quality': 0.35,       # Residual quality (most important)
        'consistency': 0.25    # Residual consistency
    }
    
    robustness_index = (
        weights['quantity'] * quantity_score +
        weights['proportion'] * proportion_score +
        weights['quality'] * quality_score +
        weights['consistency'] * consistency_score
    ) * penalty_factor
    
    # Ensure result is in [0, 1]
    robustness_index = np.clip(robustness_index, 0.0, 1.0)
    
    # Detailed metrics for debugging/analysis
    metrics = {
        'robustness_index': robustness_index,
        'n_matches': n_matches,
        'n_inliers': n_inliers,
        'inlier_ratio': inlier_ratio,
        'mean_residual': mean_residual,
        'std_residual': std_residual,
        'consistency_ratio': consistency_ratio,
        'component_scores': {
            'quantity': quantity_score,
            'proportion': proportion_score, 
            'quality': quality_score,
            'consistency': consistency_score
        },
        'min_requirements_met': min_requirements_met,
        'penalty_factor': penalty_factor,
        'pixel_tolerance': pixel_tolerance
    }
    
    return robustness_index, metrics


def estimate_transform_sift(ref_img, 
                            mov_img, 
                            scale=1.0, 
                            ref_mask=None, 
                            mov_mask=None,
                            refine_estimate=True,
                            return_upscaled_matrix=True,
                            return_raw_homology=False):
    '''Estimate transformation (xy offset and rotation) from img2 to img1 using SIFT.
    Note that using masks may marginally increase compute time.

    Args:
        ref_img (np.ndarray): Reference greyscale image.
        mov_img (np.ndarray): Moving greyscale image.
        scale (float, optional): Scale to resample images to for computing the offset. Defaults to 1.
        refine_estimate (bool, optional): Whether to try again with higher resolution if the first estimate is found to be invalid. Defaults to True.
        return_upscaled_matrix (bool, optional): Whether to return the matrix corresponding to the transformation to apply to the original image (as opposed to the resampled one). Defaults to True.
        ref_mask (np.ndarray): Boolean mask for the regions to find keypoints in for the reference greyscale image. Defaults to None.
        mov_mask (np.ndarray): Boolean mask for the regions to find keypoints in for the the moving greyscale image. Defaults to None.

    Returns:
        tuple of: 
            M (np.ndarray): Affine transformation matrix to apply to mov_img.
            output_shape (tuple): (y,x) shape for mov_img after transformation.
            ref_offset (nd.array): (x,y) offset to apply to ref_img for it to match with mov_img.
            robust_estimate (bool): Whether the estimate was valid based on the number and proportion of good matches.
            robustness_metrics (dict): Dictionary of various metrics used to determine the robustness of the estimate.
    '''
    # knnMatch will return an error if there are too many keypoints so we limit their number
    max_features=250000

    # resample images for faster computations
    ds_ref_img = resample(ref_img, scale)
    ds_mov_img = resample(mov_img, scale)

    ds_ref_mask = resample(ref_mask.astype(np.uint8), scale) if ref_mask is not None else None # cv2 does not want bool
    ds_mov_mask = resample(mov_mask.astype(np.uint8), scale) if mov_mask is not None else None

    # Find keypoints using SIFT
    sift = cv2.SIFT_create(nfeatures=max_features)
    kp1, des1 = sift.detectAndCompute(ds_ref_img, mask=ds_ref_mask)
    kp2, des2 = sift.detectAndCompute(ds_mov_img, mask=ds_mov_mask)

    if len(kp1) and len(kp2):
        # Match keypoints to each other
        # Brute force matchers is slower than flann, but it is exact
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1,des2,k=2)

        good_matches = []
        for m,n in matches:
            if m.distance < 0.7*n.distance:
                good_matches.append(m)
        
        good_matches = sorted(good_matches, key=lambda x: x.distance)
        dst_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
        src_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

        # Estimate affine transformation matrix
        try:
            M, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
        except cv2.error as e:
            if 'count >= 0' in e.err:
                M = None
            else:
                raise e
        except Exception as e:
            raise e
    else:
        M = None
               
    if M is None or np.isnan(M).all():
        output_shape = None
        ref_offset = None
        robust_estimate = False
        robustness_metrics = None
    else:
        robustness_index, robustness_metrics = calculate_sift_robustness_index(good_matches, inliers, M, src_pts, dst_pts, pixel_tolerance=20)
        robust_estimate = robustness_index >= 0.45

        if return_upscaled_matrix:
            M[:,2] /= scale
        
        if return_raw_homology:
            output_shape = None
            ref_offset = None
        else:
            M, output_shape, ref_offset = adjust_matrix_to_shape(mov_img, M)
        
    if refine_estimate and not robust_estimate and scale<0.9:
        return estimate_transform_sift(ref_img, mov_img, scale=scale+0.1, refine_estimate=False)
    else:
        if ref_offset is not None:
            ref_offset = ref_offset.astype(int)
        return M, output_shape, ref_offset, robust_estimate, robustness_metrics