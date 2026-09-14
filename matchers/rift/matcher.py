import numpy as np
import cv2

def phase_congruency(
    image,
    nscale=3,
    norient=3,
    min_wave_length=3,
    mult=1.6,
    sigma_on_f=0.75,
    k=1.0,
    cutoff=0.5,
    g=3.0,
):
    """Compute a Python implementation of the RIFT phase-congruency stage."""

    image = np.asarray(image, dtype=np.float64)

    if image.ndim != 2:
        raise ValueError("RIFT expects a grayscale 2-D image.")

    rows, cols = image.shape
    epsilon = 1e-4

    # ------------------------------------------------------------
    # Frequency coordinates
    # Equivalent to the reference phasecong3.m implementation.
    # ------------------------------------------------------------
    if cols % 2:
        xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
        xrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2:
        yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
        yrange = np.arange(-rows / 2, rows / 2) / rows

    x, y = np.meshgrid(xrange, yrange)

    radius = np.sqrt(x**2 + y**2)
    theta = np.arctan2(-y, x)

    # Move zero frequency to the top-left corner.
    radius = np.fft.ifftshift(radius)
    theta = np.fft.ifftshift(theta)

    radius[0, 0] = 1.0

    sintheta = np.sin(theta)
    costheta = np.cos(theta)

    # ------------------------------------------------------------
    # Low-pass filter
    # ------------------------------------------------------------


    # Exact coordinate construction from lowpassfilter.m.
    if cols % 2:
     xrange = np.arange(-(cols - 1) / 2, (cols - 1) / 2 + 1) / (cols - 1)
    else:
     xrange = np.arange(-cols / 2, cols / 2) / cols

    if rows % 2:
     yrange = np.arange(-(rows - 1) / 2, (rows - 1) / 2 + 1) / (rows - 1)
    else:
     yrange = np.arange(-rows / 2, rows / 2) / rows

    xx, yy = np.meshgrid(xrange, yrange)
    lp_radius = np.sqrt(xx**2 + yy**2)
    lp_radius = np.fft.ifftshift(lp_radius)

    lp = 1.0 / (1.0 + (lp_radius / 0.45) ** (2 * 15))
    

    # ------------------------------------------------------------
    # Construct log-Gabor radial filters
    # ------------------------------------------------------------
    log_gabor = []

    for scale in range(nscale):
        wavelength = min_wave_length * (mult ** scale)
        fo = 1.0 / wavelength

        filt = np.exp(
            -(
                np.log(radius / fo) ** 2
            )
            / (2 * np.log(sigma_on_f) ** 2)
        )

        filt *= lp
        filt[0, 0] = 0.0

        log_gabor.append(filt)

    image_fft = np.fft.fft2(image)

    # ------------------------------------------------------------
    # Accumulators
    # ------------------------------------------------------------
    eo = [[None for _ in range(norient)] for _ in range(nscale)]
    pc_maps = []

    pc_sum = np.zeros_like(image)

    covx2_total = np.zeros_like(image)
    covy2_total = np.zeros_like(image)
    covxy_total = np.zeros_like(image)

    energy_v1 = np.zeros_like(image)
    energy_v2 = np.zeros_like(image)
    energy_v3 = np.zeros_like(image)

    # ------------------------------------------------------------
    # Main orientation loop
    # ------------------------------------------------------------
    for orientation in range(norient):

        angl = orientation * np.pi / norient

        # Angular distance from filter orientation.
        ds = (
            sintheta * np.cos(angl)
            - costheta * np.sin(angl)
        )

        dc = (
            costheta * np.cos(angl)
            + sintheta * np.sin(angl)
        )

        dtheta = np.abs(np.arctan2(ds, dc))

        dtheta = np.minimum(
            dtheta * norient / 2.0,
            np.pi,
        )

        spread = (np.cos(dtheta) + 1.0) / 2.0

        sum_e = np.zeros_like(image)
        sum_o = np.zeros_like(image)
        sum_an = np.zeros_like(image)
        energy = np.zeros_like(image)

        max_an = None
        tau = 0.0

        # --------------------------------------------------------
        # Process all scales
        # --------------------------------------------------------
        for scale in range(nscale):

            filt = log_gabor[scale] * spread

            response = np.fft.ifft2(image_fft * filt)

            # IMPORTANT:
            # Keep the complex response.
            eo[scale][orientation] = response

            amplitude = np.abs(response)

            sum_an += amplitude
            sum_e += np.real(response)
            sum_o += np.imag(response)

            if scale == 0:
                # Median-based Rayleigh noise estimate.
                tau = np.median(sum_an) / np.sqrt(np.log(4.0))
                max_an = amplitude.copy()
            else:
                max_an = np.maximum(max_an, amplitude)

        # --------------------------------------------------------
        # Energy vector
        # --------------------------------------------------------
        energy_v1 += sum_e
        energy_v2 += np.cos(angl) * sum_o
        energy_v3 += np.sin(angl) * sum_o

        # Weighted mean filter response vector.
        x_energy = np.sqrt(sum_e**2 + sum_o**2) + epsilon

        mean_e = sum_e / x_energy
        mean_o = sum_o / x_energy

        # --------------------------------------------------------
        # Phase-deviation energy
        # --------------------------------------------------------
        for scale in range(nscale):

            response = eo[scale][orientation]

            e = np.real(response)
            o = np.imag(response)

            energy += (
                e * mean_e
                + o * mean_o
                - np.abs(e * mean_o - o * mean_e)
            )

        # --------------------------------------------------------
        # Noise threshold
        # --------------------------------------------------------
        total_tau = (
            tau
            * (1.0 - (1.0 / mult) ** nscale)
            / (1.0 - (1.0 / mult))
        )

        noise_mean = total_tau * np.sqrt(np.pi / 2.0)
        noise_sigma = total_tau * np.sqrt((4.0 - np.pi) / 2.0)

        threshold = noise_mean + k * noise_sigma

        # Soft thresholding.
        energy = np.maximum(energy - threshold, 0.0)

        # --------------------------------------------------------
        # Frequency spread weighting
        # --------------------------------------------------------
        width = (
            (sum_an / (max_an + epsilon)) - 1.0
        ) / max(nscale - 1, 1)

        weight = 1.0 / (
            1.0 + np.exp((cutoff - width) * g)
        )

        # Phase congruency for this orientation.
        pc = weight * energy / (sum_an + epsilon)

        pc_maps.append(pc)
        pc_sum += pc

        # --------------------------------------------------------
        # Covariance accumulation
        # --------------------------------------------------------
        covx = pc * np.cos(angl)
        covy = pc * np.sin(angl)

        covx2_total += covx**2
        covy2_total += covy**2
        covxy_total += covx * covy

    # ------------------------------------------------------------
    # Calculate maximum phase-congruency moment M
    # ------------------------------------------------------------
    covx2 = covx2_total / (norient / 2.0)
    covy2 = covy2_total / (norient / 2.0)
    covxy = 4.0 * covxy_total / norient

    denom = np.sqrt(
        covxy**2
        + (covx2 - covy2) ** 2
    ) + epsilon

    M = (
        covy2
        + covx2
        + denom
    ) / 2.0

    return pc_maps, eo, M

def maximum_index_map(eo):
    """
    Build the Maximum Index Map (MIM) used by RIFT.

    For each orientation, sum the absolute filter responses
    across all scales. Then select the orientation with the
    maximum response at each pixel.

    This follows:
        CS(:,:,j) = CS(:,:,j) + abs(eo{i,j})
        [~, MIM] = max(CS, [], 3)
    from the reference RIFT implementation.
    """
    nscale = len(eo)
    norient = len(eo[0])

    # CS[row, col, orientation]
    cs = np.zeros(
        (*eo[0][0].shape, norient),
        dtype=np.float64,
    )

    for orientation in range(norient):
        for scale in range(nscale):
            cs[:, :, orientation] += np.abs(
                eo[scale][orientation]
            )

    # MATLAB indices are 1..o.
    # Our descriptor histogram uses bins 0..norient-1,
    # so convert the argmax result directly to zero-based indices.
    mim = np.argmax(cs, axis=2)

    return mim


def detect_rift_keypoints(image, max_keypoints=5000):
    """
    Detect FAST keypoints on a normalized phase-congruency map.
    """
    image = np.asarray(image, dtype=np.float32)

    # Normalize to [0, 1], matching the MATLAB RIFT preprocessing.
    minimum = float(image.min())
    maximum = float(image.max())

    if maximum > minimum:
        normalized = (image - minimum) / (maximum - minimum)
    else:
        normalized = np.zeros_like(image)

    # OpenCV FAST.
    fast = cv2.FastFeatureDetector_create(
        threshold=5,
        nonmaxSuppression=True,
    )

    fast_image = np.clip(normalized * 255.0, 0, 255).astype(np.uint8)
    keypoints = fast.detect(fast_image, None)

    # Strongest responses first.
    keypoints = sorted(
        keypoints,
        key=lambda kp: kp.response,
        reverse=True,
    )

    return keypoints[:max_keypoints]

def compute_rift_descriptors(keypoints, mim, patch_size=96, norient=3):
    """
    Compute the RIFT descriptor from keypoints and a Maximum
    Index Map (MIM).

    The reference implementation uses:
        - 6 x 6 spatial cells
        - `norient` histogram bins per cell
        - L2 normalization

    Returns:
        valid_keypoints: keypoints for which a full patch exists
        descriptors: (N, 36 * norient) float32 array
    """
    mim = np.asarray(mim)

    height, width = mim.shape
    ns = 6

    valid_keypoints = []
    descriptors = []

    half = patch_size // 2

    for keypoint in keypoints:
        x, y = keypoint.pt

        # Match the MATLAB boundary logic.
        x = int(round(x))
        y = int(round(y))

        x1 = max(0, x - half)
        y1 = max(0, y - half)
        x2 = min(x + half, width)
        y2 = min(y + half, height)

        # Require a complete patch.
        if (y2 - y1) != patch_size or (x2 - x1) != patch_size:
            continue

        patch = mim[y1:y2, x1:x2]

        descriptor = []

        for j in range(ns):
            row_start = round(j * patch_size / ns)
            row_end = round((j + 1) * patch_size / ns)

            for i in range(ns):
                col_start = round(i * patch_size / ns)
                col_end = round((i + 1) * patch_size / ns)

                cell = patch[
                    row_start:row_end,
                    col_start:col_end,
                ]

                # MATLAB hist(clip(:), 1:o):
                # create one histogram bin for each MIM index.
                hist = np.array(
                    [
                        np.sum(cell == bin_index)
                        for bin_index in range(norient)
                    ],
                    dtype=np.float32,
                )

                descriptor.extend(hist)

        descriptor = np.asarray(descriptor, dtype=np.float32)

        # L2 normalization, exactly as in the reference.
        norm = np.linalg.norm(descriptor)

        if norm != 0:
            descriptor /= norm

        valid_keypoints.append(keypoint)
        descriptors.append(descriptor)

    if not descriptors:
        return (
            [],
            np.empty((0, ns * ns * norient), dtype=np.float32),
        )

    return (
        valid_keypoints,
        np.vstack(descriptors),
    )    

def match_rift_descriptors(descriptors1, descriptors2):
    """
    Nearest-neighbor matching following the reference RIFT implementation.

    The MATLAB implementation uses MaxRatio=1 and then removes
    duplicate target keypoints.
    """
    if len(descriptors1) == 0 or len(descriptors2) == 0:
        return np.empty((0, 2), dtype=np.int32)

    matcher = cv2.BFMatcher(cv2.NORM_L2)

    # One nearest neighbour, equivalent to MaxRatio=1.
    matches = matcher.match(descriptors1, descriptors2)

    # Keep the best match for each target descriptor.
    matches = sorted(matches, key=lambda m: m.distance)

    used_targets = set()
    good_matches = []

    for match in matches:
        if match.trainIdx in used_targets:
            continue

        used_targets.add(match.trainIdx)
        good_matches.append(
            [match.queryIdx, match.trainIdx]
        )

    return np.asarray(good_matches, dtype=np.int32)
class RIFTMatcher:
    """Pipeline wrapper for the RIFT multimodal image matcher."""

    def __init__(self, max_keypoints=1000, patch_size=96):
        self.max_keypoints = max_keypoints
        self.patch_size = patch_size

    def match(self, image1, image2):
        _, eo1, M1 = phase_congruency(image1)
        _, eo2, M2 = phase_congruency(image2)

        keypoints1 = detect_rift_keypoints(
            M1,
            max_keypoints=self.max_keypoints,
        )
        keypoints2 = detect_rift_keypoints(
            M2,
            max_keypoints=self.max_keypoints,
        )

        keypoints1, descriptors1 = compute_rift_descriptors(
            keypoints1,
            maximum_index_map(eo1),
            patch_size=self.patch_size,
            norient=3,
        )

        keypoints2, descriptors2 = compute_rift_descriptors(
            keypoints2,
            maximum_index_map(eo2),
            patch_size=self.patch_size,
            norient=3,
        )

        matches = match_rift_descriptors(
            descriptors1,
            descriptors2,
        )

        if len(matches) == 0:
            return {
                "points1": np.empty((0, 2), dtype=np.float32),
                "points2": np.empty((0, 2), dtype=np.float32),
                "confidence": 0.0,
            }

        points1 = np.array(
            [keypoints1[i].pt for i, _ in matches],
            dtype=np.float32,
        )

        points2 = np.array(
            [keypoints2[j].pt for _, j in matches],
            dtype=np.float32,
        )

        return {
            "points1": points1,
            "points2": points2,
            "confidence": float(len(matches)),
        }    