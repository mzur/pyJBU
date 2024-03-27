import numpy as np
cimport numpy as np

def process_row(int y, int padding, tuple ref_size, np.ndarray[np.float32_t, ndim=3] reference_padded, np.ndarray[np.float32_t, ndim=3] source_upsampled_padded, np.ndarray[np.float32_t, ndim=2] kernel_spatial, np.ndarray[np.float32_t, ndim=2] lut_range, int step, bint rgb):
    if not rgb:
        raise ValueError("Only RGB mode is supported")
    
    cdef np.ndarray[np.float64_t, ndim=2] row
    cdef np.ndarray[np.float64_t, ndim=1] X
    cdef np.ndarray[np.float64_t, ndim=1] I_p
    cdef np.ndarray[np.float64_t, ndim=2] patch_reference
    cdef np.ndarray[np.float64_t, ndim=2] patch_source_upsampled
    cdef np.ndarray[np.float64_t, ndim=2] kernel_range
    cdef np.ndarray[np.float64_t, ndim=2] weight
    cdef np.ndarray[np.float64_t, ndim=1] k_p

    row = np.zeros((ref_size[0], 3))
    y += padding
    X = np.arange(padding, reference_padded.shape[1] - padding)
    I_p = reference_padded[y, X]
    patch_reference = np.array([np.ascontiguousarray(reference_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1, 3) for x in X])
    patch_source_upsampled = np.array([np.ascontiguousarray(source_upsampled_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1, 3) for x in X])

    kernel_range = lut_range[np.abs(patch_reference - np.expand_dims(I_p, axis=1)).astype(int)]
    weight = kernel_range * kernel_spatial
    k_p = weight.sum(axis=1)
    row[X - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=1) / k_p)
    return row
