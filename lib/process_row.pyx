import numpy as np
cimport numpy as np

def process_row_rgb(int y, int padding, tuple ref_size, 
                    np.ndarray[np.float32_t, ndim=3] reference_padded, 
                    np.ndarray[np.float32_t, ndim=3] source_upsampled_padded, 
                    np.ndarray[np.float32_t, ndim=2] kernel_spatial,
                    np.ndarray[np.float32_t, ndim=1] lut_range, 
                    int step):

    cdef np.ndarray[np.float32_t, ndim=2] row = np.zeros((ref_size[0], 3), dtype=np.float32)

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

def process_row_gray(int y, int padding, tuple ref_size, 
                     np.ndarray[np.float32_t, ndim=2] reference_padded, 
                     np.ndarray[np.float32_t, ndim=2] source_upsampled_padded, 
                     np.ndarray[np.float32_t, ndim=1] kernel_spatial,
                     np.ndarray[np.float32_t, ndim=1] 
                     lut_range, 
                     int step):

    cdef np.ndarray[np.float32_t, ndim=1] row = np.zeros(ref_size[0], dtype=np.float32)

    y += padding
    X = np.arange(padding, reference_padded.shape[1] - padding)
    I_p = reference_padded[y, X]
    patch_reference = np.array([np.ascontiguousarray(reference_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1) for x in X])
    patch_source_upsampled = np.array([np.ascontiguousarray(source_upsampled_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1) for x in X])

    kernel_range = lut_range[np.abs(patch_reference - np.expand_dims(I_p, axis=1)).astype(int)]
    weight = kernel_range * kernel_spatial
    k_p = weight.sum(axis=1)
    row[X - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=1) / k_p)

    return row
