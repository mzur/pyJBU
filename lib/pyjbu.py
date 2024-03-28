'''
Joint Bilateral Upsampling
paper: https://johanneskopf.de/publications/jbu/
forked from https://github.com/mzur/pyJBU

compile Cython-based process_row: python build_cython.py build_ext --inplace
'''

import time
import cv2
import numpy as np


# CYTHON ALTERNATIVE
from process_row import process_row_rgb, process_row_gray  # type: ignore

# def process_row_rgb(y, padding, ref_size, reference_padded, source_upsampled_padded, kernel_spatial, lut_range, step):
#     row = np.zeros((ref_size[0], 3))
#     y += padding
#     X = np.arange(padding, reference_padded.shape[1] - padding)
#     I_p = reference_padded[y, X]
#     patch_reference = np.array([np.ascontiguousarray(reference_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1, 3) for x in X])
#     patch_source_upsampled = np.array([np.ascontiguousarray(source_upsampled_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1, 3) for x in X])

#     kernel_range = lut_range[np.abs(patch_reference - np.expand_dims(I_p, axis=1)).astype(int)]
#     weight = kernel_range * kernel_spatial
#     k_p = weight.sum(axis=1)
#     row[X - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=1) / k_p)
#     return row

# def process_row_gray(y, padding, ref_size, reference_padded, source_upsampled_padded, kernel_spatial, lut_range, step):
#     row = np.zeros(ref_size[0])
#     y += padding
#     X = np.arange(padding, reference_padded.shape[1] - padding)
#     I_p = reference_padded[y, X]
#     patch_reference = np.array([np.ascontiguousarray(reference_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1) for x in X])
#     patch_source_upsampled = np.array([np.ascontiguousarray(source_upsampled_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step]).reshape(-1) for x in X])
    
#     kernel_range = lut_range[np.abs(patch_reference - np.expand_dims(I_p, axis=1)).astype(int)]
#     weight = kernel_range * kernel_spatial
#     k_p = weight.sum(axis=1)
#     row[X - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=1) / k_p)
#     return row


mplib = "multiprocessing"  # multiprocessing or joblib

if mplib == "multiprocessing":  # 1.623086 seconds
    import multiprocessing
elif mplib == "joblib":         # 2.350385 seconds
    from joblib import Parallel, delayed


class JBU:
    def __init__(self, radius=2, sigma_spatial=2.5, sigma_range=None, width=None, rgb=True, mplib="multiprocessing"):
        # Parameters
        self.radius = int(radius)
        self.diameter = 2 * self.radius + 1
        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = sigma_range
        self.width = width if width else None
        self.rgb = rgb  # True if self.source.ndim == 3 else False
        self.mplib = mplib

        # spatial Gaussian function.
        self.x, self.y = np.meshgrid(np.arange(self.diameter) - self.radius, np.arange(self.diameter) - self.radius)

        self.kernel_spatial = np.exp(-1.0 * (self.x**2 + self.y**2) /  (2 * self.sigma_spatial**2))
        if self.rgb:
            self.kernel_spatial = np.repeat(self.kernel_spatial, 3).reshape(-1, 3)
        else:
            self.kernel_spatial = self.kernel_spatial.reshape(-1)


    def run(self, source, reference):
        # original images
        self.reference = reference
        self.source = source

        self.aspect_ratio = self.reference.shape[1] / self.reference.shape[0]
        self.src_size = source.shape[1::-1]
        
        if self.width is not None:
            # Calculate new reference size based on width and aspect ratio
            height = int(self.width / self.aspect_ratio)
            self.ref_size = (self.width, height)
            # Resize reference image
            self.reference = cv2.resize(self.reference, self.ref_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)
        else:
            # Use reference size
            self.ref_size = self.reference.shape[1::-1]  # (width, height)

        # Resize source image
        self.source = cv2.resize(self.source, self.ref_size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

        self.scale = self.src_size[0] / self.ref_size[0]  # scale: 0.4325
        self.step = int(np.ceil(1 / self.scale))          # step: 3
        self.padding = self.radius * self.step            # padding: 3               

        self.sigma_range = float(self.sigma_range) if self.sigma_range else np.std(self.reference)
        
        # Lookup table for range kernel
        self.lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * self.sigma_range**2))  # lut_range shape: (256,)
        
        if self.rgb:
            padding_tuple = ((self.padding, self.padding), (self.padding, self.padding), (0, 0))
        else:
            padding_tuple = ((self.padding, self.padding), (self.padding, self.padding))

        self.source_upsampled_padded = np.pad(self.source, padding_tuple, 'symmetric').astype(np.float32)
        self.reference_padded = np.pad(self.reference, padding_tuple, 'symmetric').astype(np.float32)

        return self.process_image()
    

    def process_image(self):
        if self.rgb:
            result = np.zeros((self.ref_size[1], self.ref_size[0], 3), dtype=np.float32)
        else:
            result = np.zeros((self.ref_size[1], self.ref_size[0]), dtype=np.float32)
        
        # Cython requires fixed data types
        self.reference_padded = self.reference_padded.astype(np.float32)
        self.source_upsampled_padded = self.source_upsampled_padded.astype(np.float32)
        self.kernel_spatial = self.kernel_spatial.astype(np.float32)
        self.lut_range = self.lut_range.astype(np.float32)

        params = [(y, self.padding, self.ref_size, self.reference_padded, self.source_upsampled_padded,
           self.kernel_spatial, self.lut_range, self.step) for y in range(self.ref_size[1])]
        
        process_row_func = process_row_rgb if self.rgb else process_row_gray

        if self.mplib == "multiprocessing":
            with multiprocessing.Pool() as pool:
                result_rows = pool.starmap(process_row_func, params)

        elif self.mplib == "joblib":
            result_rows = Parallel(n_jobs=-1)(delayed(process_row_func)(*param) for param in params)
        
        for i, row in enumerate(result_rows):
            result[i] = row

        return result.astype(np.uint8)


if __name__ == '__main__':

    source_path     = "images/depth.jpg"
    reference_path  = "images/color.jpg"
    output_path     = "images/output.jpg"

    use_rgb = False

    source = cv2.imread(source_path, int(use_rgb))
    reference = cv2.imread(reference_path, int(use_rgb))

    jbu = JBU(radius=2, sigma_spatial=2.5, sigma_range=6.5, width=400, rgb=use_rgb, mplib=mplib)

    start_time = time.time()  # Start the timer
    img = jbu.run(source, reference)
    end_time = time.time()  # End the timer
    print(f"{mplib}: {round(end_time - start_time, 6)} seconds")

    cv2.imshow("output", img)
    cv2.imwrite(output_path, img)
    cv2.waitKey(0)
