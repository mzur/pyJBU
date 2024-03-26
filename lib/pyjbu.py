'''
Joint Bilateral Upsampling
paper: https://johanneskopf.de/publications/jbu/
forked from https://github.com/mzur/pyJBU
'''

import cv2
import numpy as np
import multiprocessing
# from numba import jit  # TODO: process_row not yet working with numba


class JBU:
    def __init__(self, radius=2, sigma_spatial=2.5, sigma_range=None, width=None):
        # Parameters
        self.radius = int(radius)
        self.diameter = 2 * self.radius + 1
        self.sigma_spatial = float(sigma_spatial)
        self.sigma_range = sigma_range

        # original images
        self.reference = None
        self.source = None
        # working copies
        self.reference_padded = None
        self.source_upsampled_padded = None

        # internal variables
        self.width = width if width else None
        self.ref_size = None  # (width, height)
        self.src_size = None
        self.scale = None
        self.step = None
        self.padding = None

        # spatial Gaussian function.
        self.x, self.y = np.meshgrid(np.arange(self.diameter) - self.radius, np.arange(self.diameter) - self.radius)
        kernel_spatial = np.exp(-1.0 * (self.x**2 + self.y**2) /  (2 * self.sigma_spatial**2))
        self.kernel_spatial = np.repeat(kernel_spatial, 3).reshape(-1, 3)

        # Lookup table for range kernel.
        self.lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * self.sigma_range**2))
        

    def process_image(self):
        result = np.zeros((self.ref_size[1], self.ref_size[0], 3))
        with multiprocessing.Pool() as pool:
            result_rows = pool.starmap(process_row, [(y, self.padding, self.ref_size, self.reference_padded, self.source_upsampled_padded, self.kernel_spatial, self.lut_range, self.step) for y in range(self.ref_size[1])])

        for i, row in enumerate(result_rows):
            result[i] = row

        img_array = np.array(list(result)).astype(np.uint8)
        return img_array
    

    def run(self, source, reference):
        self.reference = reference
        self.source = source

        self.aspect_ratio = self.reference.shape[1] / self.reference.shape[0]
        self.src_size = source.shape[1::-1]
        
        if self.width is not None:
            # Calculate new reference size based on width and aspect ratio
            height = int(self.width / self.aspect_ratio)
            self.ref_size = (self.width, height)
            # Resize reference image
            self.reference = cv2.resize(self.reference, self.ref_size, interpolation=cv2.INTER_LINEAR)
        else:
            # Use reference size
            self.ref_size = self.reference.shape[1::-1]  # (width, height)
        
        # Resize source image
        self.source = cv2.resize(self.source, self.ref_size, interpolation=cv2.INTER_LINEAR)

        self.scale = self.src_size[0] / self.ref_size[0]
        self.step = int(np.ceil(1 / self.scale))
        self.padding = self.radius * self.step
        self.sigma_range = float(self.sigma_range) if self.sigma_range else np.std(self.reference)
        self.source_upsampled_padded = np.pad(self.source, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'symmetric').astype(np.float32)
        self.reference_padded = np.pad(self.reference, ((self.padding, self.padding), (self.padding, self.padding), (0, 0)), 'symmetric').astype(np.float32)

        return self.process_image()


#@jit(nopython=True)
def process_row(y, padding, ref_size, reference_padded, source_upsampled_padded, kernel_spatial, lut_range, step):
    row = np.zeros((ref_size[0], 3))
    y += padding
    X = np.arange(padding, reference_padded.shape[1] - padding)
    I_p = reference_padded[y, X]

    patch_reference = np.array([reference_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3) for x in X])
    patch_source_upsampled = np.array([source_upsampled_padded[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3) for x in X])

    kernel_range = lut_range[np.abs(patch_reference - np.expand_dims(I_p, axis=1)).astype(int)]
    weight = kernel_range * kernel_spatial
    k_p = weight.sum(axis=1)
    row[X - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=1) / k_p)
    return row


if __name__ == '__main__':

    source_path     = "images/depth.jpg"
    reference_path  = "images/color.jpg"
    output_path     = "images/output.jpg"

    reference = cv2.imread(reference_path)
    source = cv2.imread(source_path)

    jbu = JBU(radius=1, sigma_spatial=3.0, sigma_range=6.5, width=500)
    img = jbu.run(source, reference)

    cv2.imshow("output", img)
    cv2.imwrite(output_path, img)
    cv2.waitKey(0)
