import argparse
from PIL import Image
import numpy as np
from concurrent.futures import ProcessPoolExecutor

parser = argparse.ArgumentParser(description="Perform Joint Bilateral Upsampling with a source and reference image")
parser.add_argument("source", help="Path to the source image")
parser.add_argument("reference", help="Path to the reference image")
parser.add_argument("output", help="Path to the output image")
parser.add_argument('--radius', dest='radius', default=2, help='Radius of the filter kernels (default: 2)')
parser.add_argument('--sigma-spatial', dest='sigma_spatial', default=2.5, help='Sigma of the spatial weights (default: 2.5)')
parser.add_argument('--sigma-range', dest='sigma_range', help='Sigma of the range weights (default: standard deviation of the reference image)')
args = parser.parse_args()

source_image = Image.open(args.source)

reference_image = Image.open(args.reference)
reference = np.array(reference_image)

source_image_upsampled = source_image.resize(reference_image.size, Image.BILINEAR)
source_upsampled = np.array(source_image_upsampled)

scale = source_image.width / reference_image.width
radius = int(args.radius)
diameter = 2 * radius + 1
step = int(np.ceil(1 / scale))
padding = radius * step
sigma_spatial = float(args.sigma_spatial)
sigma_range = float(args.sigma_range) if args.sigma_range else np.std(reference)

reference = np.pad(reference, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)
source_upsampled = np.pad(source_upsampled, ((padding, padding), (padding, padding), (0, 0)), 'symmetric').astype(np.float32)

# Spatial Gaussian function.
x, y = np.meshgrid(np.arange(diameter) - radius, np.arange(diameter) - radius)
kernel_spatial = np.exp(-1.0 * (x**2 + y**2) /  (2 * sigma_spatial**2))
kernel_spatial = np.repeat(kernel_spatial, 3).reshape(-1, 3)

# Lookup table for range kernel.
lut_range = np.exp(-1.0 * np.arange(256)**2 / (2 * sigma_range**2))

def process_row(y):
   result = np.zeros((reference_image.width, 3))
   y += padding
   for x in range(padding, reference.shape[1] - padding):
      I_p = reference[y, x]
      patch_reference = reference[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)
      patch_source_upsampled = source_upsampled[y - padding:y + padding + 1:step, x - padding:x + padding + 1:step].reshape(-1, 3)

      kernel_range = lut_range[np.abs(patch_reference - I_p).astype(int)]
      weight = kernel_range * kernel_spatial
      k_p = weight.sum(axis=0)
      result[x - padding] = np.round(np.sum(weight * patch_source_upsampled, axis=0) / k_p)

   return result

executor = ProcessPoolExecutor()
result = executor.map(process_row, range(reference_image.height))
executor.shutdown(True)
Image.fromarray(np.array(list(result)).astype(np.uint8)).save(args.output)
