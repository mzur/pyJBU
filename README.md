# pyJBU

A Python implementation of Joint Bilateral Upsampling [[1]](#ref1).

## Installation

```
pip install -r requirements.txt
```

## Usage

```
usage: jbu.py [-h] [--radius RADIUS] [--sigma-spatial SIGMA_SPATIAL]
              [--sigma-range SIGMA_RANGE]
              source reference output

Perform Joint Bilateral Upsampling with a source and reference image

positional arguments:
  source                Path to the source image
  reference             Path to the reference image
  output                Path to the output image

optional arguments:
  -h, --help            show this help message and exit
  --radius RADIUS       Radius of the filter kernels (default: 2)
  --sigma-spatial SIGMA_SPATIAL
                        Sigma of the spatial weights (default: 2.5)
  --sigma-range SIGMA_RANGE
                        Sigma of the range weights (default: standard
                        deviation of the reference image)
```

The source image will be upsampled to the resolution of the reference image. The result will be written to the specified path of the output image.

## References

1. <a name="ref1"></a>Kopf, J., Cohen, M. F., Lischinski, D., & Uyttendaele, M. (2007, August). [Joint bilateral upsampling.](https://johanneskopf.de/publications/jbu/) In ACM Transactions on Graphics (ToG) (Vol. 26, No. 3, p. 96). ACM.
