import cv2
from lib.pyjbu import JBU

if __name__ == '__main__':

    source_path     = "images/depth.jpg"
    reference_path  = "images/color.jpg"
    output_path     = "images/output.jpg"

    use_rgb = False

    source = cv2.imread(source_path, int(use_rgb))
    reference = cv2.imread(reference_path, int(use_rgb))

    jbu = JBU(radius=2, sigma_spatial=2.5, sigma_range=6.5, width=800, rgb=use_rgb)

    img = jbu.run(source, reference)

    cv2.imshow("output", img)
    cv2.imwrite(output_path, img)
    cv2.waitKey(0)
