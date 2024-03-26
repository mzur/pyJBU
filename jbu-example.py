import cv2
from lib.pyjbu import JBU

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
