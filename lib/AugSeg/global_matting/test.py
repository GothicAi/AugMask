import opencv_mat as gm
import cv2

img_np = cv2.imread('GT04-image.png')
trimap_np = cv2.imread('GT04-trimap.png', cv2.IMREAD_GRAYSCALE)
alpha_np = gm.global_matting(img_np, trimap_np)
alpha_np = gm.guided_filter(img_np, trimap_np, alpha_np, 10, 1e-5)
cv2.imwrite('GT04-alpha.png', alpha_np)
