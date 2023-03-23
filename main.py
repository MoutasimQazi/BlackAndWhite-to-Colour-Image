import cv2

img = cv2.imread('bw_image.png', cv2.IMREAD_GRAYSCALE)

img_color = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

colorizer = cv2.dnn.readNetFromCaffe('colorization_deploy_v2.prototxt', 'colorization_release_v2.caffemodel')
class8 = colorizer.getLayerId("class8_ab")
conv8 = colorizer.getLayerId("conv8_313_rh")
pts = np.zeros([1, 1, 2, 313], dtype=np.float32)
pts[0, 0, :, :] = np.transpose(np.array([np.arange(313)] * 2))
colorizer.getLayer(class8).blobs = [pts]
colorizer.getLayer(conv8).blobs = [np.full([1, 313], 2.606, dtype=np.float32)]
output = colorizer.forward((img_color.astype(np.float32) / 255. - 0.5) * 2)
output = output.squeeze().transpose((1, 2, 0))
output_resized = cv2.resize(output, (img.shape[1], img.shape[0]))

cv2.imshow("Original Image", img)
cv2.imshow("Colorized Image", output_resized)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Line 03  Load the black and white image
# Line 05 Convert the image to color using cvtColor() function
# Line 07 Apply colorization algorithm to the image
# Line 18 Display the original and colorized images side by side
