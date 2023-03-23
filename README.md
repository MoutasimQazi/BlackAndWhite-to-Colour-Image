# BlackAndWhite-to-Colour-Image
In This Repositary I am Writing My Self Satisfaction Project #2
This is a Python code that uses the OpenCV library to apply a colorization algorithm to a black and white image.

main.py code first loads the black and white image 'bw_image.png' using the imread() function and converts it to color using the cvtColor() function.

It then applies a colorization algorithm to the image using a pre-trained deep neural network model. The model is loaded using the readNetFromCaffe() function, which takes two arguments: the path to the prototxt file and the path to the caffemodel file.

The code then sets up the class8_ab and conv8_313_rh layers of the model using their layer IDs and sets their corresponding weights using numpy arrays. It then applies the model to the colorized image using the forward() function and saves the output as a numpy array.

Finally, the code resizes the output to match the size of the original black and white image and displays the original and colorized images side by side using imshow() function. It waits for a key press to close the windows and then closes all windows using the destroyAllWindows() function.
