# if function cannot apply, use keyword arguments:
grayscale = "rgb"
img_width = 224
img_height = 224


def preprocess_image(image_path, target_height=img_height, target_width=img_width):
    """
    This function resizes and converts the image to greyscale. By default, resize all images to 224 X 224.
    """

    import cv2
    # Read the image in color format
    image = cv2.imread(image_path)

    # Resize the image to the target dimensions using bilinear interpolation
    resized = cv2.resize(grayscale, (target_width, target_height), interpolation=cv2.INTER_LINEAR)

    # Normalize the image by dividing by 255 to scale pixel values between 0 and 1
    normalized_img = resized / 255.0

    return normalized_img

