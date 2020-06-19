import cv2 as cv
import imutils
#resizes the image passed using a and b as window 
def retresize(image,a,b):
    resized=cv.resize(image,(a,b))
    return resized
def retshow(image, title):
    cv.imshow(title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
    return image

# returns a gaussianblurred image


def retblur(image):
    blurred = cv.GaussianBlur(image, (7, 7), 0)
    return blurred

# returns a grayscale image


def retgray(image):
    grayed = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return grayed

# does canny edge detection using a and b as tolerances
# returns a canny version of the image


def retcanny(image, a, b):
    can = cv.Canny(image, a, b)
    return can

# canny object detection


def cannyobjectdetect(image, a, b):
    graypic = retgray(image)
    inverted = 255-graypic
    post = cv.blur(inverted, (2, 2))
    post = retcanny(post, a, b)
    contours, _ = cv.findContours(
        post, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    print("The number of objects found=", len(contours))
    cv.drawContours(image, contours, -1, (0, 255, 0), 2)
    cv.imshow("Highlighted objects", image)
    cv.waitKey(0)