import cv2 as cv
from PIL import ImageGrab
class Capture:
    current_image = None

    def __init__(self):
        print("Object <Class name :: Capture> Created")

    def capture_current_image(self):
        self.current_image = ImageGrab.grab()
        print(type(self.current_image))

    def get_current_image(self):
        return self.current_image


