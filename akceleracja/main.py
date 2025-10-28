import ctypes
import numpy as np
from PIL import Image

# Load an image and convert to grayscale
img = Image.open("images/image1.png").convert("L")
img_np = np.array(img, dtype=np.uint8)
height, width = img_np.shape

# Load C library
#lib = ctypes.CDLL("./singlethread.dll")
lib = ctypes.CDLL("./multi_thread.dll")
#lib = ctypes.CDLL("./gputhread.dll")

# Define C function signature
lib.process_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
lib.process_image.restype = None

# Pass pointer to raw image data
ptr = img_np.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

# Call the C function
lib.process_image(ptr, width, height)

# Optionally convert back to image
processed_img = Image.fromarray(img_np)
processed_img.save("output/processed2.png")
