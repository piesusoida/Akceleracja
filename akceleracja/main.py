import ctypes
import numpy as np
from PIL import Image
import time
Image.MAX_IMAGE_PIXELS = None
# Load an image
img = Image.open("images/image64kx21k.png").convert("L") #greyscale
img_np = np.array(img, dtype=np.uint8)
height, width = img_np.shape

# Choose mode
lib = ctypes.CDLL("./singlethread.dll")
#lib = ctypes.CDLL("./multi_thread.dll")
#lib = ctypes.CDLL("./gputhread.dll")

# specify what is passed to the dll
lib.process_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
lib.process_image.restype = None

# Pass pointer to raw image data
ptr = img_np.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

# Call the chosen function
timer = time.time()
lib.process_image(ptr, width, height)
timer =time.time() -timer
# take the data from dll, and convert it back to an image
processed_img = Image.fromarray(img_np)
processed_img.save("output/processed7.png")
print(timer)
