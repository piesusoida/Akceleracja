import ctypes
import numpy as np
from PIL import Image
import time
Image.MAX_IMAGE_PIXELS = None
# Load an image
def process_gpu (img, i):

    
    
    img_np = np.array(img, dtype=np.uint8)
    height, width = img_np.shape

    # Choose mode
    #lib = ctypes.CDLL("./singlethread.dll")
    #lib = ctypes.CDLL("./multi_thread.dll")
    lib = ctypes.CDLL("./gputhread.dll")

    # specify what is passed to the dll
    lib.process_image.argtypes = [ctypes.POINTER(ctypes.c_ubyte), ctypes.c_int, ctypes.c_int]
    lib.process_image.restype = None

    # Pass pointer to raw image data
    ptr = img_np.ctypes.data_as(ctypes.POINTER(ctypes.c_ubyte))

    # Call the chosen function
    time.sleep(2) # sleep to make the nsight more clear
    timer = time.time()
    lib.process_image(ptr, width, height)
    timer =time.time() -timer
    time.sleep(2) # sleep to make the nsight more clear
    # take the data from dll, and convert it back to an image
    processed_img = Image.fromarray(img_np)
    processed_img.save(f"output/processed_gpu{i}.png")
    print(timer)

def process_singlethread (img, i):
    
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
    time.sleep(2) # sleep to make the nsight more clear
    timer = time.time()
    lib.process_image(ptr, width, height)
    timer =time.time() -timer
    time.sleep(2) # sleep to make the nsight more clear
    # take the data from dll, and convert it back to an image
    processed_img = Image.fromarray(img_np)
    processed_img.save(f"output/processed_singlethread{i}.png")
    print(timer)

def main():
    for j in range (2):
        for i in range(6):
            if i == 0:
                img = Image.open("images/image800.png").convert("L") #greyscale
            if i == 1:
                img = Image.open("images/image3k.webp").convert("L") #greyscale
            if i == 2:
                img = Image.open("images/image6k.jpg").convert("L") #greyscale
            if i == 3:
                img = Image.open("images/image100k.jpg").convert("L") #greyscale
            if i == 4:
                img = Image.open("images/image200k.jpg").convert("L") #greyscale
            if i == 5:
                img = Image.open("images/image64kx21k.png").convert("L") #greyscale
            if j==0:
                process_gpu(img,i)
            elif j==1:
                process_singlethread(img, i)
        

if __name__ == "__main__":
    main()