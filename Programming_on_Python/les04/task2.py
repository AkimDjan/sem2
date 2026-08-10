import numpy as np

def pad_image(image: np.ndarray, pad_size: int) -> np.ndarray:

    if pad_size < 1:
        raise ValueError("Pad size must be more or equal 1")
    
    if len(image.shape)==2:
        new_image = np.zeros(shape=(len(image)+pad_size*2,len(image[0])+pad_size*2))
        for i in range(len(image)):
            for j in range(len(image[0])):
                new_image[i+pad_size,j+pad_size]=image[i,j]
    
    elif len(image.shape)==3:
        new_image = np.zeros(shape=(len(image),len(image[0])+pad_size*2,len(image[0,0])+pad_size*2))
        for i in range(len(image[0])):
            for j in range(len(image[0,0])):
                new_image[:, i+pad_size, j+pad_size]=image[:,i,j]
    
    else:
        raise ValueError("Unsupported image dimensions")
    
    return new_image
    

def blur_image(
    image: np.ndarray,
    kernel_size: int,
) -> np.ndarray:
    
    if kernel_size%2==0 or kernel_size<1:
        raise ValueError("Kernel size must be more or equal than 1 and odd")
    
    padded = pad_image(image, kernel_size//2)
    result = np.zeros_like(image)

    if len(image.shape)==2:
        rows, cols = image.shape
        for i in range(rows):
            for j in range(cols):
                blur_space = padded[i : (i + kernel_size), j : (j + kernel_size)]
                result[i, j] = np.mean(blur_space)
    elif len(image.shape)==3:
        rows, cols, rgb = image.shape
        for i in range(rows):
            for j in range(cols):
                blur_space = padded[i : (i + kernel_size), j : (j + kernel_size), :]
                result[i, j, :] = np.mean(blur_space, axis=(0, 1))
    else:
        raise ValueError("Image must be 2D or 3D")

    return result