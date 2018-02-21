import numpy as np
import cv2

from typing import List, Union

class Augmentor:
    """
    Class creating a data augmentation object based on specified class attributes. 
    
    Args:
        rotation_range (int): Degree range for random rotations.
        shear_range (float): Shear intensity expressed as pixel shift to coor-
                             dinates of a triangle given på [[5, 5], [20, 5], 
                             [5, 20]]
        shift_range (float): Range for random horizontal and vertical shifts
                             in pixels.
        gaussian_blur_sd (float): Standard deviation range for the gauss distribution 
                                  for Gaussian blurring.
        gaussian_noise_sd (float): Standard deviation range for the gauss distribution 
                                for Gaussian noise generation.
        salt_and_pepper_noise_intensity (int): Number of pixels to add salt and pepper
                                               noise to.
        rgb_color_shift (int): Range for shifting all color channels (RGB)
        border_col (List[int]): List with RGB values for border when  shifting and
                                shearing images. Default is black.
        
    Returns:
        Initialises an augmentation object to which images can be passed for aumentation
    
    Raises:
        TypeError: Raises an exception.
    """
    
    def __init__(self, rotation_range: int = 0, shear_range: float = 0, shift_range: float = 0,
                 gaussian_blur_sd: float = 0, gauss_noise_sd: float = 0,
                 salt_and_pepper_noise_intensity: int = 0, rgb_color_shift: int = 0,
                 border_col: List[int] = [0, 0, 0]):
        
        self.rotation_range = rotation_range
        self.shear_range = shear_range
        self.shift_range = shift_range
        self.gaussian_blur_sd = gaussian_blur_sd
        self.gauss_noise_sd = gauss_noise_sd
        self.salt_and_pepper_noise_intensity = salt_and_pepper_noise_intensity
        self.rgb_color_shift = rgb_color_shift
        self.border_col = border_col
        
    def _rotate(self) -> List[List[Union[float, int]]]:
        """
        Random rotation matrix
        
        Args:
            None
            
        Returns:
            A random rotation matrix for use in affine transformation
        """
        
        # Generate random rotation angle
        rotate_angle = np.random.uniform(-self.rotation_range, self.rotation_range)
        
        # Generate matrix
        rotation = cv2.getRotationMatrix2D((self.shape[1] / 2, self.shape[2] / 2), rotate_angle, 1)
        
        return rotation
    
    def _shear(self) -> List[List[Union[float, int]]]:
        """
        Random shear matrix
        
        Args:
            None
            
        Returns:
            A random shear matrix for use in affine transformation
        """
        
        # Define a traingle in the original image
        points_from = np.float32([[5, 5], [20, 5], [5, 20]])
        
        # Shift traingle, but center two coordinates
        point_1 = 5 + np.random.uniform(-self.shear_range, self.shear_range)
        point_2 = 20 + np.random.uniform(-self.shear_range, self.shear_range)
        
        # Define ned triangle
        points_to = np.float32([[point_1, 5], [point_2, point_1], [5, point_2]])
        
        # Compute shear matrix
        shearing = cv2.getAffineTransform(points_from, points_to)

        return shearing
    
    def _translate(self) -> List[List[Union[float, int]]]:
        """
        Random shifting matrix
        
        Args:
            None
            
        Returns:
            A random shifting matrix for use in affine transformation
        """
        translate_x = np.random.uniform(-self.shift_range, self.shift_range)
        translate_y = np.random.uniform(-self.shift_range, self.shift_range)
        translation = np.float32([[1, 0, translate_x], [0, 1, translate_y]])
        
        return translation
    
    def _warp_affine(self, img: np.ndarray, transformation_matrix: List[List[Union[float, int]]]) -> np.ndarray:
        """
        Wrapper around cv2's affine transformation function
        
        Args:
            img (np.ndarray): A three dimensional tensor (height, width, channels)
            transformation_matrix (List[List[Union[float, int]]]): A transformation matrix to
                                                                   pass to the affine trans-
                                                                   formation
            
        Returns:
            The image (np.ndarray), where the given affine transformation has been applied.
        """
        
        img = cv2.warpAffine(img,
                             transformation_matrix,
                             (self.shape[1], self.shape[0]),
                             borderMode = cv2.BORDER_CONSTANT,
                             borderValue = tuple(self.border_col))
        
        return img
    
    def _color_shift(self) -> np.ndarray:
        """
        Shift color in a given random direction
        
        Args:
            None
            
        Returns:
            A matrix that can be added to the image to shift the colors of all channels
        """
        
        correction = np.random.uniform(-self.rgb_color_shift, self.rgb_color_shift)
        
        return np.ones(self.shape) * correction
    
    def _blur(self, img: np.ndarray) -> np.ndarray:
        """
        Add gausian blurring in a 5x5 kernel
        
        Args:
            img (np.ndarray): A three dimensional tensor (height, width, channels)
            
        Returns:
            The image (np.ndarray) where Gassian blur have been added.
        """
        
        rand_sd = np.random.randn() * self.gaussian_blur_sd
        
        return cv2.GaussianBlur(img, (5, 5), rand_sd)

    def _salt_and_pepper_noise(self, img: np.ndarray) -> np.ndarray:
        """
        Generate a salt and pepper noise matrix
        
        Args:
            img (np.ndarray): A three dimensional tensor (height, width, channels)
            
        Returns:
            The image (np.ndarray) where salt and noise have been added.
        """
        
        salt_vs_pepper = 0.5
        out = np.copy(img)
        
        # Salt mode
        num_salt = np.ceil(self.salt_and_pepper_noise_intensity * self.size * salt_vs_pepper)
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in self.shape[:2]]
        out[coords] = 255

        # Pepper mode
        num_pepper = np.ceil(self.salt_and_pepper_noise_intensity* self.size * (1. - salt_vs_pepper))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in self.shape[:2]]
        out[coords] = 0
        
        return out
    
    def _gauss_noise(self, mean: Union[int, float] = 0) -> np.ndarray:
        """
            Generate a gausian distributed noise matrix
        
        Args:
            None
            
        Returns:
            A matrix with Gausian distributed noise that can be added to the image.
        """
        
        sd = self.gauss_noise_sd
        
        return np.random.normal(mean, 1, self.shape) * sd
    
    def augment(self, images: np.ndarray) -> np.ndarray:
        """
            Run image augmentation
        
        Args:
            images (np.ndarray): A four dimensional tensor of images, as 
                                 (batch, height, width, channels)
            
        Returns:
            A matrix with Gausian distributed noise that can be added to the image.
        """
        
        self.shape = images[0].shape
        self.size = images[0].size
        
        augmented_list = []
        for i in range(images.shape[0]):
            img = images[i]
            
            # Affine transformations
            img = self._warp_affine(img, self._rotate())
            img = self._warp_affine(img, self._translate())
            img = self._warp_affine(img, self._shear())
            
            # Gaussian blur
            if self.gaussian_blur_sd > 0:
                img = self._blur(img)
            
            # Gaussian noise
            img = img + self._gauss_noise()
            
            # Salt and pepper noise
            img = self._salt_and_pepper_noise(img)
            
            # Shift color in random direction
            img = img + self._color_shift()
            
            # Clip values betweeen 0 and 255 - Ensure datatype is 
            # unsigned integer in the range 0 to 255.
            img = np.clip(img, 0, 255).astype(np.uint8)
            
            augmented_list.append(img)
        
        return np.stack(augmented_list, axis = 0)
    