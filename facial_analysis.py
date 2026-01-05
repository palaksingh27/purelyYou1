import cv2
import numpy as np
import logging
import math
from collections import Counter

logger = logging.getLogger(__name__)

def analyze_face(image):
    """
    Analyzes the facial features from an image using OpenCV.
    
    Args:
        image: OpenCV image in BGR format
        
    Returns:
        dict: Dictionary containing facial analysis results
    """
    try:
        # Log image shape for debugging
        logger.info(f"Input image shape: {image.shape}")
        
        # Make sure image is not too large (resize if needed)
        max_size = 1200
        if image.shape[0] > max_size or image.shape[1] > max_size:
            scale = max_size / max(image.shape[0], image.shape[1])
            new_width = int(image.shape[1] * scale)
            new_height = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_width, new_height))
            logger.info(f"Resized image to shape: {image.shape}")
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Load pre-trained face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Try different scale factors for better face detection
        scale_factors = [1.1, 1.05, 1.15, 1.2]
        min_neighbors_options = [4, 3, 5]
        
        faces = []
        for scale in scale_factors:
            for min_neighbors in min_neighbors_options:
                faces = face_cascade.detectMultiScale(gray, scale, min_neighbors)
                if len(faces) > 0:
                    logger.info(f"Face detected with scale={scale}, min_neighbors={min_neighbors}")
                    break
            if len(faces) > 0:
                break
        
        # If no face is detected, try to analyze the whole image instead
        if len(faces) == 0:
            logger.warning("No face detected in the image, analyzing the whole image instead")
            
            # Analyze the center portion of the image (assuming it contains the face)
            height, width = image.shape[:2]
            center_x, center_y = width // 2, height // 2
            margin = min(width, height) // 3  # Use a third of the image
            
            # Extract center region
            y_start = max(0, center_y - margin)
            y_end = min(height, center_y + margin)
            x_start = max(0, center_x - margin)
            x_end = min(width, center_x + margin)
            
            face_img = image[y_start:y_end, x_start:x_end]
            logger.info(f"Using center portion of image with dimensions: {face_img.shape}")
            
            # Continue with analysis on this region
        else:
            # Use the detected face
            x, y, w, h = faces[0]
            
            # Add margin to face for better analysis
            margin = int(0.1 * w)
            y_start = max(0, y - margin)
            y_end = min(image.shape[0], y + h + margin)
            x_start = max(0, x - margin)
            x_end = min(image.shape[1], x + w + margin)
            
            face_img = image[y_start:y_end, x_start:x_end]
            logger.info(f"Detected face with dimensions: {face_img.shape}")
        
# This section is no longer needed as we already extracted the face_img
        
        # Extract features
        features = {}
        
        # Analyze skin tone using color analysis
        features['skin_tone'] = analyze_skin_tone(face_img)
        logger.info(f"Detected skin tone: {features['skin_tone']}")
        
        # Analyze skin type based on image texture
        features['skin_type'] = analyze_skin_type(face_img)
        logger.info(f"Detected skin type: {features['skin_type']}")
        
        # Detect skin concerns
        features['concerns'] = detect_skin_concerns(face_img)
        logger.info(f"Detected concerns: {features['concerns']}")
        
        return features
        
    except Exception as e:
        logger.error(f"Error in facial analysis: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # Return default values as fallback
        # Use color distribution of the image to make a reasonable guess
        try:
            # Analyze the whole image instead for basic color information
            img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(img_hsv)
            
            # Get average values for basic analysis
            h_mean = np.mean(h)
            s_mean = np.mean(s)
            v_mean = np.mean(v)
            
            # Determine skin tone based on brightness and hue
            skin_tone = "medium"  # Default
            if v_mean > 200:
                skin_tone = "fair-neutral"
            elif v_mean > 170:
                skin_tone = "light-neutral"
            elif v_mean < 120:
                skin_tone = "deep-warm"
            
            # Determine skin type based on saturation
            skin_type = "normal"  # Default
            if s_mean > 70:
                skin_type = "oily"
            elif s_mean < 40:
                skin_type = "dry"
            
            # Basic concerns based on image properties
            concerns = ["texture"]
            if np.std(v) > 50:  # High variance in brightness
                concerns.append("uneven tone")
            if v_mean < 130:
                concerns.append("dullness")
            
            return {
                'skin_type': skin_type,
                'skin_tone': skin_tone,
                'concerns': concerns
            }
        except Exception as nested_e:
            logger.error(f"Error in fallback analysis: {str(nested_e)}")
            # Absolute last resort
            return {
                'skin_type': 'normal',
                'skin_tone': 'medium',
                'concerns': ['general care']
            }

def analyze_skin_tone(face_img):
    """
    Analyze the skin tone by sampling color from cheek regions and forehead
    with more sophisticated skin detection
    """
    # Convert to multiple color spaces for better analysis
    hsv_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    ycrcb_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    
    # Get image dimensions
    height, width = face_img.shape[:2]
    
    # Define skin sampling regions
    # Cheeks and forehead are good areas for skin tone analysis
    left_cheek_x = int(width * 0.25)
    right_cheek_x = int(width * 0.75)
    cheek_y = int(height * 0.55)
    forehead_y = int(height * 0.2)
    forehead_x = int(width * 0.5)
    
    # Sample size
    sample_size = 30
    
    # Extract sample regions
    left_sample = hsv_img[cheek_y-sample_size//2:cheek_y+sample_size//2, 
                          left_cheek_x-sample_size//2:left_cheek_x+sample_size//2]
    
    right_sample = hsv_img[cheek_y-sample_size//2:cheek_y+sample_size//2, 
                           right_cheek_x-sample_size//2:right_cheek_x+sample_size//2]
    
    forehead_sample = hsv_img[forehead_y-sample_size//2:forehead_y+sample_size//2,
                             forehead_x-sample_size//2:forehead_x+sample_size//2]
    
    # Create skin mask to filter out non-skin pixels
    # YCrCb is excellent for skin detection
    min_YCrCb = np.array([0, 135, 85], np.uint8)
    max_YCrCb = np.array([255, 180, 135], np.uint8)
    
    # Apply skin mask to samples using YCrCb space
    left_ycrcb = ycrcb_img[cheek_y-sample_size//2:cheek_y+sample_size//2, 
                          left_cheek_x-sample_size//2:left_cheek_x+sample_size//2]
    right_ycrcb = ycrcb_img[cheek_y-sample_size//2:cheek_y+sample_size//2, 
                           right_cheek_x-sample_size//2:right_cheek_x+sample_size//2]
    forehead_ycrcb = ycrcb_img[forehead_y-sample_size//2:forehead_y+sample_size//2,
                              forehead_x-sample_size//2:forehead_x+sample_size//2]
    
    # Create masks for each sample
    left_mask = cv2.inRange(left_ycrcb, min_YCrCb, max_YCrCb)
    right_mask = cv2.inRange(right_ycrcb, min_YCrCb, max_YCrCb)
    forehead_mask = cv2.inRange(forehead_ycrcb, min_YCrCb, max_YCrCb)
    
    # Apply masks to HSV samples
    left_hsv_masked = cv2.bitwise_and(left_sample, left_sample, mask=left_mask)
    right_hsv_masked = cv2.bitwise_and(right_sample, right_sample, mask=right_mask)
    forehead_hsv_masked = cv2.bitwise_and(forehead_sample, forehead_sample, mask=forehead_mask)
    
    # Take a simpler approach that doesn't rely on combining arrays of different shapes
    # Get all pixels from the face image for analysis
    all_pixels = face_img.reshape(-1, 3)  # Reshape to a 2D array of pixels
    
    # Convert all pixels to HSV
    all_hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV).reshape(-1, 3)
    
    # Filter skin-colored pixels using HSV ranges
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([30, 255, 255], dtype=np.uint8)
    
    skin_mask = cv2.inRange(all_hsv.reshape(face_img.shape), lower_skin, upper_skin)
    skin_pixels = all_hsv[skin_mask.flatten() > 0]
    
    # If not enough skin pixels are found, use the whole image
    if skin_pixels.shape[0] < 100:
        valid_pixels = all_hsv
    else:
        valid_pixels = skin_pixels
    
    # Calculate average color
    average_color = np.mean(valid_pixels, axis=0)
    
    # Extract hue, saturation, value
    h, s, v = average_color
    
    # Enhanced skin tone categories with undertones
    if v < 110:  # Very low brightness
        if h < 15 or h > 165:  # Red undertones
            return "deep-warm"
        else:
            return "deep-cool"
    elif 110 <= v < 140:
        if h < 20:  # Reddish/golden undertone
            return "warm-tan"
        else:
            return "neutral-tan"
    elif 140 <= v < 170:
        if h < 17:  # More yellow/golden undertone
            return "medium-warm"
        elif 17 <= h < 25:
            return "medium-neutral"
        else:
            return "medium-cool"
    elif 170 <= v < 210:
        if s < 60:  # Lower saturation
            if h < 17:
                return "light-warm"
            else:
                return "light-cool"
        else:
            return "light-neutral"
    else:  # High brightness
        if s < 50:  # Low saturation with high value indicates fair skin
            if h < 15:
                return "fair-warm"
            else:
                return "fair-cool"
        else:
            return "fair-neutral"

def analyze_skin_type(face_img):
    """
    Analyze the skin type based on enhanced texture features and
    t-zone vs cheek comparison for better combination skin detection
    """
    # Get image dimensions
    height, width = face_img.shape[:2]
    
    # Convert to grayscale
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Convert to multiple color spaces for analysis
    ycrcb = cv2.cvtColor(face_img, cv2.COLOR_BGR2YCrCb)
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Define different facial regions for analysis
    # T-zone (forehead and nose - typically oilier)
    t_zone_forehead = gray[int(height*0.1):int(height*0.3), int(width*0.3):int(width*0.7)]
    t_zone_nose = gray[int(height*0.3):int(height*0.6), int(width*0.4):int(width*0.6)]
    
    # Cheeks (can be drier than t-zone)
    left_cheek = gray[int(height*0.4):int(height*0.6), int(width*0.1):int(width*0.3)]
    right_cheek = gray[int(height*0.4):int(height*0.6), int(width*0.7):int(width*0.9)]
    
    # Use Laplacian for texture detection
    laplacian_full = cv2.Laplacian(blurred, cv2.CV_64F)
    
    # Calculate various texture metrics
    texture_variance = np.var(laplacian_full)
    
    # Analyze t-zone brightness (proxy for oiliness)
    t_zone_brightness = np.mean(np.hstack([t_zone_forehead.flatten(), t_zone_nose.flatten()]))
    
    # Analyze cheek brightness
    cheek_brightness = np.mean(np.hstack([left_cheek.flatten(), right_cheek.flatten()]))
    
    # Calculate brightness differential (higher in combination skin)
    brightness_diff = abs(t_zone_brightness - cheek_brightness)
    
    # Analyze high frequency texture components using FFT
    # This can help detect fine lines and dryness
    f_transform = np.fft.fft2(blurred)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # High frequency components are in the outer regions of the spectrum
    height_f, width_f = magnitude_spectrum.shape
    center_y, center_x = height_f // 2, width_f // 2
    radius = min(center_y, center_x) // 2
    
    # Create a mask for high frequency components
    y, x = np.ogrid[:height_f, :width_f]
    mask_area = (y - center_y) ** 2 + (x - center_x) ** 2 > radius ** 2
    
    # Extract high frequency content
    high_freq_energy = np.mean(magnitude_spectrum[mask_area])
    
    # Oil analysis from HSV saturation and value
    s_channel = hsv[:, :, 1]  # Saturation channel
    v_channel = hsv[:, :, 2]  # Value channel
    
    s_mean = np.mean(s_channel)
    v_mean = np.mean(v_channel)
    
    # Combine all metrics for classification
    # Use a weighted approach
    
    # Check for combination skin first (significant t-zone vs cheek difference)
    if brightness_diff > 12:
        return "combination"
    
    # Check for dry skin (high texture variance, lower brightness, high frequency details)
    if texture_variance > 100 and t_zone_brightness < 130 and high_freq_energy > 20:
        if s_mean < 50:  # Lower saturation can indicate dryness
            return "very-dry"
        return "dry"
    
    # Check for oily skin (high brightness, smoother texture)
    if t_zone_brightness > 150 and texture_variance < 80:
        if v_mean > 160 and s_mean > 60:  # Higher saturation and value can indicate oiliness
            return "very-oily"
        return "oily"
    
    # Check for sensitive skin (can be detected by red channel analysis)
    red_channel = face_img[:, :, 2]  # BGR format, red is index 2
    red_variance = np.var(red_channel)
    if red_variance > 1000:  # High variance in red channel can indicate sensitivity
        return "sensitive"
    
    # Default to normal if no strong indicators
    return "normal"

def detect_skin_concerns(face_img):
    """
    Detect common skin concerns with advanced image processing techniques
    """
    concerns = []
    concern_confidence = {}
    
    # Convert to appropriate color spaces for different analyses
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(face_img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(face_img, cv2.COLOR_BGR2LAB)
    
    # Get image dimensions
    height, width = face_img.shape[:2]
    
    # Define regions for targeted analysis
    forehead = gray[int(height * 0.1):int(height * 0.3), int(width * 0.3):int(width * 0.7)]
    cheeks = np.vstack([
        gray[int(height * 0.4):int(height * 0.6), int(width * 0.1):int(width * 0.3)],  # left cheek
        gray[int(height * 0.4):int(height * 0.6), int(width * 0.7):int(width * 0.9)]   # right cheek
    ])
    
    # 1. AGING/WRINKLE DETECTION - Enhanced with HOG and multi-scale edge detection
    
    # Apply different edge detection methods for better wrinkle detection
    edges_canny = cv2.Canny(gray, 50, 150)
    
    # Calculate Sobel gradients for directional edge detection (horizontal wrinkles)
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    
    # Absolute gradient magnitudes
    abs_sobel_x = cv2.convertScaleAbs(sobel_x)
    abs_sobel_y = cv2.convertScaleAbs(sobel_y)
    
    # Compute overall gradient magnitude
    gradient_mag = cv2.addWeighted(abs_sobel_x, 0.5, abs_sobel_y, 0.5, 0)
    
    # Look for horizontal lines (potential forehead wrinkles)
    horizontal_density = np.sum(sobel_y) / (sobel_y.shape[0] * sobel_y.shape[1])
    
    # Analyze forehead region specifically for wrinkles
    forehead_edges = cv2.Canny(forehead, 30, 150)
    forehead_edge_density = np.sum(forehead_edges) / (forehead_edges.shape[0] * forehead_edges.shape[1])
    
    # Combine wrinkle detection metrics
    wrinkle_score = edge_density = np.sum(edges_canny) / (edges_canny.shape[0] * edges_canny.shape[1])
    wrinkle_score = wrinkle_score * 0.7 + forehead_edge_density * 0.3  # Weight forehead higher
    
    if wrinkle_score > 5:
        confidence = min(100, wrinkle_score * 10)
        concerns.append("aging")
        concern_confidence["aging"] = confidence
        if horizontal_density > 0.2:
            concerns.append("wrinkles")
            concern_confidence["wrinkles"] = confidence
    
    # 2. ACNE/BLEMISH DETECTION - Improved with better color segmentation
    
    # Red/inflamed areas in HSV (more precise ranges)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    
    lower_red2 = np.array([160, 70, 50])  # Adjusted range
    upper_red2 = np.array([180, 255, 255])
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    
    # Combine masks
    red_mask = red_mask1 + red_mask2
    
    # Apply morphological operations to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_OPEN, kernel)  # Remove noise
    red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel)  # Close small holes
    
    # Count spots/acne (connected components analysis)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(red_mask, connectivity=8)
    
    # Filter by size (small spots are likely noise, large spots are likely acne)
    acne_count = 0
    for i in range(1, num_labels):  # Skip background (label 0)
        if stats[i, cv2.CC_STAT_AREA] > 30 and stats[i, cv2.CC_STAT_AREA] < 500:
            acne_count += 1
    
    red_ratio = np.sum(red_mask) / (red_mask.shape[0] * red_mask.shape[1])
    
    if red_ratio > 0.03 or acne_count > 2:
        concerns.append("acne")
        concern_confidence["acne"] = min(100, max(red_ratio * 1000, acne_count * 10))
        
        if red_ratio > 0.05:
            concerns.append("redness")
            concern_confidence["redness"] = min(100, red_ratio * 1200)
    
    # 3. UNEVEN SKIN TONE DETECTION - Enhanced with LAB color space and local analysis
    
    # L channel variation indicates lightness/darkness unevenness
    l_channel = lab[:, :, 0]
    l_std = np.std(l_channel)
    
    # A channel variation can indicate redness unevenness
    a_channel = lab[:, :, 1]
    a_std = np.std(a_channel)
    
    # Calculate local variation (using sliding window std dev)
    def local_std_dev(img, window_size=20):
        h, w = img.shape
        result = np.zeros_like(img, dtype=np.float32)
        pad = window_size // 2
        padded = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_REFLECT)
        
        for i in range(h):
            for j in range(w):
                window = padded[i:i+window_size, j:j+window_size]
                result[i, j] = np.std(window)
        
        return np.mean(result)
    
    local_l_std = local_std_dev(l_channel)
    
    unevenness_score = l_std * 0.6 + a_std * 0.3 + local_l_std * 0.1
    
    if unevenness_score > 18:
        concerns.append("uneven tone")
        concern_confidence["uneven tone"] = min(100, unevenness_score * 3)
        
        # Check for hyperpigmentation
        if l_std > 25 and np.min(l_channel) < 70:
            concerns.append("hyperpigmentation")
            concern_confidence["hyperpigmentation"] = min(100, l_std * 2.5)
    
    # 4. TEXTURE ISSUES DETECTION - Using wavelet analysis
    
    # Simplify with variance of Laplacian for texture
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
    texture_var = np.var(laplacian)
    
    # Analyze high frequency components using FFT (correlates with rough texture)
    f_transform = np.fft.fft2(blurred)
    f_shift = np.fft.fftshift(f_transform)
    magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1)
    
    # Measure high frequency energy
    height_f, width_f = magnitude_spectrum.shape
    center_y, center_x = height_f // 2, width_f // 2
    radius = min(center_y, center_x) // 2
    
    y, x = np.ogrid[:height_f, :width_f]
    high_freq_mask = (y - center_y) ** 2 + (x - center_x) ** 2 > radius ** 2
    
    high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
    
    if texture_var > 300 or high_freq_energy > 25:
        concerns.append("texture")
        concern_confidence["texture"] = min(100, max(texture_var / 10, high_freq_energy * 2))
    
    # 5. DRYNESS DETECTION - Cheek analysis
    
    # Calculate brightness in the cheek regions (drier skin is often less bright)
    cheek_brightness = np.mean(cheeks)
    
    # Check for dry patches using texture analysis on cheeks
    cheeks_blurred = cv2.GaussianBlur(cheeks, (5, 5), 0)
    cheeks_laplacian = cv2.Laplacian(cheeks_blurred, cv2.CV_64F)
    cheeks_texture = np.var(cheeks_laplacian)
    
    if cheek_brightness < 130 and cheeks_texture > 200:
        concerns.append("dryness")
        dryness_score = ((130 - cheek_brightness) * 0.5) + (cheeks_texture / 10)
        concern_confidence["dryness"] = min(100, dryness_score)
    
    # 6. DULLNESS DETECTION - Based on brightness and saturation
    
    # Analyze overall brightness
    brightness = np.mean(gray)
    
    # Analyze saturation (less saturated skin can appear dull)
    s_channel = hsv[:, :, 1]
    s_mean = np.mean(s_channel)
    
    if brightness < 120 or s_mean < 20:
        concerns.append("dullness")
        dullness_score = ((120 - brightness) * 0.7) + ((20 - s_mean) * 1.5)
        concern_confidence["dullness"] = min(100, max(10, dullness_score))
    
    # 7. SENSITIVITY DETECTION - Red channel analysis
    
    # Red channel in BGR format
    red_channel = face_img[:, :, 2]
    red_variance = np.var(red_channel)
    
    # A channel from LAB space (red-green)
    a_channel_mean = np.mean(a_channel)
    
    if red_variance > 800 and a_channel_mean > 128:
        concerns.append("sensitivity")
        sensitivity_score = (red_variance / 100) * 0.6 + ((a_channel_mean - 128) * 2)
        concern_confidence["sensitivity"] = min(100, sensitivity_score)
    
    # 8. PORE SIZE DETECTION
    
    # Detect pores using morphological operations
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Invert if needed (pores should be dark spots)
    if np.mean(binary) > 127:
        binary = cv2.bitwise_not(binary)
    
    # Use tophat to detect small dark spots
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    tophat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
    
    _, pores = cv2.threshold(tophat, 15, 255, cv2.THRESH_BINARY)
    
    # Count and measure pores
    num_pores, pore_labels, pore_stats, _ = cv2.connectedComponentsWithStats(pores, connectivity=8)
    
    # Filter by size and count
    large_pores = 0
    for i in range(1, num_pores):  # Skip background
        if 10 < pore_stats[i, cv2.CC_STAT_AREA] < 100:
            large_pores += 1
    
    if large_pores > 10:
        concerns.append("enlarged pores")
        concern_confidence["enlarged pores"] = min(100, large_pores * 3)
    
    # If no concerns detected, skin is likely healthy
    if not concerns:
        concerns.append("healthy")
        concern_confidence["healthy"] = 100
    
    # Sort concerns by confidence
    sorted_concerns = sorted(concerns, key=lambda x: concern_confidence.get(x, 0), reverse=True)
    
    # Return sorted concerns (most confident first)
    return sorted_concerns[:5]  # Limit to top 5 concerns
