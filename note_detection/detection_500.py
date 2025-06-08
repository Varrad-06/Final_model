import cv2
import numpy as np
import os
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pytesseract
from PIL import Image
import re

# Set the path to the Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

class CurrencyDetector500:
    def __init__(self, image_input):
        self.path = image_input
        if isinstance(image_input, str):
            self.test_img = cv2.imread(image_input)
        else:
            image_input.seek(0)
            in_memory_file = np.frombuffer(image_input.read(), np.uint8)
            self.test_img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)
        self.score_set_list = []
        self.best_extracted_img_list = []
        self.avg_ssim_list = []
        self.left_BL_result = []
        self.right_BL_result = []
        self.result_list = []
        self.number_panel_result = []
        self.result_images = {}
        
        # Resize and preprocess the image
        self.test_img = cv2.resize(self.test_img, (1167, 519))
        self.blur_test_img = cv2.GaussianBlur(self.test_img, (5, 5), 0)
        self.gray_test_image = cv2.cvtColor(self.blur_test_img, cv2.COLOR_BGR2GRAY)
        
        # Values for specifying search area of features 1 to 7
        self.search_area_list = [
            [200, 300, 200, 370],
            [1050, 1500, 300, 450],
            [100, 450, 20, 120],
            [690, 1050, 20, 120],
            [820, 1050, 350, 430],
            [700, 810, 330, 430],
            [400, 650, 0, 100]
        ]
        
        # Values of max_area and min_area for each feature for features 1 to 7
        self.feature_area_limits_list = [
            [12000, 17000],
            [10000, 18000],
            [20000, 30000],
            [24000, 36000],
            [15000, 25000],
            [7000, 13000],
            [11000, 18000]
        ]
        
        self.NUM_OF_FEATURES = 7
        
    def calculate_ssim(self, template_img, query_img):
        min_w = min(template_img.shape[1], query_img.shape[1])
        min_h = min(template_img.shape[0], query_img.shape[0])
        
        # Resizing the two images so that both have same dimensions
        img1 = cv2.resize(template_img, (min_w, min_h))
        img2 = cv2.resize(query_img, (min_w, min_h))
        
        # Conversion to gray-scale
        img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        
        # Find the SSIM score and return
        score = ssim(img1, img2)
        return score
    
    def compute_orb(self, template_img, query_img):
        # Creating orb object
        nfeatures = 700
        scaleFactor = 1.2
        nlevels = 8
        edgeThreshold = 15
        
        # Initialize the ORB detector algorithm 
        orb = cv2.ORB_create(
            nfeatures,
            scaleFactor,
            nlevels,
            edgeThreshold
        )
        
        # Find the keypoints and descriptors with ORB
        kpts1, descs1 = orb.detectAndCompute(template_img, None)
        kpts2, descs2 = orb.detectAndCompute(query_img, None)
        
        # Return early if no keypoints found
        if kpts1 is None or kpts2 is None or len(kpts1) < 4 or len(kpts2) < 4:
            return None, None, kpts1, kpts2, None
        
        # Brute Force Matching
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Finding matches between the 2 descriptor sets
        matches = bf.match(descs1, descs2)
        
        # Sort the matches in the order of their distance
        dmatches = sorted(matches, key=lambda x: x.distance)
        
        # Return early if not enough matches
        if len(dmatches) < 4:
            return None, None, kpts1, kpts2, dmatches
        
        # Image homography
        # extract the matched keypoints
        src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)
        
        # find homography matrix and do perspective transform
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        h, w = template_img.shape[:2]
        pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
        
        if M is not None:
            dst = cv2.perspectiveTransform(pts, M)
        else:
            dst = None
            
        # Returning necessary data
        return dst, dst_pts, kpts1, kpts2, dmatches
    
    def test_feature_1_to_7(self):
        NUMBER_OF_TEMPLATES = 6
        self.score_set_list = []
        self.best_extracted_img_list = []
        self.avg_ssim_list = []
        
        # Create the Dataset structure if it doesn't exist
        base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'Dataset')
        
        # Create sample data path
        features_dataset_path = os.path.join(base_path, '500_Features Dataset')
        
        # Skip template check if dataset doesn't exist
        if not os.path.exists(features_dataset_path):
            print("Dataset not found. Creating placeholder results for visualization.")
            for j in range(self.NUM_OF_FEATURES):
                feature_num = j + 1
                print(f'ANALYSIS OF FEATURE {feature_num} - Using placeholder')
                
                # Add dummy values when dataset is missing
                self.score_set_list.append([0.3])  # Use a value below threshold
                self.avg_ssim_list.append(0.3)
                
                # Create a placeholder image with feature number
                placeholder_img = np.ones((100, 100, 3), dtype=np.uint8) * 255
                cv2.putText(placeholder_img, f"F{feature_num}", (30, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
                
                self.best_extracted_img_list.append([placeholder_img, 0.3])
                
                # Store in result_images for display
                self.result_images[f'feature_{feature_num}'] = {
                    'template': placeholder_img,
                    'detected': placeholder_img,
                    'score': 0.3  # Below threshold but visible for UI
                }
            return
                
        # Iterating for each feature
        for j in range(self.NUM_OF_FEATURES):
            print(f'ANALYSIS OF FEATURE {j+1}')
            
            score_set = []           # SSIM scores for each template of current feature will be stored here
            max_score = -1           # Stores max SSIM score
            max_score_img = None     # Stores extracted image with max SSIM score for the current feature
            
            # Performing feature detection, extraction and comparison for each template stored in dataset 
            for i in range(NUMBER_OF_TEMPLATES):
                template_path = os.path.join(base_path, f'500_Features Dataset/Feature {j+1}/{i+1}.jpg')
                
                if not os.path.exists(template_path):
                    print(f"Template not found: {template_path}")
                    continue
                
                template_img = cv2.imread(template_path)
                
                template_img_blur = cv2.GaussianBlur(template_img, (5, 5), 0)
                template_img_gray = cv2.cvtColor(template_img_blur, cv2.COLOR_BGR2GRAY)
                
                test_img_mask = self.gray_test_image.copy()
                
                # Creating a mask to search the current template
                search_area = self.search_area_list[j]
                
                test_img_mask[:, :search_area[0]] = 0
                test_img_mask[:, search_area[1]:] = 0
                test_img_mask[:search_area[2], :] = 0
                test_img_mask[search_area[3]:, :] = 0
                
                # Feature detection using ORB 
                dst, dst_pts, kpts1, kpts2, dmatches = self.compute_orb(template_img_gray, test_img_mask)
                
                # Error handling
                if dst is None:
                    print('An Error occurred - Homography matrix is of NoneType')
                    continue
                
                query_img = self.test_img.copy()
                
                # Drawing polygon around the region where the current template has been detected on the test currency note
                res_img1 = cv2.polylines(query_img, [np.int32(dst)], True, (0, 0, 255), 1, cv2.LINE_AA)
                
                # Find the details of a bounding rectangle that bounds the above polygon
                (x, y, w, h) = cv2.boundingRect(dst)
                
                # Checking if the area of the detected region is within the min and max area allowed for current feature 
                min_area = self.feature_area_limits_list[j][0]
                max_area = self.feature_area_limits_list[j][1]
                
                feature_area = w*h
                if feature_area < min_area or feature_area > max_area:
                    (x, y, w, h) = cv2.boundingRect(dst_pts)
                    
                    feature_area = w*h
                    if feature_area < min_area or feature_area > max_area:
                        print('Template Discarded- Area of extracted feature is outside permitted range!')
                        continue
                
                # Draw the rectangle
                cv2.rectangle(res_img1, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # SSIM
                # Crop out the region inside the green rectangle (matched region)
                crop_img = self.blur_test_img[y:y+h, x:x+w]
                
                score = self.calculate_ssim(template_img_blur, crop_img)
                
                score_set.append(score)
                print(f'SSIM score: {score}\n')
                
                # Keeping details about extracted region with highest SSIM score
                if score > max_score:
                    max_score = score
                    max_score_img = crop_img
                    
                    # Save matched feature image for display
                    if i == 0:  # Save only the first template match for simplicity
                        self.result_images[f'feature_{j+1}'] = {
                            'template': template_img,
                            'detected': crop_img,
                            'score': score
                        }
                
            # After all templates for this feature
            self.score_set_list.append(score_set)
            print(f'SSIM score set of Feature {j+1}: {score_set}\n')
            if len(score_set) != 0:
                feat_avg_ssim = sum(score_set) / len(score_set)
                self.avg_ssim_list.append(feat_avg_ssim)
                print(f'Average SSIM of Feature {j+1}: {feat_avg_ssim}\n')
                # Store the average score for display in the template
                if f'feature_{j+1}' in self.result_images:
                    self.result_images[f'feature_{j+1}']['score'] = feat_avg_ssim
                else:
                    self.result_images[f'feature_{j+1}'] = {'score': feat_avg_ssim}
            else:
                print('No SSIM scores were found for this feature!')
                self.avg_ssim_list.append(0.0)
                print(f'Average SSIM of Feature {j+1}: 0\n')
                # Store 0.0 as the score for display
                if f'feature_{j+1}' in self.result_images:
                    self.result_images[f'feature_{j+1}']['score'] = 0.0
                else:
                    self.result_images[f'feature_{j+1}'] = {'score': 0.0}
            self.best_extracted_img_list.append([max_score_img, max_score])
            
    def test_feature_8(self):
        # Check Feature 8 - Left bleed lines
        print('\nANALYSIS OF FEATURE 8: LEFT BLEED LINES\n')
        
        # Cropping the region in which left bleed lines are present
        crop = self.test_img[120:240, 12:35]
        
        img = crop.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        
        # Save thresholded image for display
        self.result_images['feature_8'] = {
            'thresholded': thresh
        }
        
        whitePixelValue = 255
        blackPixelValue = 0
        
        width = thresh.shape[1]
        
        result = []
        num_of_cols = 0
        
        print('Number of black regions found in each column: ')
        
        # Iteration over each column in the cropped image
        for j in range(width):
            col = thresh[:, j:j+1]
            count = 0
            
            # Iterating over each row (or pixel) in the current column
            for i in range(len(col)-1):
                pixel1_value = col[i][0]
                pixel2_value = col[i+1][0]
                
                # Handle error pixels
                if pixel1_value != 0 and pixel1_value != 255:
                    pixel1_value = 255
                if pixel2_value != 0 and pixel2_value != 255:
                    pixel2_value = 255
                
                # If current pixel is white and next pixel is black, increment counter
                if pixel1_value == whitePixelValue and pixel2_value == blackPixelValue:
                    count += 1
            
            # If count is valid, add to results
            if count > 0 and count < 10:
                print(count)
                result.append(count)
                num_of_cols += 1
            else:
                print(f'{count} Erroneous -> discarded')
        
        print(f'\nNumber of columns examined: {width}')
        print(f'Number of non-erroneous columns found: {num_of_cols}')
        
        if num_of_cols != 0:
            average_count = sum(result) / num_of_cols
        else:
            average_count = -1
            print('Error occurred - Division by 0')
        
        print(f'\nAverage number of black regions is: {average_count}')
        
        # Storing the thresholded image and average number of bleed lines detected 
        self.left_BL_result = [thresh, average_count]
        
        # Update result images dict with the count
        self.result_images['feature_8']['count'] = average_count
    
    def test_feature_9(self):
        # Check Feature 9 - Right bleed lines
        print('\nANALYSIS OF FEATURE 9: RIGHT BLEED LINES\n')
        
        # Cropping the region in which right bleed lines are present
        crop = self.test_img[120:260, 1135:1155]
        
        img = crop.copy()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        _, thresh = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        
        # Save thresholded image for display
        self.result_images['feature_9'] = {
            'thresholded': thresh
        }
        
        whitePixelValue = 255
        blackPixelValue = 0
        
        width = thresh.shape[1]
        
        result = []
        num_of_cols = 0
        
        print('Number of black regions found in each column: ')
        
        # Iteration over each column in the cropped image
        for j in range(width):
            col = thresh[:, j:j+1]
            count = 0
            
            # Iterating over each row (or pixel) in the current column
            for i in range(len(col)-1):
                pixel1_value = col[i][0]
                pixel2_value = col[i+1][0]
                
                # Handle error pixels
                if pixel1_value != 0 and pixel1_value != 255:
                    pixel1_value = 255
                if pixel2_value != 0 and pixel2_value != 255:
                    pixel2_value = 255
                
                # If current pixel is white and next pixel is black, increment counter
                if pixel1_value == whitePixelValue and pixel2_value == blackPixelValue:
                    count += 1
            
            # If count is valid, add to results
            if count > 0 and count < 10:
                print(count)
                result.append(count)
                num_of_cols += 1
            else:
                print(f'{count} Erroneous -> discarded')
        
        print(f'\nNumber of columns examined: {width}')
        print(f'Number of non-erroneous columns found: {num_of_cols}')
        
        if num_of_cols != 0:
            average_count = sum(result) / num_of_cols
        else:
            average_count = -1
            print('Error occurred - Division by 0')
        
        print(f'\nAverage number of black regions is: {average_count}')
        
        # Storing the thresholded image and average number of bleed lines detected 
        self.right_BL_result = [thresh, average_count]
        
        # Update result images dict with the count
        self.result_images['feature_9']['count'] = average_count
    
    def test_feature_10(self):
        # Feature 10: Currency Number Panel
        print('\nANALYSIS OF FEATURE 10: NUMBER PANEL\n')
        
        # Cropping out the number panel
        crop = self.gray_test_image[410:510, 690:1090]
        crop_bgr = self.test_img[410:510, 690:1090]
        
        # Save the original number panel for display
        self.result_images['feature_10'] = {
            'original': crop_bgr
        }
        
        # Preprocess: threshold, denoise, etc.
        gray = cv2.threshold(crop, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        gray = cv2.dilate(gray, kernel, iterations=1)

        # Use Tesseract on the whole region
        detected_number = pytesseract.image_to_string(
            gray,
            config='--psm 7 --oem 3 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        )
        detected_number = detected_number.strip().replace(' ', '').replace('\n', '')
        # Clean: keep only uppercase letters and digits
        detected_number = re.sub(r'[^A-Z0-9]', '', detected_number.upper())

        print(f"Detected number: '{detected_number}' (length: {len(detected_number)})")
        print([ord(c) for c in detected_number])
        test_passed = len(detected_number) == 9

        if test_passed:
            print(f'Test Passed! - Detected number: {detected_number}')
        else:
            print(f'Test Failed! - Found {len(detected_number)} characters instead of 9')
            print(f'Detected characters: {detected_number}')

        # Store the result
        self.number_panel_result = [crop_bgr, test_passed]
        self.result_images['feature_10']['processed'] = crop_bgr
        self.result_images['feature_10']['detected'] = test_passed
        self.result_images['feature_10']['digit_count'] = len(detected_number)
        self.result_images['feature_10']['detected_number'] = detected_number
    
    def test_result(self):
        # Result analysis
        print('\n\nRESULT ANALYSIS\n')
        
        # Stores the min allowed SSIM score for each feature
        min_ssim_score_list = [0.4, 0.4, 0.5, 0.4, 0.5, 0.45, 0.5]
        
        self.result_list = []
        successful_features_count = 0
        
        # Feature 1 to 7: Results
        for i in range(self.NUM_OF_FEATURES):
            avg_score = self.avg_ssim_list[i]
            img, max_score = self.best_extracted_img_list[i]
            status = False
            min_allowed_score = min_ssim_score_list[i]
            
            print(f"Feature {i+1}: avg_score={avg_score}, max_score={max_score}, min_allowed_score={min_allowed_score}")
            if avg_score >= min_allowed_score or max_score >= 0.79:
                status = True
                successful_features_count += 1
                print(f'Feature {i+1}: Successful')
            else:
                status = False
                print(f'Feature {i+1}: Unsuccessful')
            
            self.result_list.append([img, avg_score, max_score, status])
        
        # Feature 8: Left Bleed lines
        img8, line_count8 = self.left_BL_result[:]
        status8 = line_count8 >= 4.7 and line_count8 <= 5.6
        if status8:
            successful_features_count += 1
            print('Feature 8: Successful - 5 bleed lines found in left part of currency note')
        else:
            print('Feature 8: Unsuccessful!')
        self.result_list.append([img8, line_count8, status8])
        
        # Feature 9: Right Bleed lines
        img9, line_count9 = self.right_BL_result[:]
        status9 = line_count9 >= 4.7 and line_count9 <= 5.6
        if status9:
            successful_features_count += 1
            print('Feature 9: Successful - 5 bleed lines found in right part of currency note')
        else:
            print('Feature 9: Unsuccessful!')
        self.result_list.append([img9, line_count9, status9])
        
        # Feature 10: Currency Number Panel
        img10, status10 = self.number_panel_result[:2]  # Only take the first two elements
        if status10:
            successful_features_count += 1
            print('Feature 10: Successful - 9 digits found in number panel of currency note')
        else:
            print('Feature 10: Unsuccessful!')
        self.result_list.append([img10, status10])
        
        # Final Result
        print('\nResult Summary:')
        print(f'{successful_features_count} out of 10 features are VERIFIED!')
        
        # Determine if the note is authentic (at least 7 out of 10 features)
        is_authentic = successful_features_count >= 7
        
        return {
            'successful_features': successful_features_count,
            'total_features': 10,
            'is_authentic': is_authentic,
            'result_details': self.result_list,
            'result_images': self.result_images
        }
    
    def run_detection(self):
        # Run all tests
        self.test_feature_1_to_7()
        self.test_feature_8()
        self.test_feature_9()
        self.test_feature_10()
        return self.test_result()
    
    def get_image_base64(self, img):
        """Convert an image to base64 string for HTML display"""
        if img is None or isinstance(img, bool):
            return None
            
        # Convert to RGB (from BGR) if needed
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
        # Encode image to PNG format
        _, buffer = cv2.imencode('.png', img)
        
        # Convert to base64 string
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return img_base64 