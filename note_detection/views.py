from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage, default_storage
from django.urls import reverse
from django.http import JsonResponse, HttpResponse
import os
import time
import threading
import uuid
import cv2
import numpy as np
from django.core.files import File
from django.core.files.base import ContentFile
import dj_database_url

from .forms import CurrencyImageForm, ImageUploadForm
from .models import CurrencyImage
from .detection_100 import CurrencyDetector100
from .detection_200 import CurrencyDetector200
from .detection_500 import CurrencyDetector500

DATABASES = {
    'default': dj_database_url.config(default=os.environ.get('postgresql://dataset_fgzh_user:pAarrb9wcK6XuG1nSnOxjUvdOZwuVEgE@dpg-d10khlmmcj7s73bq20mg-a/dataset_fgzh'))
}

def get_detector_class(denomination):
    """Helper function to get the appropriate detector class"""
    detectors = {
        '100': CurrencyDetector100,
        '200': CurrencyDetector200,
        '500': CurrencyDetector500
    }
    return detectors.get(denomination)

def home(request):
    """Home page view with the currency note upload form"""
    if request.method == 'POST':
        form = CurrencyImageForm(request.POST, request.FILES)
        if form.is_valid():
            currency_image = form.save()
            return redirect('process_image', image_id=currency_image.id)
        else:
            # Render the form with errors
            return render(request, 'note_detection/home.html', {'form': form})
    else:
        form = ImageUploadForm()
    return render(request, 'note_detection/home.html', {'form': form})

def upload_image(request):
    """Handle image upload and start detection process"""
    if request.method == 'POST':
        form = CurrencyImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image using the model's ImageField
            currency_image = form.save()
            # Redirect to a processing page, passing the image ID
            return redirect('process_image', image_id=currency_image.id)
    # If form is invalid, redirect back to home
    return redirect('home')

def process_image(request, image_id):
    """Display a page showing that the image is being processed"""
    try:
        # Get the uploaded image
        currency_image = CurrencyImage.objects.get(id=image_id)
        
        # Start processing in a separate thread
        def process_in_background():
            try:
                # Read image from S3
                image_file = currency_image.image.open('rb')
                file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                if img is None:
                    currency_image.error_message = "Uploaded image could not be read. Please upload a valid image file."
                    currency_image.processing_complete = True
                    currency_image.save()
                    return
                
                # Ensure image is 3-channel BGR
                if len(img.shape) == 2:
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                elif img.shape[2] == 4:
                    img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
                
                denomination = currency_image.denomination
                
                # Auto-crop and perspective correct the note before detection
                cropped = auto_crop_note(img)
                # Always resize to model input size
                cropped = cv2.resize(cropped, (1167, 519))

                # Save processed image back to S3
                _, buffer = cv2.imencode('.jpg', cropped)
                currency_image.image.save('processed.jpg', ContentFile(buffer.tobytes()))
                currency_image.refresh_from_db()  # Ensure the new image is loaded

                # Re-open the image for reading
                image_file = currency_image.image.open('rb')
                image_file.seek(0)  # Ensure pointer is at start

                # Get the appropriate detector class
                detector_class = get_detector_class(denomination)
                if not detector_class:
                    currency_image.error_message = 'Invalid denomination selected'
                    currency_image.processing_complete = True
                    currency_image.save()
                    return
                
                # Create detector and run detection
                detector = detector_class(image_file)
                result = detector.run_detection()
                
                # Update the currency image record with results
                currency_image.is_authentic = result['is_authentic']
                currency_image.features_detected = result['successful_features']
                currency_image.detection_details = f"Features detected: {result['successful_features']}/{result['total_features']}"
                currency_image.processing_complete = True
                currency_image.save()
                
            except Exception as e:
                # Log the error and update the currency image
                currency_image.error_message = f"Error during processing: {str(e)}"
                currency_image.processing_complete = True
                currency_image.save()
        
        # Start processing in background
        threading.Thread(target=process_in_background).start()
        
        return render(request, 'note_detection/processing.html', {
            'image_id': image_id
        })
    
    except CurrencyImage.DoesNotExist:
        return redirect('home')

def check_progress(request, image_id):
    """Check if processing is complete for AJAX calls"""
    try:
        currency_image = CurrencyImage.objects.get(id=image_id)
        
        if currency_image.processing_complete:
            if currency_image.error_message:
                return JsonResponse({
                    'complete': True,
                    'error': currency_image.error_message
                })
            return JsonResponse({
                'complete': True,
                'result_url': reverse('result', args=[image_id])
            })
        else:
            return JsonResponse({
                'complete': False
            })
    
    except CurrencyImage.DoesNotExist:
        return JsonResponse({
            'complete': False,
            'error': 'Image not found'
        })

def result(request, image_id):
    """Display the currency detection results"""
    try:
        # Get the currency image
        currency_image = CurrencyImage.objects.get(id=image_id)
        
        # If processing is not complete, redirect to processing page
        if currency_image.is_authentic is None:
            return redirect('process_image', image_id=image_id)
        
        # Get the appropriate detector class
        denomination = currency_image.denomination or '100'
        detector_class = get_detector_class(denomination)
        if not detector_class:
            return redirect('home')
        
        # Run the detector on the image to get visualization data
        detector = detector_class(currency_image.image)
        result = detector.run_detection()
        
        # Get base64 encoded images for display
        result_images = {}
        for key, data in detector.result_images.items():
            result_images[key] = {}
            for img_key, img in data.items():
                if img_key in ['template', 'detected', 'thresholded', 'original', 'processed']:
                    result_images[key][img_key] = detector.get_image_base64(img)
                else:
                    result_images[key][img_key] = img
        
        context = {
            'currency_image': currency_image,
            'is_authentic': currency_image.is_authentic,
            'features_detected': currency_image.features_detected,
            'total_features': 10,
            'result_images': result_images
        }
        
        return render(request, 'note_detection/result.html', context)
    
    except CurrencyImage.DoesNotExist:
        return redirect('home')

def about(request):
    """Display information about the project"""
    return render(request, 'note_detection/about.html')

# Utility functions for perspective correction

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(image, pts):
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def auto_crop_note(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edged = cv2.Canny(blur, 50, 200)
    contours, _ = cv2.findContours(edged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            warped = four_point_transform(image, pts)
            return warped
    return image  # fallback: return original if no rectangle found

def s3_test(request):
    try:
        with default_storage.open('test.txt', 'w') as f:
            f.write('hello')
        return HttpResponse('S3 write succeeded')
    except Exception as e:
        return HttpResponse(f'S3 write failed: {e}')

def run_migrations(request):
    from django.core.management import call_command
    from django.http import HttpResponse
    try:
        call_command('makemigrations', 'note_detection')
        call_command('migrate')
        return HttpResponse('Migrations completed successfully.')
    except Exception as e:
        return HttpResponse(f'Migration failed: {e}')
