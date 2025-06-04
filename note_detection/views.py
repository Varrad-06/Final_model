from django.shortcuts import render, redirect
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from django.urls import reverse
from django.http import JsonResponse
import os
import time
import threading
import uuid
import cv2
import numpy as np
from django.core.files import File

from .forms import CurrencyImageForm, ImageUploadForm
from .models import CurrencyImage
from .detection import CurrencyDetector100
from .detection_200 import CurrencyDetector200
from .detection_500 import CurrencyDetector500

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
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            try:
                # Get the selected denomination
                denomination = form.cleaned_data['denomination']
                image = form.cleaned_data['image']

                # Save the uploaded image to MEDIA_ROOT/uploads
                upload_dir = os.path.join(settings.MEDIA_ROOT, 'uploads')
                os.makedirs(upload_dir, exist_ok=True)
                unique_filename = f"{uuid.uuid4()}_{image.name}"
                full_path = os.path.join(upload_dir, unique_filename)

                with open(full_path, 'wb+') as destination:
                    for chunk in image.chunks():
                        destination.write(chunk)

                # Save the file to the model using Django's File object
                with open(full_path, 'rb') as f:
                    django_file = File(f)
                    currency_image = CurrencyImage(
                        image=f'uploads/{unique_filename}',
                        denomination=denomination
                    )
                    currency_image.save()

                # Redirect to processing page
                return redirect('process_image', image_id=currency_image.id)

            except Exception as e:
                # Clean up the uploaded file if it exists
                if os.path.exists(full_path):
                    os.remove(full_path)
                return JsonResponse({'error': f'Error processing image: {str(e)}'})
    else:
        form = ImageUploadForm()
    
    return render(request, 'note_detection/home.html', {'form': form})

def upload_image(request):
    """Handle image upload and start detection process"""
    if request.method == 'POST':
        form = CurrencyImageForm(request.POST, request.FILES)
        if form.is_valid():
            # Save the uploaded image
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
                # Get the full path to the uploaded image
                image_path = currency_image.image.path
                denomination = currency_image.denomination
                
                # Auto-crop and perspective correct the note before detection
                img = cv2.imread(image_path)
                cropped = auto_crop_note(img)
                # Always resize to model input size
                cropped = cv2.resize(cropped, (1167, 519))
                cv2.imwrite(image_path, cropped)

                # Get the appropriate detector class
                detector_class = get_detector_class(denomination)
                if not detector_class:
                    currency_image.error_message = 'Invalid denomination selected'
                    currency_image.processing_complete = True
                    currency_image.save()
                    return
                
                # Create detector and run detection
                detector = detector_class(image_path)
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
        detector = detector_class(currency_image.image.path)
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
