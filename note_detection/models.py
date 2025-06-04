from django.db import models
import os

# Create your models here.
class CurrencyImage(models.Model):
    DENOMINATION_CHOICES = [
        ('100', '100 Rupee Note'),
        ('200', '200 Rupee Note'),
        ('500', '500 Rupee Note'),
    ]
    
    image = models.ImageField(upload_to='currency_images/')
    denomination = models.CharField(max_length=3, choices=DENOMINATION_CHOICES, default='100')
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_authentic = models.BooleanField(null=True, blank=True)
    features_detected = models.IntegerField(null=True, blank=True)
    detection_details = models.TextField(blank=True, null=True)
    processing_complete = models.BooleanField(default=False)
    error_message = models.TextField(blank=True, null=True)
    
    def __str__(self):
        return f"{self.denomination} Rupee Note - {self.uploaded_at.strftime('%Y-%m-%d %H:%M:%S')}"
        
    def filename(self):
        return os.path.basename(self.image.name)
