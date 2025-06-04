from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('upload/', views.upload_image, name='upload_image'),
    path('process/<int:image_id>/', views.process_image, name='process_image'),
    path('check-progress/<int:image_id>/', views.check_progress, name='check_progress'),
    path('result/<int:image_id>/', views.result, name='result'),
    path('about/', views.about, name='about'),
    path('s3-test/', views.s3_test, name='s3_test'),
] 