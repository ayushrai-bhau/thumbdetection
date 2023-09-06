from django.urls import path,include
from .views import *


urlpatterns = [
    path('image-upload/',Thumb_face_detection.as_view()),

]