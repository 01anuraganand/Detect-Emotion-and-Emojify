from django.urls import path
from . import views
urlpatterns = [
    path('', views.index, name = 'index'),
    path('detected_emotion/', views.render_upload_photo_classify_emotion, name = 'detected_emotion'),
    path('upload_image_for_nst/', views.render_upload_photo_for_nst, name = 'upload_image_for_nst'),
    path('styled_image/', views.neural_style_transfer_call, name= 'styled_image'),

]