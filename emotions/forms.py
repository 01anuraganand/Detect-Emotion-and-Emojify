from django import forms
from django.forms import fields
from matplotlib.pyplot import cla
from .models import Image, NSTImage

class ImageForm(forms.ModelForm):
    class Meta:
        model = Image
        fields = ['photo']

class NSTImageForm(forms.ModelForm):
    class Meta:
        model = NSTImage
        fields = ['gen_img']