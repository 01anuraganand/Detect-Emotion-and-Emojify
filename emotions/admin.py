from django.contrib import admin
from . models import Image, NSTImage
# Register your models here.

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'photo']

@admin.register(NSTImage)
class NSTImageAdmin(admin.ModelAdmin):
    list_display = ['gen_img']