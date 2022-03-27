from django.contrib import admin
from . models import Image, NSTImage, EmojiImage
# Register your models here.

@admin.register(Image)
class ImageAdmin(admin.ModelAdmin):
    list_display = ['id', 'photo']

@admin.register(NSTImage)
class NSTImageAdmin(admin.ModelAdmin):
    list_display = ['gen_img']

@admin.register(EmojiImage)
class EmojiImageAdmin(admin.ModelAdmin):
    list_display = ['emoji_img']