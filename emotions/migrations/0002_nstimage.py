# Generated by Django 4.0.3 on 2022-03-09 02:41

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('emotions', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='NSTImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('gen_img', models.ImageField(upload_to='nstimages')),
            ],
        ),
    ]
