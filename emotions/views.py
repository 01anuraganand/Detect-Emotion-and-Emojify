import re
from urllib import request
from django.shortcuts import render
from asgiref.sync import async_to_sync
from .models import Image, NSTImage
from .forms import ImageForm, NSTImageForm
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import PIL
import cv2
# Create your views here.
def index(request):
    return render(request, 'emotions/html/index.html')


def upload_photo(request):
    to_delete = Image.objects.all()
    to_delete.delete()
    if request.method == 'POST':
        form = ImageForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
    form = ImageForm()
    img = Image.objects.all()
    return form, img

@async_to_sync
async def classify_emotion(request,img_path):

    frame = cv2.imread(img_path)
    faceCascade = cv2.CascadeClassifier('emotions\MODELS\haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = faceCascade.detectMultiScale(gray, 1.1,4)

    for x, y, w, h in faces:
        roi_gray = gray[y:y+h, x:x+w]
        #roi_color = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (255, 0,0),2)
        faces = faceCascade.detectMultiScale(roi_gray)
        if len(faces) == 0:
            print('------------------------->Face not detected')
        else:
            for(ex, ey, ew, eh) in faces:
                face_roi = roi_gray[ey: ey+eh, ex : ex+ew]

    img = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
    filename = 'te.jpg'
    cv2.imwrite(filename,img)
    
    img = image.load_img(filename, color_mode='rgb',target_size=(48, 48))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    requested_model = request.POST.get('model')

    if requested_model == 'DenseNet201':
        model_path = 'emotions\MODELS\DenseNet201_max_val.h5'
    elif requested_model == 'VGG19':
        model_path = 'emotions\MODELS\VGG19_max_val.h5'
    elif requested_model == 'ResNet50':
        model_path = 'emotions\MODELS\ResNet50_max_val.h5'
    elif requested_model == 'MobileNetV2':
        model_path = 'emotions\MODELS\MobileNetV2_max_val.h5'

    model_emo = tf.keras.models.load_model(model_path)
    classname_mapping = {'0': 'Angry', '1': 'Disgust', '2': 'Fear', '3': 'Happy', '4': 'Sad', '5': 'Surprise', '6' : 'Neutral'}
    pred = model_emo.predict(x)
    np.set_printoptions(suppress = True)
    return max(pred[0])*100, pd.Series(str(pred.argmax())).map(classname_mapping)[0], requested_model

def render_upload_photo_classify_emotion(request):
    form = img = max_pred = pred = model =  None
    form, img = upload_photo(request)
    img_path = [str(x.photo.url) for x in img] 
    #print(img_path)
    if len(img_path) > 0:
        try:
            pred, max_pred, model = classify_emotion(request, img_path =img_path[0][1:] )
            pred = round(pred,2)
        except:
            print('-----------')
            #return render(request,'emotions/html/upload_image_to_emotion.html')

    return render(request, 'emotions/html/upload_image_to_emotion.html', {'form' : form, 'img' : img,'pred': pred, 'max_pred': max_pred , 'model': model})



# Neural Style Transfer
def initialize_model(img_size = 400):
    vgg = tf.keras.applications.VGG19(include_top = False, input_shape = (img_size, img_size, 3), weights = 'emotions/MODELS/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5')
    vgg.trainable = False
    return vgg

def compute_content_cost(content_output, generated_output):
    a_C = content_output[-1]
    a_G = generated_output[-1]
    
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    
    a_C_unrolled = tf.reshape(a_C, shape = [m, n_H*n_W, n_C])
    a_G_unrolled = tf.reshape(a_G, shape = [m, n_H*n_W, n_C])
    
    J_content = (1/(4*n_H*n_W*n_C))*tf.reduce_sum(tf.square(tf.subtract(a_C_unrolled, a_G_unrolled)))
    
    return J_content

def gram_matrix(A):
    #GA gram matix
    GA = tf.matmul(A, tf.transpose(A))
    return GA

def compute_layer_style_cost(a_S, a_G):
    m, n_H, n_W, n_C = a_G.get_shape().as_list()
    # reshape the image from(n_H*n_W, n_C) to (n_C, n_H*n_W)
    
    a_S = tf.transpose(tf.reshape(a_S, shape = [-1, n_C]))
    a_G = tf.transpose(tf.reshape(a_G, shape = [-1, n_C]))
    
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)
    
    J_style_layer = tf.reduce_sum(tf.square(GS - GG))/(4*((n_H * n_W * n_C)**2))
    
    return J_style_layer

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS):
    J_style = 0
    
    a_S = style_image_output[:-1]
    
    a_G = generated_image_output[:-1]
    
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])
        
        J_style += weight[1] * J_style_layer
        
    return J_style

@tf.function()
def total_cost(J_content, J_style, alpha = 10, beta = 40):
    J = alpha * J_content + beta*J_style
    
    return J

def get_image(content_img_path, style_img_path, img_size = 400):
    
    content_image = np.array(PIL.Image.open(content_img_path).resize((img_size, img_size)))
    content_image = tf.constant(np.reshape(content_image, ((1,) + content_image.shape)))
    
    style_image = np.array(PIL.Image.open(style_img_path).resize((img_size, img_size)))
    style_image = tf.constant(np.reshape(style_image, ((1,) + style_image.shape)))
    
    return content_image, style_image

def random_generated_image(nst_img_path):
    content_image, _ = get_image(nst_img_path, 'emotions\download.png', img_size = 400)
    generated_img = tf.Variable(tf.image.convert_image_dtype(content_image, tf.float32))
    noise = tf.random.uniform(tf.shape(generated_img), -0.25, 0.25)
    generated_img = tf.add(generated_img, noise)
    generated_img = tf.clip_by_value(generated_img, clip_value_min = 0.0, clip_value_max = 1.0)
    
    return generated_img

def get_layer_outputs(vgg, layer_names):
    outputs = [vgg.get_layer(layer[0]).output for layer in layer_names]
    
    model = tf.keras.Model([vgg.input], outputs)
    return model

def get_all_input_to_train(content_path, style_path='emotions\download.png'):
    content_layer = [('block5_conv4',1)]
    STYLE_LAYERS = [
        ('block1_conv1', 0.2),
        ('block2_conv1', 0.2),
        ('block3_conv1', 0.2),
        ('block4_conv1', 0.2),
        ('block5_conv1', 0.2)
        ]
    vgg = initialize_model()    
    vgg_model_outputs = get_layer_outputs(vgg, STYLE_LAYERS + content_layer)
    content_image, style_image = get_image(content_path, style_path, img_size = 400)    
    #content_target = vgg_model_outputs(content_image)
    #style_targets = vgg_model_outputs(style_image)
    preprocessed_content = tf.Variable(tf.image.convert_image_dtype(content_image,tf.float32))
    a_C = vgg_model_outputs(preprocessed_content)
    preprocessed_style = tf.Variable(tf.image.convert_image_dtype(style_image, tf.float32))
    a_S  = vgg_model_outputs(preprocessed_style)
    
    return content_layer, STYLE_LAYERS, a_C, a_S, vgg_model_outputs
    

def clip_0_1(image):
    return tf.clip_by_value(image, clip_value_min = 0.0, clip_value_max = 1.0)

def tensor_to_image(tensor):
    tensor = tensor * 255
    tensor = np.array(tensor, dtype = np.uint8)
    if np.ndim(tensor) >3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)



@tf.function()
def train_step(generated_image, alpha = 10, beta = 40):
    optimizer = tf.keras.optimizers.Adam(learning_rate = 0.01)
    nst_img_path = Image.objects.get()
    content_layer, STYLE_LAYERS, a_C, a_S, vgg_model_outputs = get_all_input_to_train(content_path=nst_img_path.photo)
    with tf.GradientTape() as tape:
        a_G = vgg_model_outputs(generated_image)
        
        J_style = compute_style_cost(a_S, a_G, STYLE_LAYERS)
        
        J_content = compute_content_cost(a_C, a_G)
        
        J = total_cost(J_content, J_style, alpha = alpha, beta = beta)
        
    grad = tape.gradient(J, generated_image)
    
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(clip_0_1(generated_image))
    
    return J

def render_upload_photo_for_nst(request):
    form = img = None
    form, img = upload_photo(request)
    
    nst_img_path = [str(x.photo.url) for x in img] 
    
    nst_true = True
    return render(request, 'emotions/html/nst_input.html', {'form' : form, 'img' : img , 'nst': True})

def neural_style_transfer_call(request):
    tf.config.run_functions_eagerly(True)

    nst_img_path = Image.objects.get()
    generated = random_generated_image(nst_img_path.photo)
    generated_image = tf.Variable(generated)
    epochs = 6
    to_delete = NSTImage.objects.all()
    to_delete.delete()

    for i in range(epochs):
        train_step(generated_image)
        if i % 2 == 0:
            print(f"Epoch {i} ")
        if i % 2 == 0:
            image = tensor_to_image(generated_image)
            image.save(f"nstimages/image.jpg")

            save_file = NSTImage()
            path = "nstimages/image.jpg"
            save_file.gen_img.name = path
            save_file.save()

            img = NSTImage.objects.latest('id')
            
    return render(request, 'emotions/html/nst.html', {'img': img, 'orig_img': nst_img_path})