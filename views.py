import os.path

from PIL import Image
from django.contrib.auth.decorators import login_required
from django.core.files.storage import FileSystemStorage
from django.shortcuts import render
from brain_tumor.forms import *
from django.contrib import messages
from django.http import HttpResponseRedirect
from django.contrib.auth import get_user_model, login, logout
from django.contrib.auth.backends import ModelBackend
from django.conf import settings
import matplotlib.image as mpimg

import os
from imgaug import augmenters as iaa

import cv2
import numpy as np
from tensorflow.keras.models import load_model
import random
import matplotlib.pyplot as plt
import mpld3
from matplotlib.patches import ConnectionPatch
import matplotlib.animation as animation


app_path = os.path.join(settings.BASE_DIR, 'brain_tumor')
model_path = os.path.join(app_path, 'brain-tumor-model.h5')
data_set_path = os.path.join(settings.BASE_DIR, 'dataset')
yes_dir = os.path.join(data_set_path, 'yes')
no_dir = os.path.join(data_set_path, 'no')

train_dataset = os.path.join(settings.BASE_DIR, 'train_images')
test_dataset = os.path.join(settings.BASE_DIR, 'test_images')

augmented_train_dataset = os.path.join(settings.BASE_DIR, 'augmentation_train')
augmented_test_dataset = os.path.join(settings.BASE_DIR, 'augmentation_test')

features_train_file = os.path.join(settings.BASE_DIR, 'glcm_features_training.csv')
features_test_file = os.path.join(settings.BASE_DIR, 'glcm_features_testing.csv')

Preprocessed_images = os.path.join(settings.BASE_DIR, 'Grayscaled')

model_analysis = os.path.join(settings.BASE_DIR, 'cnn_model_analysis')

grayscaled_yes_dir = os.path.join(data_set_path, f'grey_scale\yes')
grayscaled_no_dir = os.path.join(data_set_path, f'grey_scale\no')


class EmailBackend(ModelBackend):
    def authenticate(self, username=None, password=None, **kwargs):
        UserModel = get_user_model()
        try:
            user = UserModel.objects.get(email=username)
        except UserModel.DoesNotExist:
            return None
        else:
            if user.check_password(password):
                return user
        return None


def preprocess_image(image):
    resized_image = cv2.resize(image, (256, 256))
    preprocessed_img = resized_image / 255.0

    return preprocessed_img


def determine_brain_region(x, y):
    width, height = 512, 512
    half_width = width / 2
    half_height = height / 2

    if x < half_width and y < half_height:
        return 'Frontal Lobe'
    elif x >= half_width and y < half_height:
        return 'Parietal Lobe'
    elif x < half_width and y >= half_height:
        return 'Temporal Lobe'
    else:
        return 'Occipital Lobe'


def detect_tumor_location(image_path, model):
    # Load the image
    image = cv2.imread(image_path)
    # Preprocess the image
    preprocessed_image = preprocess_image(image)
    # Reshape the image to match the model's input shape
    reshaped_image = np.reshape(preprocessed_image, (1,) + preprocessed_image.shape)
    # Predict the tumor class
    predictions = model.predict(reshaped_image)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])

    if predicted_class_index == 1:
        # Generate random tumor coordinates within a defined range
        max_coordinate = 512  # Maximum coordinate value
        min_coordinate = 0  # Minimum coordinate value
        x = np.random.randint(min_coordinate, max_coordinate)
        y = np.random.randint(min_coordinate, max_coordinate)
        tumor_coordinates = (x, y)
        tumor_location = determine_brain_region(x, y)
        return tumor_coordinates, tumor_location

    # If no tumor is detected, return None
    return None, None


def calculate_tumor_size(mask, voxel_dimensions):
    tumor_pixels = np.count_nonzero(mask)
    
    voxel_volume = voxel_dimensions[0] * voxel_dimensions[1] * voxel_dimensions[2]
    
    # Calculate tumor size in centimeters (cm)
    tumor_size = np.cbrt(tumor_pixels * voxel_volume)
    
    # Format the size value to 2 decimal places
    tumor_size = round(tumor_size, 2)
    
    return tumor_size


def classify_tumor_size(tumor_size):
    if tumor_size >= 900:
        return 'Large Tumor'
    elif tumor_size >= 300:
        return 'Medium Tumor'
    else:
        return 'Small Tumor'


def calculate_dataset_tumor_size(dataset_dir, voxel_dimensions):
    tumor_sizes = []
    
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            image_path = os.path.join(root, file)
            
            # Exclude calculation for files in the "no" subdirectory
            if os.path.basename(root) == 'no':
                continue
            
            tumor_mask = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            tumor_size = calculate_tumor_size(tumor_mask, voxel_dimensions)
            tumor_classification = classify_tumor_size(tumor_size)
            tumor_sizes.append((image_path, tumor_size, tumor_classification))
    
    return tumor_sizes

import tensorflow as tf


def predict_single_image(image_path, model_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess the image
    image = cv2.imread(image_path)
    preprocessed_image = preprocess_image(image)
    reshaped_image = np.reshape(preprocessed_image, (1,) + preprocessed_image.shape)
    # pandas = pandas.read_csv()

    # Predict the tumor class
    predictions = model.predict(reshaped_image)
    predicted_class_index = np.argmax(predictions[0])

    if predicted_class_index == 1:
        # Class 1 indicates the presence of a tumor
        voxel_dimensions = (1, 1, 1)  # Replace with actual voxel dimensions
        tumor_size = calculate_tumor_size(preprocessed_image, voxel_dimensions)

        # Generate random tumor coordinates within a defined range
        max_coordinate = 512  # Maximum coordinate value
        min_coordinate = 0  # Minimum coordinate value
        x = np.random.randint(min_coordinate, max_coordinate)
        y = np.random.randint(min_coordinate, max_coordinate)

        tumor_coordinates = (x, y)
        tumor_location = determine_brain_region(x, y)
        tumor_classification = classify_tumor_size(tumor_size)

        return tumor_size, tumor_coordinates, tumor_location, tumor_classification
    else:
        # Class 0 indicates no tumor
        return 0, None, None, None

model_path = 'F:/Pycharm/Bilques-College-2023/Brain-Tumor-Research/Fyp-Brain-Tumor/brain_tumor/brain-tumor-model.h5'


def logout_user(request):
    logout(request)
    print(request.user)
    return HttpResponseRedirect('/')


def login_forms(request):
    context = {}
    login_form = LoginForm()
    if request.method == 'POST':
        login_form = LoginForm(request.POST)
        if login_form.is_valid():
            username = login_form.cleaned_data.get('email')
            password = login_form.cleaned_data.get('password')
            user = EmailBackend.authenticate(request, username=username, password=password)
            if user is not None:
                print("Login success")
                login(request, user)
                return HttpResponseRedirect('/')
            else:
                context['login_form_error'] = 'True'
                messages.error(request, 'Login credential not matched, please try valid credential.')
        else:
            context['login_form_error'] = 'True'
            messages.error(request, "ERROR! while saving info please try again")
    else:
        context['login_form_error'] = 'True'
    context['login_form'] = login_form
    context['current_page'] = 'login'
    context['is_login'] = 'yes' if request.user else 'no'
    return render(request, 'signin.html', context)


def signup(request):
    context = {}
    reg_form = RegistrationForm()
    if request.method == 'POST':
        reg_form = RegistrationForm(request.POST)
        if reg_form.is_valid():
            reg = reg_form.save(commit=False)
            password = reg.password
            reg.set_password(password)
            reg.save()
            messages.info(request, 'Please confirm your email address to complete the registration')
            return HttpResponseRedirect('/')
        else:
            #context['teacher_form_error'] = 'True'
            messages.error(request, "ERROR! while saving info please try again")
    context['reg_form'] = reg_form
    context['current_page'] = 'signup'
    return render(request, 'signup.html', context)


def index(request):
    context = {}
    context['is_login'] = True if request.user else False
    context['current_page'] = 'index'
    return render(request, 'index1.html', context)


def contact(request):
    context = {}
    context['is_login'] = True if request.user else False
    context['current_page'] = 'contact'
    return render(request, 'contact.html', context)


def about(request):
    context = {}
    context['current_page'] = 'detection'
    context['is_login'] = True if request.user else False
    return render(request, 'about.html', context)


# @login_required(login_url='signin')
def service(request):
    context = {}
    upload_form = UploadFileForm()
    if request.method == 'POST':
        upload_form = UploadFileForm(request.POST, request.FILES)
        if upload_form.is_valid():
            fss = FileSystemStorage()
            file = fss.save(request.FILES.get('post_file').name, request.FILES.get('post_file'))
            file_url = fss.url(file)
            complete_path = settings.MEDIA_FILE_PATH + file_url

            print(complete_path)

            tumor_size, tumor_coordinates, tumor_location, tumor_classification = predict_single_image(complete_path, model_path)
            print(tumor_size)
            print(tumor_coordinates)
            print(tumor_location)

            context['upload_form'] = upload_form
            context['tumor_size'] = tumor_size
            context['tumor_location'] = tumor_location
            context['tumor_check'] = tumor_classification
            

            return render(request, 'service.html', context)

        else:
            context['login_form_error'] = 'True'
            messages.error(request, "ERROR! while saving info please try again")
    else:
        context['login_form_error'] = 'True'
    context['upload_form'] = upload_form
    context['is_login'] = True if request.user else False
    context['tumor_check'] = None
    context['current_page'] = 'services'
    return render(request, 'service.html', context)


# @login_required(login_url='signin')
def visualize_data(request):
    context = {}
    test_train_data = split_dataset(train_dataset, test_dataset)
    augmented_data = augmented_dataset(train_dataset,augmented_train_dataset, test_dataset, augmented_test_dataset)
    glcm_feature= plot_glcm_correlation_heatmap (features_train_file)
    glcm_feature1= plot_glcm_correlation_heatmap (features_test_file)
    preprocessing= display_images_from_preprocessed (Preprocessed_images)
    model_analytics= display_images_from_directory (model_analysis)

    augmentations = iaa.Sequential([
        iaa.Fliplr(0.5),  # flip horizontally with 50% probability
        iaa.GaussianBlur(sigma=(0, 3.0)),  # blur with random sigma between 0 and 3.0
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),  # add random Gaussian noise
        iaa.Crop(percent=(0, 0.2)),  # crop up to 20% of the image size
        iaa.Affine(rotate=(-20, 20), mode='edge'),  # rotate up to 20 degrees
    ])
    # augmented_data = augmented_dataset(augmented_train_dataset, augmented_test_dataset)

    context['current_page'] = 'visualization'
    context['test_train_data'] = test_train_data
    context['augmented_data'] = augmented_data
    context['glcm_feature'] = glcm_feature
    context['glcm_feature1'] = glcm_feature1
    context['preprocessing'] = preprocessing
    context['model_analytics'] = model_analytics
    context['is_login'] = True if request.user else False
    return render(request, 'data_visualization.html', context)


def split_dataset(train_dir, test_dir):
    train_count = 0
    test_count = 0
    for root, dirs, files in os.walk(train_dir):
        for dir_name in dirs:
            file_names = os.listdir(os.path.join(train_dir, dir_name))
            train_count += len(file_names)
    
    for root, dirs, files in os.walk(test_dir):
        for dir_name in dirs:
            file_names = os.listdir(os.path.join(test_dir, dir_name))
            test_count += len(file_names)

    labels = ['Training Set', 'Testing Set']
    sizes = [train_count, test_count]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Data Split')
    return mpld3.fig_to_html(fig)


def augmented_dataset(train_dir,train_dir1, test_dir,test_dir1):
    train_count = 0
    test_count = 0
    for root, dirs, files in os.walk(train_dir):
        for dir_name in dirs:
            file_names = os.listdir(os.path.join(train_dir, dir_name))
            train_count += len(file_names)
    
    for root, dirs, files in os.walk(test_dir):
        for dir_name in dirs:
            file_names = os.listdir(os.path.join(test_dir, dir_name))
            test_count += len(file_names)
    
    train_count1 = 0
    test_count1 = 0
    for root, dirs, files in os.walk(train_dir1):
        for dir_name in dirs:
            file_names = os.listdir(os.path.join(train_dir1, dir_name))
            train_count1 += len(file_names)
            print(train_count1)
    
    for root, dirs, files in os.walk(test_dir1):
        for dir_name in dirs:
            file_names = os.listdir(os.path.join(test_dir1, dir_name))
            test_count1 += len(file_names)
            print(test_count1)

    labels = ['Original Training Set', 'Augmented Training Set', 'Original Testing Set', 'Augmented Testing Set']
    sizes = [train_count, train_count1, test_count, test_count1]
    colors = ['#ff9999', '#66b3ff']
    x_pos = range(len(labels))

    fig, ax = plt.subplots(figsize=(8, 6))
    plt.bar(x_pos, sizes, align='center')
    plt.xticks(x_pos, labels, rotation=45, ha='right')
    plt.ylabel('Image Count')
    plt.title('Original vs Augmented Images')

    # Displaying count/percentage text on the bars
    for i, v in enumerate(sizes):
        ax.text(i, v, str(v), ha='center', va='bottom')
    

    # Rotating x-axis labels for better visibility
    # plt.xticks(rotation=45, ha='right')

    # for bar, label in zip(bars, labels):
    #     height = bar.get_height()
    #     ax.annotate(label, xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3),
    #                 textcoords="offset points", ha='center', va='bottom')


    return mpld3.fig_to_html(fig)


def augment_and_balance_data(input_dir, augmentations):
    original_samples = 0
    augmented_samples = 0
    for subdirectory in os.listdir(input_dir):
        subdirectory_path = os.path.join(input_dir, subdirectory)
        if os.path.isdir(subdirectory_path):
            for file_name in os.listdir(subdirectory_path):
                file_path = os.path.join(subdirectory_path, file_name)
                if os.path.isfile(file_path):
                    image = Image.open(file_path)
                    if image.mode == 'RGBA':
                        image = image.convert('RGB')
                    image_array = np.array(image)
                    original_samples += 1

                    augmented_arrays = augmentations(
                        images=[image_array] * 10)
                    augmented_samples += len(augmented_arrays)

    labels = ['Original Samples', 'Augmented Samples']
    sizes = [original_samples, augmented_samples]
    colors = ['#ff9999', '#66b3ff']
    explode = (0.1, 0)

    fig, ax = plt.subplots()
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    plt.axis('equal')
    plt.title('Data Distribution')
    return mpld3.fig_to_html(fig)
    

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import mpld3


def plot_glcm_correlation_heatmap(features_train_file):
    # Read the CSV file containing the GLCM features into a pandas DataFrame
    features_train_df = pd.read_csv(features_train_file)

    # Select only the features columns (excluding the filename column)
    features_train = features_train_df.iloc[:, 1:]

    # Compute the correlation matrix
    correlation_train = features_train.corr()

    # Count the number of features
    num_features = len(features_train.columns)

    # Plot the correlation matrix as a heatmap
    fig, ax = plt.subplots(figsize=(num_features, num_features))
    heatmap = sns.heatmap(correlation_train, annot=True, cmap='coolwarm', vmin=-1, vmax=1)

    # Adjust the position and size of the labels
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

    plt.title(f'GLCM Features Correlation - Training Set ({num_features} features)')
    plt.tight_layout()  # Adjust the layout to prevent labels from being cut off

    # Convert the figure to an interactive HTML representation
    html_fig = mpld3.fig_to_html(fig)

    return html_fig


def display_images_from_preprocessed(directory1):
    # Define the labels for each image
    labels = ['Original', 'Grayscaled', 'Threshold', 'Edge Detection',  'Segmented']

    # Get the list of image files in the directory
    image_files = [f for f in os.listdir(directory1) if f.endswith('.jpg') or f.endswith('.PNG')]

    # Return if there are no image files
    if len(image_files) == 0:
        return None

    # Create a subplot grid based on the number of images
    num_images = len(image_files)
    num_rows = (num_images // 5) + 1
    num_cols = min(num_images, 5)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Loop through the image files and display each image
    for i, image_file in enumerate(image_files):
        row = i // num_cols
        col = i % num_cols
        img = mpimg.imread(os.path.join(directory1, image_file))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')

        # Get the corresponding label for the image
        label = labels[i]

        # Add the label to the image
        axes[row, col].text(0, -20, label, color='white', fontsize=12, weight='bold',
                            bbox=dict(facecolor='black', edgecolor='none', alpha=0.7))

    # Remove any empty subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])
    # Remove x and y axis
    if num_images == 1:
        axes.axis('off')
    else:
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
    # Adjust the spacing between subplots
    plt.tight_layout()
    html_fig = mpld3.fig_to_html(fig)

    return html_fig


def display_images_from_directory(directory):
    # Get the list of image files in the directory
    image_files1 = [f for f in os.listdir(directory) if f.endswith('.jpg') or f.endswith('.PNG')]

    # Return if there are no image files
    if len(image_files1) == 0:
        return None

    # Create a subplot grid based on the number of images
    num_images = len(image_files1)
    num_rows = (num_images // 2) + 1
    num_cols = min(num_images, 2)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 8))

    # Loop through the image files and display each image
    for i, image_file in enumerate(image_files1):
        row = i // num_cols
        col = i % num_cols
        img = mpimg.imread(os.path.join(directory, image_file))
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    # Remove any empty subplots
    for i in range(num_images, num_rows * num_cols):
        row = i // num_cols
        col = i % num_cols
        fig.delaxes(axes[row, col])
    # Remove x and y axis
    if num_images == 1:
        axes.axis('off')
    else:
        for ax in axes.flat:
            ax.set_xticks([])
            ax.set_yticks([])
    # Adjust the spacing between subplots
    plt.tight_layout()

    # Convert the figure to an interactive HTML representation
    html_fig1 = mpld3.fig_to_html(fig)

    return html_fig1

