import uuid

from django import forms
from brain_tumor.models import User
from django.core.exceptions import ValidationError
import os 


def validate_file_extension(value):
    ext = os.path.splitext(value.name)[1]  # [0] returns path+filename
    valid_extensions = ['.jpg', '.png']
    if not ext.lower() in valid_extensions:
        raise ValidationError('Unsupported file extension.')
    

class RegistrationForm(forms.ModelForm):
    username = forms.CharField(
        min_length=4, max_length=15,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'User Name'})
    )
    first_name = forms.CharField(
        min_length=3, max_length=15,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'First Name'})
    )
    last_name = forms.CharField(
        min_length=3, max_length=15,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Last Name'})
    )
    email = forms.EmailField(
        min_length=6, max_length=40,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Email Address'})
    )
    password = forms.CharField(
        min_length=6, max_length=20,
        widget=forms.PasswordInput(render_value=False, attrs={'placeholder': 'Password', 'class': 'form-control'})
    )
    confirm_password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Repeat Password', 'class': 'form-control'})
    )

    class Meta:
        model = User
        fields = ('username', 'first_name', 'last_name', 'email', 'password')
    
    def clean_email(self):
        email = self.cleaned_data['email']
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError("Email already exists")
        return email

    def clean_password(self):
        password = self.cleaned_data['password']
        confirm_password = self.data['confirm_password']
        if password != confirm_password:
            raise forms.ValidationError("password does not matched")
        return password


class LoginForm(forms.Form):
    email = forms.EmailField(
        min_length=6, max_length=40, required=False,
        widget=forms.TextInput(attrs={'class': 'form-control', 'placeholder': 'Email Address'})
    )
    password = forms.CharField(
        min_length=6, max_length=20,
        widget=forms.PasswordInput(attrs={'placeholder': 'Password', 'class': 'form-control'})
    )


class UploadFileForm(forms.Form):
    post_file = forms.FileField(
        label="Choose Image File",
        widget=forms.FileInput(attrs={'class': 'form-control bg-light border-0'}),
        required=False, validators=[validate_file_extension]
    )

    def clean_post_file(self):
        uploaded_file = self.cleaned_data['post_file']
        file_extension = os.path.splitext(uploaded_file.name)[-1]
        unique_filename = f"{uuid.uuid4().hex}{file_extension}"
        uploaded_file.name = unique_filename
        return uploaded_file




    

