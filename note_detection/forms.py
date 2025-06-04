from django import forms
from .models import CurrencyImage

class CurrencyImageForm(forms.ModelForm):
    class Meta:
        model = CurrencyImage
        fields = ['image', 'denomination']
        widgets = {
            'image': forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'}),
            'denomination': forms.Select(attrs={'class': 'form-select'})
        }

class ImageUploadForm(forms.Form):
    denomination = forms.ChoiceField(
        choices=CurrencyImage.DENOMINATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'})
    )
    image = forms.ImageField(
        label='Select Currency Image',
        widget=forms.FileInput(attrs={'class': 'form-control', 'accept': 'image/*'})
    ) 