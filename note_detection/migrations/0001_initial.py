# Generated by Django 5.1.7 on 2025-06-04 11:13

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='CurrencyImage',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(upload_to='currency_images/')),
                ('denomination', models.CharField(choices=[('100', '100 Rupee Note'), ('200', '200 Rupee Note'), ('500', '500 Rupee Note')], default='100', max_length=3)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
                ('is_authentic', models.BooleanField(blank=True, null=True)),
                ('features_detected', models.IntegerField(blank=True, null=True)),
                ('detection_details', models.TextField(blank=True, null=True)),
                ('processing_complete', models.BooleanField(default=False)),
                ('error_message', models.TextField(blank=True, null=True)),
            ],
        ),
    ]
