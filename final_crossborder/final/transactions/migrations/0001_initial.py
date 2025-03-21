# Generated by Django 4.2 on 2025-03-17 04:15

from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='UserProfile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('phone_number', models.CharField(max_length=15)),
                ('address', models.TextField()),
                ('preferred_currency', models.CharField(choices=[('USD', 'US Dollar'), ('EUR', 'Euro'), ('GBP', 'British Pound'), ('JPY', 'Japanese Yen'), ('INR', 'Indian Rupee')], default='USD', max_length=3)),
                ('user', models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='Transaction',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('transaction_type', models.CharField(choices=[('send', 'Send Money'), ('receive', 'Receive Money')], max_length=10)),
                ('amount', models.DecimalField(decimal_places=2, max_digits=10)),
                ('source_currency', models.CharField(choices=[('USD', 'US Dollar'), ('EUR', 'Euro'), ('GBP', 'British Pound'), ('JPY', 'Japanese Yen'), ('INR', 'Indian Rupee')], max_length=3)),
                ('target_currency', models.CharField(choices=[('USD', 'US Dollar'), ('EUR', 'Euro'), ('GBP', 'British Pound'), ('JPY', 'Japanese Yen'), ('INR', 'Indian Rupee')], max_length=3)),
                ('recipient_name', models.CharField(max_length=100)),
                ('recipient_account', models.CharField(max_length=50)),
                ('recipient_bank', models.CharField(max_length=100)),
                ('status', models.CharField(choices=[('pending', 'Pending'), ('completed', 'Completed'), ('failed', 'Failed')], default='pending', max_length=10)),
                ('created_at', models.DateTimeField(auto_now_add=True)),
                ('updated_at', models.DateTimeField(auto_now=True)),
                ('notes', models.TextField(blank=True, null=True)),
                ('conversion_rate', models.DecimalField(decimal_places=4, max_digits=10)),
                ('converted_amount', models.DecimalField(decimal_places=2, max_digits=10)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, to=settings.AUTH_USER_MODEL)),
            ],
            options={
                'ordering': ['-created_at'],
            },
        ),
    ]
