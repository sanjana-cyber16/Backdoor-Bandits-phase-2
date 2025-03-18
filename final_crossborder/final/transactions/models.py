from django.db import models
from django.contrib.auth.models import User
from django.core.validators import MinValueValidator, MaxValueValidator
from decimal import Decimal
from django.core.exceptions import ValidationError
from django.utils import timezone
from datetime import timedelta

class Transaction(models.Model):
    TRANSACTION_TYPES = (
        ('send', 'Send Money'),
        ('receive', 'Receive Money'),
    )
    
    CURRENCY_CHOICES = (
        ('USD', 'US Dollar'),
        ('EUR', 'Euro'),
        ('GBP', 'British Pound'),
        ('JPY', 'Japanese Yen'),
        ('INR', 'Indian Rupee'),
    )
    
    STATUS_CHOICES = (
        ('pending', 'Pending'),
        ('completed', 'Completed'),
        ('failed', 'Failed'),
    )

    user = models.ForeignKey(User, on_delete=models.CASCADE)
    transaction_type = models.CharField(max_length=10, choices=TRANSACTION_TYPES)
    amount = models.DecimalField(
        max_digits=10, 
        decimal_places=2,
        validators=[
            MinValueValidator(Decimal('1.00'), message="Minimum amount is ₹1"),
            MaxValueValidator(Decimal('100000.00'), message="Maximum amount is ₹1,00,000")
        ]
    )
    source_currency = models.CharField(max_length=3, choices=CURRENCY_CHOICES)
    target_currency = models.CharField(max_length=3, choices=CURRENCY_CHOICES)
    recipient_name = models.CharField(max_length=100)
    recipient_account = models.CharField(max_length=50)
    recipient_bank = models.CharField(max_length=100)
    sender_name = models.CharField(max_length=100, blank=True, null=True)
    sender_account = models.CharField(max_length=50, blank=True, null=True)
    sender_bank = models.CharField(max_length=100, blank=True, null=True)
    status = models.CharField(max_length=10, choices=STATUS_CHOICES, default='pending')
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    notes = models.TextField(blank=True, null=True)
    conversion_rate = models.DecimalField(max_digits=10, decimal_places=4)
    converted_amount = models.DecimalField(max_digits=10, decimal_places=2)
    ip_address = models.GenericIPAddressField(null=True, blank=True)
    user_agent = models.CharField(max_length=255, null=True, blank=True)
    is_suspicious = models.BooleanField(default=False)
    suspicious_reason = models.CharField(max_length=500, blank=True, null=True)
    ml_risk_score = models.IntegerField(null=True, blank=True, help_text="ML model risk score percentage (0-100)")

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.transaction_type} - {self.amount} {self.source_currency} to {self.recipient_name}"

    def clean(self):
        # Check transaction frequency limit
        fifteen_minutes_ago = timezone.now() - timedelta(minutes=15)
        recent_transactions = Transaction.objects.filter(
            user=self.user,
            created_at__gte=fifteen_minutes_ago,
            status='completed'
        ).count()

        if recent_transactions >= 3:
            raise ValidationError("You can only make 3 transactions within 15 minutes. Please try again later.")

        # Validate amount limits for INR
        if self.source_currency == 'INR':
            if self.amount < Decimal('1.00'):
                raise ValidationError("Minimum amount for INR transactions is ₹1")
            if self.amount > Decimal('100000.00'):
                raise ValidationError("Maximum amount for INR transactions is ₹1,00,000")

class UserProfile(models.Model):
    user = models.OneToOneField(User, on_delete=models.CASCADE)
    phone_number = models.CharField(max_length=15, blank=True, null=True)
    address = models.TextField(blank=True, null=True)
    preferred_currency = models.CharField(max_length=3, choices=Transaction.CURRENCY_CHOICES, default='USD')
    account_number = models.CharField(max_length=20, blank=True, null=True)
    bank_name = models.CharField(max_length=100, blank=True, null=True)
    ifsc_code = models.CharField(max_length=11, blank=True, null=True)
    balance = models.DecimalField(max_digits=12, decimal_places=2, default=0.00)
    
    def __str__(self):
        return self.user.username 