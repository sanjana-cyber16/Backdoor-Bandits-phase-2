from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from .models import Transaction, UserProfile
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import login
import pandas as pd
from decimal import Decimal
import requests
from django.http import HttpResponse, JsonResponse
from datetime import datetime
import os
from django.core.exceptions import ValidationError
from django.utils import timezone
from datetime import timedelta
from django.core.cache import cache
from django.views.decorators.http import require_http_methods
from django.views.decorators.csrf import csrf_protect
from .utils import (
    check_fraudulent_account, 
    check_active_customer, 
    add_to_fraudster_database,
    update_customer_balance,
    check_rapid_transfer,
    check_large_incoming_transfers,
    detect_fraud_with_ml,
    analyze_recipient_transactions
)
from django.contrib.auth.models import User
from .account_utils import assign_account_number_to_user, sync_all_users_with_bank_database
from django.db.models import Count
from django.db.models.functions import TruncMonth
from django.db import models

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            user = form.save()
            
            # Get first and last name from the form if provided
            first_name = request.POST.get('first_name', '')
            last_name = request.POST.get('last_name', '')
            
            if first_name:
                user.first_name = first_name
            if last_name:
                user.last_name = last_name
            
            user.save()
            
            # Log the user in
            login(request, user)
            
            # Create user profile
            profile = UserProfile.objects.create(user=user)
            
            # Assign a unique account number
            full_name = f"{first_name} {last_name}" if first_name and last_name else None
            account_number = assign_account_number_to_user(user, full_name)
            
            messages.success(request, f'Registration successful! Your account number is: {account_number}')
            return redirect('dashboard')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def dashboard(request):
    # Ensure user has an account number
    if not hasattr(request.user, 'userprofile') or not request.user.userprofile.account_number:
        full_name = f"{request.user.first_name} {request.user.last_name}" if request.user.first_name and request.user.last_name else None
        assign_account_number_to_user(request.user, full_name)
    
    transactions = Transaction.objects.filter(user=request.user).order_by('-created_at')[:5]
    context = {
        'transactions': transactions,
        'total_sent': Transaction.objects.filter(user=request.user, transaction_type='send').count(),
        'total_received': Transaction.objects.filter(user=request.user, transaction_type='receive').count(),
    }
    return render(request, 'transactions/dashboard.html', context)

def get_client_ip(request):
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip

def get_exchange_rate(source_currency, target_currency):
    # You'll need to sign up at https://freecurrencyapi.com/ to get an API key
    api_key = os.getenv('CURRENCY_API_KEY', 'YOUR_API_KEY')  # Replace with your API key or use environment variable
    url = f"https://api.freecurrencyapi.com/v1/latest?apikey={api_key}&base_currency={source_currency}&currencies={target_currency}"
    
    try:
        response = requests.get(url)
        data = response.json()
        return Decimal(str(data['data'][target_currency]))
    except Exception as e:
        # If API call fails, use a fallback conversion rate (not recommended for production)
        default_rates = {
            'USD': {'EUR': 0.92, 'GBP': 0.79, 'JPY': 151.41, 'INR': 82.95},
            'EUR': {'USD': 1.09, 'GBP': 0.86, 'JPY': 164.93, 'INR': 90.34},
            'GBP': {'USD': 1.27, 'EUR': 1.16, 'JPY': 191.49, 'INR': 104.91},
            'JPY': {'USD': 0.0066, 'EUR': 0.0061, 'GBP': 0.0052, 'INR': 0.55},
            'INR': {'USD': 0.012, 'EUR': 0.011, 'GBP': 0.0095, 'JPY': 1.83}
        }
        return Decimal(str(default_rates[source_currency].get(target_currency, 1.0)))

@login_required
@csrf_protect
@require_http_methods(["GET", "POST"])
def transfer_money(request):
    if request.method == 'POST':
        try:
            # Rate limiting check using cache
            cache_key = f'transfer_attempt_{request.user.id}'
            attempt_count = cache.get(cache_key, 0)
            
            if attempt_count >= 3:
                messages.error(request, 'Maximum transaction limit reached. Please try again after 15 minutes.')
                return redirect('dashboard')

            # Validate amount
            try:
                amount = Decimal(request.POST.get('amount', '0'))
                if amount < Decimal('1.00'):
                    raise ValidationError("Minimum amount is ₹1")
                if amount > Decimal('100000.00'):
                    raise ValidationError("Maximum amount is ₹1,00,000")
            except (ValueError, ValidationError) as e:
                messages.error(request, str(e))
                return render(request, 'transactions/transfer.html', {
                    'currencies': Transaction.CURRENCY_CHOICES
                })

            # Check if sender has sufficient balance
            sender_balance = request.user.userprofile.balance
            if sender_balance < amount:
                messages.error(request, f'Insufficient balance. Your current balance is ₹{sender_balance}.')
                return render(request, 'transactions/transfer.html', {
                    'currencies': Transaction.CURRENCY_CHOICES,
                    'form_data': request.POST,
                    'insufficient_balance': True,
                    'current_balance': sender_balance
                })

            source_currency = request.POST.get('source_currency')
            target_currency = request.POST.get('target_currency')
            recipient_name = request.POST.get('recipient_name')
            recipient_account = request.POST.get('recipient_account')
            recipient_bank = request.POST.get('recipient_bank')
            
            # Check for fraudulent accounts
            is_fraud, fraud_reason, risk_level = check_fraudulent_account(
                account_number=recipient_account,
                recipient_name=recipient_name,
                bank_name=recipient_bank
            )
            
            if is_fraud:
                warning_message = f"This transfer has been blocked: Account found in fraud database"
                messages.error(request, warning_message)
                
                # Always block transactions to accounts in fraudster database
                error_message = f"Transaction blocked. Reason: {fraud_reason}. Risk Level: {risk_level}"
                messages.error(request, error_message)
                return render(request, 'transactions/transfer.html', {
                    'currencies': Transaction.CURRENCY_CHOICES,
                    'form_data': request.POST,
                    'is_fraudulent': True,
                    'fraud_reason': fraud_reason,
                    'risk_level': risk_level,
                    'recipient_risk': 100  # Set risk to 100%
                })
            
            # ML-based fraud detection
            transaction_data = {
                'amount': amount,
                'transaction_type': 'send',
                'recipient_bank': recipient_bank,
                'recipient_account': recipient_account,
                'sender_account': request.user.userprofile.account_number
            }
            
            is_ml_fraud, fraud_score, ml_reason = detect_fraud_with_ml(transaction_data)
            
            # Convert the score to a percentage for display
            ml_risk_score = round(fraud_score * 100)
            
            # Analyze recipient's transaction history
            recipient_risk, risk_factors = analyze_recipient_transactions(recipient_account, amount)
            
            # If recipient risk is high, flag it
            recipient_high_risk = recipient_risk > 75
            
            if is_ml_fraud:
                warning_message = f"Our fraud detection system flagged this transfer: {ml_reason}"
                messages.warning(request, warning_message)
                
                # For high confidence fraud predictions, block the transaction
                if fraud_score > 0.9:
                    messages.error(request, "Transfer blocked due to high fraud risk. Please contact customer support.")
                    return render(request, 'transactions/transfer.html', {
                        'currencies': Transaction.CURRENCY_CHOICES,
                        'form_data': request.POST,
                        'fraud_score': ml_risk_score
                    })
                else:
                    # Display warning but allow transaction to proceed
                    messages.warning(request, f"Transaction risk score: {ml_risk_score}%. Proceed with caution.")
            
            # Get real-time conversion rate
            conversion_rate = get_exchange_rate(source_currency, target_currency)
            
            # Create and validate transaction
            transaction = Transaction(
                user=request.user,
                transaction_type='send',
                amount=amount,
                source_currency=source_currency,
                target_currency=target_currency,
                recipient_name=recipient_name,
                recipient_account=recipient_account,
                recipient_bank=recipient_bank,
                conversion_rate=conversion_rate,
                converted_amount=amount * conversion_rate,
                status='pending',
                is_suspicious=is_fraud or is_ml_fraud or recipient_high_risk,
                suspicious_reason=fraud_reason or ml_reason or (recipient_high_risk and "High risk recipient profile"),
                ml_risk_score=ml_risk_score
            )

            # Validate transaction limits and amount
            try:
                transaction.full_clean()
            except ValidationError as e:
                messages.error(request, str(e))
                return render(request, 'transactions/transfer.html', {
                    'currencies': Transaction.CURRENCY_CHOICES
                })

            # Update sender's balance in UserProfile
            sender_profile = request.user.userprofile
            sender_profile.balance -= amount
            sender_profile.save()
            
            # Update sender's balance in Excel file
            update_customer_balance(request.user.username, amount, is_credit=False)
            
            # Save transaction and update rate limiting
            transaction.status = 'completed'
            transaction.save()
            
            # Try to find recipient in our system and update their balance
            try:
                recipient_account = request.POST.get('recipient_account')
                if recipient_account:
                    is_active, customer_data = check_active_customer(account_number=recipient_account)
                    if is_active:
                        recipient_username = customer_data['username']
                        recipient_user = User.objects.get(username=recipient_username)
                        
                        # Check for large incoming transfers (money laundering detection)
                        converted_amount = amount * conversion_rate
                        is_suspicious, reason = check_large_incoming_transfers(
                            user=recipient_user,
                            current_amount=converted_amount
                        )
                        
                        # If suspicious, add to fraudster database and warn user
                        if is_suspicious:
                            add_to_fraudster_database(
                                account_number=recipient_account,
                                recipient_name=recipient_user.get_full_name() or recipient_username,
                                bank_name=customer_data['bank_name'],
                                ifsc_code=customer_data.get('ifsc_code', ''),
                                reason=reason,
                                risk_level="High"
                            )
                            
                            warning_message = f"WARNING: This transaction has been flagged as suspicious. {reason}. For security reasons, this transfer has been blocked."
                            messages.error(request, warning_message)
                            
                            # Refund the sender's balance
                            sender_profile = request.user.userprofile
                            sender_profile.balance += amount
                            sender_profile.save()
                            update_customer_balance(request.user.username, amount, is_credit=True)
                            
                            # Delete the transaction
                            transaction.delete()
                            
                            return render(request, 'transactions/transfer.html', {
                                'currencies': Transaction.CURRENCY_CHOICES,
                                'form_data': request.POST,
                                'is_fraudulent': True,
                                'fraud_reason': reason,
                                'risk_level': "High"
                            })
                        
                        # Create the receive transaction
                        receive_transaction = Transaction(
                            user=recipient_user,
                            transaction_type='receive',
                            amount=amount * conversion_rate,
                            source_currency=target_currency,
                            target_currency=target_currency,
                            recipient_name=recipient_user.get_full_name() or recipient_username,
                            recipient_account=customer_data['account_number'],
                            recipient_bank=customer_data['bank_name'],
                            sender_name=request.user.get_full_name() or request.user.username,
                            sender_account=request.user.userprofile.account_number,
                            sender_bank=request.user.userprofile.bank_name,
                            conversion_rate=Decimal('1.0'),
                            converted_amount=amount * conversion_rate,
                            status='completed',
                            notes=f"Received from {request.user.username}"
                        )
                        receive_transaction.save()
                        
                        # Update recipient's balance in UserProfile
                        recipient_profile = recipient_user.userprofile
                        recipient_profile.balance += amount * conversion_rate
                        recipient_profile.save()
                        
                        # Update recipient's balance in Excel file
                        update_customer_balance(recipient_username, amount * conversion_rate, is_credit=True)
            except Exception as e:
                print(f"Error creating receive transaction: {e}")
                # Don't rollback the sender's transaction, as this might be an international transfer
                # where the recipient is not in our system
            
            # Update rate limiting cache
            cache.set(cache_key, attempt_count + 1, timeout=900)  # 15 minutes = 900 seconds
            
            messages.success(request, f'Transfer completed successfully! Converted {amount} {source_currency} to {amount * conversion_rate} {target_currency}')
            return redirect('dashboard')
            
        except Exception as e:
            messages.error(request, f'Transfer failed: {str(e)}')
            
    return render(request, 'transactions/transfer.html', {
        'currencies': Transaction.CURRENCY_CHOICES
    })

@login_required
def transaction_history(request):
    transactions = Transaction.objects.filter(user=request.user)
    
    if request.GET.get('export') == 'excel':
        # Create Excel file
        df = pd.DataFrame(list(transactions.values()))
        
        # Convert timezone-aware datetimes to timezone-naive
        for col in df.select_dtypes(include=['datetime64[ns, UTC]']).columns:
            df[col] = df[col].dt.tz_localize(None)
        
        response = HttpResponse(content_type='application/vnd.ms-excel')
        response['Content-Disposition'] = f'attachment; filename=transactions_{datetime.now().strftime("%Y%m%d")}.xlsx'
        df.to_excel(response, index=False)
        return response
        
    return render(request, 'transactions/history.html', {'transactions': transactions})

@login_required
@csrf_protect
@require_http_methods(["GET", "POST"])
def domestic_transfer(request):
    if request.method == 'POST':
        try:
            # Rate limiting check using cache
            cache_key = f'transfer_attempt_{request.user.id}'
            attempt_count = cache.get(cache_key, 0)
            
            if attempt_count >= 3:
                messages.error(request, 'Maximum transaction limit reached. Please try again after 15 minutes.')
                return redirect('dashboard')

            # Validate amount
            try:
                amount = Decimal(request.POST.get('amount', '0'))
                if amount < Decimal('1.00'):
                    raise ValidationError("Minimum amount is ₹1")
                if amount > Decimal('100000.00'):
                    raise ValidationError("Maximum amount is ₹1,00,000")
            except (ValueError, ValidationError) as e:
                messages.error(request, str(e))
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15
                })

            # Check if sender has sufficient balance
            sender_balance = request.user.userprofile.balance
            if sender_balance < amount:
                messages.error(request, f'Insufficient balance. Your current balance is ₹{sender_balance}.')
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15,
                    'form_data': request.POST,
                    'insufficient_balance': True,
                    'current_balance': sender_balance
                })

            # Get recipient details
            recipient_name = request.POST.get('recipient_name')
            recipient_account = request.POST.get('recipient_account')
            recipient_bank = request.POST.get('recipient_bank')
            ifsc_code = request.POST.get('ifsc_code', '')
            
            # Check for fraudulent accounts
            is_fraud, fraud_reason, risk_level = check_fraudulent_account(
                account_number=recipient_account,
                recipient_name=recipient_name,
                bank_name=recipient_bank,
                ifsc_code=ifsc_code
            )
            
            if is_fraud:
                warning_message = f"This transfer has been blocked: Account found in fraud database"
                messages.error(request, warning_message)
                
                # Always block transactions to accounts in fraudster database
                error_message = f"Transaction blocked. Reason: {fraud_reason}. Risk Level: {risk_level}"
                messages.error(request, error_message)
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15,
                    'form_data': request.POST,
                    'is_fraudulent': True,
                    'fraud_reason': fraud_reason,
                    'risk_level': risk_level,
                    'recipient_risk': 100  # Set risk to 100%
                })
            
            # Check for rapid transfers (potential fraud)
            is_rapid, frequency = check_rapid_transfer(
                user=request.user,
                amount=amount,
                recipient_account=recipient_account,
                within_minutes=15
            )
            
            if is_rapid:
                warning_message = f"Unusual transfer frequency detected. This is your {frequency} transfer to this account in the last 15 minutes."
                messages.warning(request, warning_message)
                
                # Add recipient to fraudster database if extremely frequent transfers
                if frequency > 5:
                    add_to_fraudster_database(
                        account_number=recipient_account,
                        recipient_name=recipient_name,
                        bank_name=recipient_bank,
                        ifsc_code=ifsc_code,
                        reason="Extremely frequent transfers in short time period",
                        risk_level="Medium"
                    )
                    
            # ML-based fraud detection
            transaction_data = {
                'amount': amount,
                'transaction_type': 'send',
                'recipient_bank': recipient_bank,
                'recipient_account': recipient_account,
                'sender_account': request.user.userprofile.account_number
            }
            
            is_ml_fraud, fraud_score, ml_reason = detect_fraud_with_ml(transaction_data)
            
            # Convert the score to a percentage for display
            ml_risk_score = round(fraud_score * 100)
            
            # Analyze recipient's transaction history
            recipient_risk, risk_factors = analyze_recipient_transactions(recipient_account, amount)
            
            # Check if recipient has a high risk profile
            recipient_high_risk = recipient_risk > 75
            if recipient_high_risk:
                risk_reason = "High risk based on recipient's transaction history"
                messages.warning(request, risk_reason)
                
                # Create a formatted list of risk factors
                risk_factors_str = ', '.join(risk_factors)
                messages.warning(request, f"Risk factors: {risk_factors_str}")
                
                # Block the transaction if high risk
                messages.error(request, "Transfer blocked due to high risk profile of the recipient. Please contact customer support.")
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15,
                    'form_data': request.POST,
                    'recipient_risk': recipient_risk,
                    'risk_factors': risk_factors
                })
            
            if is_ml_fraud:
                warning_message = f"Our fraud detection system flagged this transfer: {ml_reason}"
                messages.warning(request, warning_message)
                
                # For high confidence fraud predictions, block the transaction
                if fraud_score > 0.9:
                    messages.error(request, "Transfer blocked due to high fraud risk. Please contact customer support.")
                    return render(request, 'transactions/domestic-transfer.html', {
                        'min_amount': 1.00,
                        'max_amount': 100000.00,
                        'max_transactions_per_window': 3,
                        'time_window': 15,
                        'form_data': request.POST,
                        'fraud_score': ml_risk_score
                    })
            
            # Check if recipient is an active customer
            is_active, customer_data = check_active_customer(
                recipient_name=recipient_name,
                account_number=recipient_account
            )
            
            if not is_active:
                warning_message = "The recipient is not registered in our system. Please verify the account details."
                messages.error(request, warning_message)
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15,
                    'form_data': request.POST
                })
            
            # Verify that the recipient name matches the account
            if customer_data and customer_data['full_name'].lower() != recipient_name.lower():
                warning_message = "The recipient name does not match our records for this account number."
                messages.error(request, warning_message)
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15,
                    'form_data': request.POST
                })

            # Find the recipient user for additional checks
            try:
                recipient_username = customer_data['username']
                recipient_user = User.objects.get(username=recipient_username)
                
                # Check for large incoming transfers (money laundering detection)
                is_suspicious_large, reason_large = check_large_incoming_transfers(
                    user=recipient_user,
                    current_amount=amount
                )
                
                # If suspicious, add to fraudster database and warn user
                if is_suspicious_large:
                    add_to_fraudster_database(
                        account_number=recipient_account,
                        recipient_name=recipient_name,
                        bank_name=recipient_bank,
                        ifsc_code=ifsc_code,
                        reason=reason_large,
                        risk_level="High"
                    )
                    
                    warning_message = f"WARNING: This transaction has been flagged as suspicious. {reason_large}. For security reasons, this transfer has been blocked."
                    messages.error(request, warning_message)
                    return render(request, 'transactions/domestic-transfer.html', {
                        'min_amount': 1.00,
                        'max_amount': 100000.00,
                        'max_transactions_per_window': 3,
                        'time_window': 15,
                        'is_fraudulent': True,
                        'fraud_reason': reason_large,
                        'risk_level': "High",
                        'form_data': request.POST
                    })
            except User.DoesNotExist:
                # If recipient user doesn't exist in our system, just continue
                pass
            except Exception as e:
                print(f"Error checking large transfers: {e}")
            
            # Get client information for security tracking
            ip_address = get_client_ip(request)
            user_agent = request.META.get('HTTP_USER_AGENT', '')
            
            # Create and validate transaction (domestic transfers use INR for both source and target)
            transaction = Transaction(
                user=request.user,
                transaction_type='send',
                amount=amount,
                source_currency='INR',
                target_currency='INR',
                recipient_name=recipient_name,
                recipient_account=recipient_account,
                recipient_bank=recipient_bank,
                conversion_rate=Decimal('1.0'),  # No conversion for domestic transfers
                converted_amount=amount,
                status='pending',
                ip_address=ip_address,
                user_agent=user_agent,
                notes=f"IFSC: {ifsc_code}",  # Store IFSC code in notes
                is_suspicious=is_fraud or is_rapid or is_ml_fraud or recipient_high_risk,
                suspicious_reason=fraud_reason or ml_reason or (recipient_high_risk and "High risk recipient profile"),
                ml_risk_score=ml_risk_score
            )

            # Validate transaction limits and amount
            try:
                transaction.full_clean()
            except ValidationError as e:
                messages.error(request, str(e))
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15
                })

            # Update sender's balance in UserProfile
            sender_profile = request.user.userprofile
            sender_profile.balance -= amount
            sender_profile.save()
            
            # Update sender's balance in Excel file
            update_customer_balance(request.user.username, amount, is_credit=False)
            
            # Save transaction and update rate limiting
            transaction.status = 'completed'
            transaction.save()
            
            # Create a corresponding 'receive' transaction for the recipient
            try:
                # Find the recipient user
                recipient_username = customer_data['username']
                recipient_user = User.objects.get(username=recipient_username)
                
                # Create the receive transaction
                receive_transaction = Transaction(
                    user=recipient_user,
                    transaction_type='receive',
                    amount=amount,
                    source_currency='INR',
                    target_currency='INR',
                    recipient_name=recipient_user.get_full_name() or recipient_username,
                    recipient_account=customer_data['account_number'],
                    recipient_bank=customer_data['bank_name'],
                    sender_name=request.user.get_full_name() or request.user.username,
                    sender_account=request.user.userprofile.account_number,
                    sender_bank=request.user.userprofile.bank_name,
                    conversion_rate=Decimal('1.0'),
                    converted_amount=amount,
                    status='completed',
                    ip_address=ip_address,
                    notes=f"Received from {request.user.username}"
                )
                receive_transaction.save()
                
                # Update recipient's balance in UserProfile
                recipient_profile = recipient_user.userprofile
                recipient_profile.balance += amount
                recipient_profile.save()
                
                # Update recipient's balance in Excel file
                update_customer_balance(recipient_username, amount, is_credit=True)
                
            except User.DoesNotExist:
                # If recipient user doesn't exist in our system, just log it
                print(f"Recipient user {recipient_username} not found in our system")
            except Exception as e:
                print(f"Error creating receive transaction: {e}")
                # Rollback sender's balance update if recipient update fails
                sender_profile.balance += amount
                sender_profile.save()
                update_customer_balance(request.user.username, amount, is_credit=True)
                messages.error(request, f'Transfer failed: {str(e)}')
                return render(request, 'transactions/domestic-transfer.html', {
                    'min_amount': 1.00,
                    'max_amount': 100000.00,
                    'max_transactions_per_window': 3,
                    'time_window': 15,
                    'form_data': request.POST
                })
            
            # Update rate limiting cache
            cache.set(cache_key, attempt_count + 1, timeout=900)  # 15 minutes = 900 seconds
            
            messages.success(request, f'Domestic transfer of ₹{amount} completed successfully!')
            return redirect('dashboard')
            
        except Exception as e:
            messages.error(request, f'Transfer failed: {str(e)}')
            
    return render(request, 'transactions/domestic-transfer.html', {
        'min_amount': 1.00,
        'max_amount': 100000.00,
        'max_transactions_per_window': 3,
        'time_window': 15
    })

@require_http_methods(["POST"])
def check_account(request):
    """Check if an account is fraudulent or suspicious"""
    
    # Get form data
    account_number = request.POST.get('account_number', '')
    recipient_name = request.POST.get('recipient_name', '')
    bank_name = request.POST.get('recipient_bank', '')
    ifsc_code = request.POST.get('ifsc_code', '')
    amount = request.POST.get('amount', '0')
    
    try:
        amount_decimal = Decimal(amount)
    except (ValueError, TypeError, InvalidOperation):
        amount_decimal = Decimal('0')
    
    # Default response
    response_data = {
        'is_fraudulent': False,
        'reason': None,
        'risk_level': None,
        'is_suspicious': False,
        'suspicious_reason': None,
        'is_active': False,
        'name_mismatch': False,
        'recipient_analysis': {
            'risk_percentage': 0,
            'risk_factors': []
        }
    }
    
    # Check if fraudulent
    is_fraudulent, reason, risk_level = check_fraudulent_account(
        account_number=account_number,
        recipient_name=recipient_name,
        bank_name=bank_name,
        ifsc_code=ifsc_code
    )
    
    # Update response with fraud check results
    if is_fraudulent:
        response_data['is_fraudulent'] = True
        response_data['reason'] = reason
        response_data['risk_level'] = risk_level
        
        # If account is in fraudster database, immediately set risk to 100%
        response_data['recipient_analysis'] = {
            'risk_percentage': 100,
            'risk_factors': [f"Account found in fraud database: {reason}", f"Risk level: {risk_level}"]
        }
        
        # Return early - no need for further checks
        return JsonResponse(response_data)
    
    # Check if account exists in our system
    is_active, customer_data = check_active_customer(account_number=account_number)
    response_data['is_active'] = is_active
    
    # Check for name mismatch
    if is_active and customer_data and recipient_name and 'full_name' in customer_data:
        if customer_data['full_name'].lower() != recipient_name.lower():
            response_data['name_mismatch'] = True
    
    # Check for suspicious transfer patterns
    if request.user.is_authenticated and amount_decimal > 0:
        is_rapid, frequency = check_rapid_transfer(
            user=request.user,
            amount=amount_decimal,
            recipient_account=account_number,
            within_minutes=15
        )
        
        if is_rapid:
            response_data['is_suspicious'] = True
            response_data['suspicious_reason'] = f"Unusual transfer frequency detected. This is your {frequency} transfer to this account in the last 15 minutes."
    
    # Run ML-based fraud detection
    if amount_decimal > 0:
        transaction_data = {
            'amount': amount_decimal,
            'transaction_type': 'send',
            'recipient_bank': bank_name,
            'recipient_account': account_number,
            'sender_account': request.user.userprofile.account_number if request.user.is_authenticated else None
        }
        
        is_ml_fraud, fraud_score, ml_reason = detect_fraud_with_ml(transaction_data)
        
        if is_ml_fraud:
            response_data['is_suspicious'] = True
            response_data['suspicious_reason'] = ml_reason
            response_data['ml_risk_score'] = round(fraud_score * 100)
    
    # Analyze recipient's transaction history
    if account_number and amount_decimal > 0:
        risk_percentage, risk_factors = analyze_recipient_transactions(account_number, amount_decimal)
        
        response_data['recipient_analysis'] = {
            'risk_percentage': risk_percentage,
            'risk_factors': risk_factors
        }
        
        # If recipient analysis shows high risk, mark as suspicious
        if risk_percentage > 75:
            if not response_data['is_suspicious']:
                response_data['is_suspicious'] = True
                response_data['suspicious_reason'] = "High risk based on recipient's transaction history"
    
    return JsonResponse(response_data)

@require_http_methods(["POST"])
def get_account_info(request):
    """AJAX endpoint to get account information based on account number"""
    account_number = request.POST.get('account_number')
    
    if not account_number:
        return JsonResponse({'success': False, 'message': 'Account number is required'})
    
    # Check if account exists in our system
    is_active, customer_data = check_active_customer(account_number=account_number)
    
    if not is_active or not customer_data:
        return JsonResponse({'success': False, 'message': 'Account not found'})
    
    # Return account information
    return JsonResponse({
        'success': True,
        'full_name': customer_data.get('full_name', ''),
        'username': customer_data.get('username', ''),
        'bank_name': customer_data.get('bank_name', ''),
        'ifsc_code': customer_data.get('ifsc_code', '')
    })

@login_required
def profile(request):
    # Ensure user has an account number
    if not hasattr(request.user, 'userprofile') or not request.user.userprofile.account_number:
        full_name = f"{request.user.first_name} {request.user.last_name}" if request.user.first_name and request.user.last_name else None
        assign_account_number_to_user(request.user, full_name)
    
    if request.method == 'POST':
        # Update user information
        first_name = request.POST.get('first_name')
        last_name = request.POST.get('last_name')
        phone_number = request.POST.get('phone_number')
        address = request.POST.get('address')
        
        # Update user model
        request.user.first_name = first_name
        request.user.last_name = last_name
        request.user.save()
        
        # Update user profile
        profile = request.user.userprofile
        profile.phone_number = phone_number
        profile.address = address
        profile.save()
        
        # Check if name changed and update in active customers database
        full_name = f"{first_name} {last_name}"
        if full_name.strip():
            try:
                from .account_utils import add_user_to_active_customers
                add_user_to_active_customers(request.user, profile.account_number)
            except Exception as e:
                print(f"Error updating active customers database: {e}")
        
        messages.success(request, 'Profile updated successfully!')
        return redirect('profile')
    
    return render(request, 'transactions/profile.html')

@login_required
def fraud_dashboard(request):
    """
    Dashboard showing fraud statistics and ML model predictions
    """
    # Only allow access to staff members for this sensitive information
    if not request.user.is_staff:
        messages.error(request, "You don't have permission to access this page.")
        return redirect('dashboard')
        
    # Get all transactions
    all_transactions = Transaction.objects.all()
    total_count = all_transactions.count()
    
    # Get suspicious transactions
    suspicious_transactions = all_transactions.filter(is_suspicious=True)
    suspicious_count = suspicious_transactions.count()
    
    # Calculate percentage
    fraud_percentage = 0
    if total_count > 0:
        fraud_percentage = round((suspicious_count / total_count) * 100, 2)
    
    # Get transactions by month (for chart)
    transactions_by_month = (
        all_transactions
        .annotate(month=TruncMonth('created_at'))
        .values('month')
        .annotate(total=Count('id'), suspicious=Count('id', filter=models.Q(is_suspicious=True)))
        .order_by('month')
    )
    
    # Format data for charts
    months = []
    totals = []
    suspicious_counts = []
    
    for item in transactions_by_month:
        months.append(item['month'].strftime('%b %Y'))
        totals.append(item['total'])
        suspicious_counts.append(item['suspicious'])
    
    # Recent suspicious transactions
    recent_suspicious = suspicious_transactions.order_by('-created_at')[:10]
    
    # Get transactions by fraud reason
    fraud_reasons = {}
    for transaction in suspicious_transactions:
        if transaction.suspicious_reason:
            reason = transaction.suspicious_reason
            if len(reason) > 50:
                reason = reason[:50] + "..."
            fraud_reasons[reason] = fraud_reasons.get(reason, 0) + 1
    
    # Sort by frequency
    sorted_reasons = {k: v for k, v in sorted(fraud_reasons.items(), key=lambda item: item[1], reverse=True)}
    top_reasons = dict(list(sorted_reasons.items())[:5])
    
    # ML Risk Score distribution
    transactions_with_ml_score = all_transactions.exclude(ml_risk_score__isnull=True)
    ml_score_count = transactions_with_ml_score.count()
    
    # Risk categories
    risk_categories = {
        'Low Risk (0-50%)': 0,
        'Medium Risk (51-80%)': 0,
        'High Risk (81-100%)': 0
    }
    
    # Count transactions in each risk category
    for transaction in transactions_with_ml_score:
        if transaction.ml_risk_score <= 50:
            risk_categories['Low Risk (0-50%)'] += 1
        elif transaction.ml_risk_score <= 80:
            risk_categories['Medium Risk (51-80%)'] += 1
        else:
            risk_categories['High Risk (81-100%)'] += 1
    
    # Calculate average ML risk score
    avg_ml_score = 0
    if ml_score_count > 0:
        avg_ml_score = round(sum(t.ml_risk_score for t in transactions_with_ml_score) / ml_score_count, 1)
    
    context = {
        'total_count': total_count,
        'suspicious_count': suspicious_count,
        'fraud_percentage': fraud_percentage,
        'recent_suspicious': recent_suspicious,
        'months': months,
        'totals': totals,
        'suspicious_counts': suspicious_counts,
        'top_reasons': top_reasons,
        'risk_categories': risk_categories,
        'avg_ml_score': avg_ml_score,
        'ml_score_count': ml_score_count,
    }
    
    return render(request, 'transactions/fraud_dashboard.html', context) 