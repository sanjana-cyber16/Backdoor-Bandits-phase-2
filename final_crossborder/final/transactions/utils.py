import pandas as pd
import os
from django.conf import settings
from datetime import datetime, timedelta, timezone
from decimal import Decimal
import joblib
import numpy as np
from django.utils import timezone

# Load the fraud detection model once when the module is imported
try:
    FRAUD_MODEL_PATH = os.path.join(settings.BASE_DIR, 'fraud_detection.pkl')
    fraud_model = joblib.load(FRAUD_MODEL_PATH)
    print("Fraud detection model loaded successfully")
except Exception as e:
    print(f"Error loading fraud detection model: {e}")
    fraud_model = None

def load_fraudster_database():
    """Load the fraudster database from Excel file"""
    try:
        file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'fraudster_database.xlsx')
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading fraudster database: {e}")
        return pd.DataFrame()  # Return empty DataFrame if file not found or error

def load_active_customers():
    """Load the active customers database from Excel file"""
    try:
        file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'active_customers.xlsx')
        return pd.read_excel(file_path)
    except Exception as e:
        print(f"Error loading active customers database: {e}")
        return pd.DataFrame()  # Return empty DataFrame if file not found or error

def check_fraudulent_account(account_number=None, recipient_name=None, bank_name=None, ifsc_code=None):
    """
    Check if the provided details match any known fraudulent accounts
    Returns a tuple of (is_fraudulent, reason, risk_level)
    """
    df = load_fraudster_database()
    
    if df.empty:
        return False, None, None
    
    # Create filters for each provided parameter
    filters = []
    if account_number:
        filters.append(df['account_number'].astype(str) == str(account_number))
    if recipient_name:
        filters.append(df['recipient_name'].str.lower() == recipient_name.lower())
    if bank_name:
        filters.append(df['bank_name'].str.lower() == bank_name.lower())
    if ifsc_code:
        filters.append(df['ifsc_code'].str.lower() == ifsc_code.lower())
    
    # If no filters provided, return not fraudulent
    if not filters:
        return False, None, None
    
    # Combine filters with OR logic (any match is suspicious)
    combined_filter = filters[0]
    for f in filters[1:]:
        combined_filter = combined_filter | f
    
    matches = df[combined_filter]
    
    if not matches.empty:
        # Return the first match's reason and risk level
        first_match = matches.iloc[0]
        return True, first_match['reason'], first_match['risk_level']
    
    return False, None, None

def check_active_customer(recipient_name=None, account_number=None):
    """
    Check if the recipient is an active customer in our system
    Returns a tuple of (is_active, customer_data)
    """
    df = load_active_customers()
    
    if df.empty:
        return False, None
    
    # Check if account exists
    if account_number:
        account_matches = df[df['account_number'].astype(str) == str(account_number)]
        if not account_matches.empty:
            return True, account_matches.iloc[0].to_dict()
    
    # Check if name exists
    if recipient_name:
        name_matches = df[df['full_name'].str.lower() == recipient_name.lower()]
        if not name_matches.empty:
            return True, name_matches.iloc[0].to_dict()
            
    # If we get here, no match was found
    return False, None

def add_to_fraudster_database(account_number, recipient_name, bank_name=None, ifsc_code=None, reason="Suspicious rapid transfer pattern", risk_level="Medium"):
    """
    Add a new entry to the fraudster database
    """
    try:
        # Load existing database
        df = load_fraudster_database()
        
        # Check if account already exists in database
        if not df.empty:
            existing = df[df['account_number'].astype(str) == str(account_number)]
            if not existing.empty:
                # Account already in database, no need to add again
                return True
        
        # Create new entry
        new_entry = pd.DataFrame([{
            "account_number": account_number,
            "recipient_name": recipient_name,
            "bank_name": bank_name if bank_name else "",
            "ifsc_code": ifsc_code if ifsc_code else "",
            "reason": reason,
            "risk_level": risk_level,
            "date_added": datetime.now().strftime("%Y-%m-%d")
        }])
        
        # Append to existing database
        updated_df = pd.concat([df, new_entry], ignore_index=True)
        
        # Save updated database
        file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'fraudster_database.xlsx')
        updated_df.to_excel(file_path, index=False)
        
        return True
    except Exception as e:
        print(f"Error adding to fraudster database: {e}")
        return False

def update_customer_balance(username, amount, is_credit=True, reset=False):
    """
    Update a customer's balance in the active customers database
    
    Args:
        username: The username of the customer
        amount: The amount to add/subtract/set
        is_credit: True to add, False to subtract (ignored if reset=True)
        reset: If True, set the balance to the amount instead of adding/subtracting
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Load existing database
        df = load_active_customers()
        
        if df.empty:
            return False
        
        # Find the customer
        customer_idx = df[df['username'] == username].index
        if len(customer_idx) == 0:
            return False
        
        # Update balance
        if reset:
            # Set the balance to the specified amount
            df.loc[customer_idx[0], 'balance'] = float(amount)
        else:
            # Add or subtract the amount
            if is_credit:
                df.loc[customer_idx[0], 'balance'] += float(amount)
            else:
                df.loc[customer_idx[0], 'balance'] -= float(amount)
        
        # Save updated database
        file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'active_customers.xlsx')
        df.to_excel(file_path, index=False)
        
        return True
    except Exception as e:
        print(f"Error updating customer balance: {e}")
        return False

def check_rapid_transfer(user, amount, recipient_account, within_minutes=15):
    """
    Check if this user has received and then sent a similar amount within the specified time window
    Returns a tuple of (is_suspicious, transaction_data)
    """
    from .models import Transaction
    
    try:
        # Get recent received transactions for this user
        time_threshold = datetime.now() - timedelta(minutes=within_minutes)
        recent_received = Transaction.objects.filter(
            user=user,
            transaction_type='receive',
            created_at__gte=time_threshold,
            status='completed'
        ).order_by('-created_at')
        
        if not recent_received:
            return False, None
        
        # Check if any received amount is similar to the amount being sent
        for transaction in recent_received:
            # Check if amounts are within 10% of each other
            received_amount = float(transaction.amount)
            sending_amount = float(amount)
            
            if abs(received_amount - sending_amount) / received_amount <= 0.1:  # Within 10%
                return True, transaction
        
        return False, None
    except Exception as e:
        print(f"Error checking rapid transfer: {e}")
        return False, None

def check_large_incoming_transfers(user, current_amount):
    """
    Check if a user has received large amounts of money in a suspicious pattern
    
    Criteria:
    1. First receives total sum >= 2 lakhs (200,000)
    2. Then receives another large amount (4-10 lakhs)
    
    Args:
        user: The user to check
        current_amount: The current incoming amount
        
    Returns:
        tuple: (is_suspicious, reason)
    """
    from .models import Transaction
    
    try:
        # Get all completed receive transactions for this user
        past_receives = Transaction.objects.filter(
            user=user,
            transaction_type='receive',
            status='completed'
        ).order_by('created_at')
        
        if not past_receives:
            return False, None
        
        # Calculate total received amount before this transaction
        total_received = sum(Decimal(str(t.amount)) for t in past_receives)
        
        # Convert current_amount to Decimal if it's not already
        if not isinstance(current_amount, Decimal):
            current_amount = Decimal(str(current_amount))
        
        # Check for suspicious pattern:
        # 1. Already received >= 2 lakhs (200,000)
        # 2. Current incoming amount is large (4-10 lakhs)
        if total_received >= Decimal('200000'):
            if current_amount >= Decimal('400000') and current_amount <= Decimal('1000000'):
                reason = f"Account previously received ₹{total_received:,.2f} and is now receiving another ₹{current_amount:,.2f}"
                return True, reason
        
        return False, None
    except Exception as e:
        print(f"Error checking large incoming transfers: {e}")
        return False, None

def detect_fraud_with_ml(transaction_data):
    """
    Use the machine learning model to detect if a transaction is potentially fraudulent.
    If the ML model fails, fallback to rule-based risk assessment.
    
    Args:
        transaction_data: Dictionary containing the transaction details
        
    Returns:
        tuple: (is_fraudulent, score, reason)
            - is_fraudulent: Boolean indicating if the transaction is likely fraudulent
            - score: Fraud probability score (0-1)
            - reason: Reason for flagging as fraud (or None if not fraudulent)
    """
    # Try ML-based prediction first
    ml_result = _predict_with_ml_model(transaction_data)
    
    # If ML prediction returns very low scores for everything (likely model issue),
    # fallback to rule-based assessment
    if ml_result[1] < 0.01:
        return _rule_based_fraud_detection(transaction_data)
    else:
        return ml_result
    
def _predict_with_ml_model(transaction_data):
    """Try to use the ML model for prediction"""
    if fraud_model is None:
        print("Warning: Fraud detection model not available")
        return False, 0.3, "Fraud detection model not available"
    
    try:
        # Extract features needed for prediction
        # Convert string/object types to numeric where possible to match model expectations
        user_id = transaction_data.get('sender_account', '0')
        user_id_numeric = int(hash(user_id) % 10000)  # Convert to numeric hash
        
        account_num = transaction_data.get('recipient_account', '0')
        account_num_numeric = int(hash(account_num) % 10000)  # Convert to numeric hash
        
        # Device encoding: web=1, mobile=2, other=0
        device_map = {'web': 1, 'mobile': 2}
        device = transaction_data.get('device', 'web')
        device_code = device_map.get(device, 0)
        
        # Location encoding: online=1, branch=2, other=0
        location_map = {'online': 1, 'branch': 2}
        location = transaction_data.get('location', 'online')
        location_code = location_map.get(location, 0)
        
        # Payment method encoding: bank_transfer=1, card=2, wallet=3, other=0
        payment_map = {'bank_transfer': 1, 'card': 2, 'wallet': 3}
        payment = transaction_data.get('payment_method', 'bank_transfer')
        payment_code = payment_map.get(payment, 0)
        
        # Risk level encoding: Low=1, Medium=2, High=3
        risk_map = {'Low': 1, 'Medium': 2, 'High': 3}
        risk = transaction_data.get('risk_level', 'Medium') 
        risk_code = risk_map.get(risk, 2)
        
        features = {
            'User_ID': user_id_numeric,
            'Transaction_Amount': float(transaction_data.get('amount', 0)),
            'Transaction_Type': 1 if transaction_data.get('transaction_type', '') == 'send' else 0,
            'Time_of_Transaction': datetime.now().hour,
            'Device_Used': device_code,
            'Location': location_code,
            'Previous_Fraudulent_Transactions': 0,
            'Account_Age': 365,  # Default to 1 year
            'Number_of_Transactions_Last_24H': 1,
            'Payment_Method': payment_code,
            'account_number': account_num_numeric,
            'reason': 0,  # Default to 0 for reason code
            'risk_level': risk_code
        }
        
        # Convert to DataFrame
        df = pd.DataFrame([features])
        
        # Ensure all columns are numeric types
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Make prediction
        try:
            # Try predict_proba first (for classifiers that support probability)
            fraud_score = fraud_model.predict_proba(df)[0][1]  # Index 1 is typically the fraud class
        except (AttributeError, IndexError):
            try:
                # Fallback to predict if predict_proba is not available
                prediction = fraud_model.predict(df)[0]
                fraud_score = float(prediction)
                if isinstance(prediction, (np.ndarray, list)):
                    fraud_score = float(prediction[0])
                # Normalize score to 0-1 range if needed
                if fraud_score > 1:
                    fraud_score = fraud_score / 100
            except Exception as predict_error:
                print(f"Error in model prediction: {predict_error}")
                return False, 0.3, f"Error in prediction: {str(predict_error)}"
        
        # Threshold can be adjusted based on desired sensitivity
        is_fraudulent = fraud_score > 0.7
        
        reason = None
        if is_fraudulent:
            # Create a more descriptive reason
            reason_parts = []
            if float(transaction_data.get('amount', 0)) > 50000:
                reason_parts.append("high transaction amount")
            if datetime.now().hour < 5 or datetime.now().hour > 22:
                reason_parts.append("unusual transaction time")
            if transaction_data.get('source_currency', 'INR') != transaction_data.get('target_currency', 'INR'):
                reason_parts.append("international transfer")
                
            if reason_parts:
                reason = f"ML model detected unusual pattern: {', '.join(reason_parts)} (score: {fraud_score:.2f})"
            else:
                reason = f"ML model detected unusual transaction pattern (score: {fraud_score:.2f})"
        
        return is_fraudulent, fraud_score, reason
        
    except Exception as e:
        # Log the detailed error
        print(f"Error during fraud detection: {e}")
        print(f"Transaction data: {transaction_data}")
        # Return a more specific error message
        return False, 0.3, f"Error analyzing transaction: {str(e)[:100]}"

def _rule_based_fraud_detection(transaction_data):
    """
    Rule-based fraud detection as fallback when ML model isn't performing well
    This implements a heuristic approach based on transaction characteristics
    """
    risk_score = 0.0
    risk_factors = []
    
    # Extract transaction details with default values
    amount = float(transaction_data.get('amount', 0))
    sender_account = transaction_data.get('sender_account', '')
    recipient_account = transaction_data.get('recipient_account', '')
    source_currency = transaction_data.get('source_currency', 'INR')
    target_currency = transaction_data.get('target_currency', 'INR')
    transaction_time = datetime.now().hour
    device = transaction_data.get('device', 'web')
    location = transaction_data.get('location', 'online')
    payment_method = transaction_data.get('payment_method', 'bank_transfer')
    recipient_bank = transaction_data.get('recipient_bank', '')
    
    # 1. Check transaction amount (higher amounts = higher risk)
    if amount > 100000:
        risk_score += 0.35
        risk_factors.append("very large transaction amount")
    elif amount > 50000:
        risk_score += 0.25
        risk_factors.append("large transaction amount")
    elif amount > 25000:
        risk_score += 0.15
        risk_factors.append("medium-high transaction amount")
    elif amount > 10000:
        risk_score += 0.05
    
    # 2. Check for international transfers (higher risk)
    if source_currency != target_currency:
        risk_score += 0.20
        risk_factors.append("international transfer")
        
        # Higher risk for certain destination currencies
        high_risk_currencies = ['USD', 'EUR', 'GBP']
        if target_currency in high_risk_currencies:
            risk_score += 0.05
    
    # 3. Check transaction time (transactions at unusual hours are higher risk)
    if transaction_time < 5 or transaction_time > 22:
        risk_score += 0.15
        risk_factors.append("unusual transaction time")
    
    # 4. Check device and location combination
    if device == 'mobile' and location == 'online':
        risk_score += 0.05
    elif device == 'web' and location == 'online':
        risk_score += 0.02
    
    # 5. Check payment method (some methods are higher risk)
    if payment_method == 'wallet':
        risk_score += 0.10
        risk_factors.append("high-risk payment method")
    elif payment_method == 'card':
        risk_score += 0.05
    
    # 6. Check bank name (if available)
    if recipient_bank:
        suspicious_banks = ['Unknown Bank', 'Foreign Bank', 'Test Bank']
        if any(suspicious in recipient_bank for suspicious in suspicious_banks):
            risk_score += 0.15
            risk_factors.append("suspicious recipient bank")
    
    # 7. Apply account-specific checks
    if not sender_account or not recipient_account:
        risk_score += 0.10
        risk_factors.append("missing account information")
    
    # Cap the risk score at 1.0
    risk_score = min(risk_score, 1.0)
    
    # Determine if the transaction should be flagged as fraudulent
    is_fraudulent = risk_score > 0.7
    
    # Prepare reason text
    reason = None
    if risk_factors:
        if is_fraudulent:
            reason = f"High-risk transaction detected: {', '.join(risk_factors)}"
        elif risk_score > 0.4:
            reason = f"Medium-risk transaction: {', '.join(risk_factors)}"
    
    return is_fraudulent, risk_score, reason

def analyze_recipient_transactions(recipient_account, transaction_amount):
    """
    Analyze the recipient's past transactions to detect suspicious patterns
    
    Args:
        recipient_account: The recipient's account number
        transaction_amount: The current transaction amount
        
    Returns:
        tuple: (risk_percentage, risk_factors)
            - risk_percentage: 0-100 indicating risk level
            - risk_factors: List of detected risk factors
    """
    from .models import Transaction
    
    risk_factors = []
    base_risk = 0  # Base risk percentage
    
    try:
        # Input validation
        if not recipient_account or not isinstance(recipient_account, str):
            risk_factors.append("Invalid recipient account format")
            return 40, risk_factors
            
        try:
            transaction_amount = float(transaction_amount)
            if transaction_amount <= 0:
                risk_factors.append("Invalid transaction amount")
                return 40, risk_factors
        except (ValueError, TypeError):
            risk_factors.append("Transaction amount must be a number")
            return 40, risk_factors
            
        # First check if account is in fraudster database
        is_fraudulent, reason, risk_level = check_fraudulent_account(account_number=recipient_account)
        
        if is_fraudulent:
            risk_factors.append(f"Account found in fraud database: {reason}")
            if risk_level == "High":
                risk_factors.append("High-risk flagged account")
            return 100, risk_factors  # Immediately return 100% risk
        
        # Get all transactions where this account was the recipient
        past_transactions = Transaction.objects.filter(
            recipient_account=recipient_account,
            status='completed'
        ).order_by('-created_at')
        
        # If no transaction history, assign a moderate base risk
        if not past_transactions.exists():
            risk_factors.append("No transaction history available")
            base_risk = 40
            return base_risk, risk_factors
            
        # Get total received amount and count
        total_received = sum(t.converted_amount for t in past_transactions)
        transaction_count = past_transactions.count()
        
        # Check for unusually large transaction amounts
        avg_transaction = total_received / transaction_count if transaction_count > 0 else 0
        if transaction_amount > avg_transaction * 5 and transaction_amount > 10000:
            risk_factors.append(f"Amount is {round(transaction_amount / avg_transaction, 1)}x larger than average ({avg_transaction:.2f})")
            base_risk += 25
        
        # Check for frequency of suspicious transactions
        suspicious_count = past_transactions.filter(is_suspicious=True).count()
        suspicious_percentage = (suspicious_count / transaction_count * 100) if transaction_count > 0 else 0
        
        if suspicious_percentage > 50:
            risk_factors.append(f"{round(suspicious_percentage)}% of past transactions were flagged as suspicious")
            base_risk += 35
        elif suspicious_percentage > 25:
            risk_factors.append(f"{round(suspicious_percentage)}% of past transactions had suspicious indicators")
            base_risk += 20
        
        # Check for rapid money movement patterns (money mule indicator)
        last_week = timezone.now() - timedelta(days=7)
        recent_transactions = past_transactions.filter(created_at__gte=last_week)
        
        received_then_sent = False
        for transaction in recent_transactions:
            # Check if this account received money and then sent it out quickly
            received_time = transaction.created_at
            sent_after_receiving = Transaction.objects.filter(
                sender_account=recipient_account,
                created_at__gt=received_time,
                created_at__lt=received_time + timedelta(hours=24)
            ).exists()
            
            if sent_after_receiving:
                received_then_sent = True
                break
        
        if received_then_sent:
            risk_factors.append("Rapid money movement detected (received and sent within 24 hours)")
            base_risk += 30
        
        # Consider previous ML risk scores if they're available
        try:
            ml_scored_transactions = past_transactions.exclude(ml_risk_score__isnull=True)
            if ml_scored_transactions.exists():
                avg_ml_score = sum(t.ml_risk_score for t in ml_scored_transactions) / ml_scored_transactions.count()
                
                if avg_ml_score > 75:
                    risk_factors.append(f"Average ML risk score of {round(avg_ml_score)}% from past transactions")
                    base_risk += 25
                elif avg_ml_score > 50:
                    risk_factors.append(f"Moderate ML risk score ({round(avg_ml_score)}%) from past transactions")
                    base_risk += 15
        except Exception as ml_error:
            # If there's an error with ML scores, just continue without them
            print(f"Error processing ML scores: {ml_error}")
        
        # Cap the risk at 100%
        final_risk = min(base_risk, 100)
        
        return final_risk, risk_factors
        
    except Exception as e:
        print(f"Error analyzing recipient transactions: {e}")
        
        # Return a more informative error message
        error_type = type(e).__name__
        risk_factors.append(f"Error during analysis: {error_type}")
        
        # Add debug information
        try:
            if 'recipient_account' not in locals() or recipient_account is None:
                risk_factors.append("Invalid recipient account")
            elif 'transaction_amount' not in locals() or transaction_amount is None:
                risk_factors.append("Invalid transaction amount")
            else:
                if isinstance(recipient_account, str) and len(recipient_account) > 4:
                    risk_factors.append(f"For account: {recipient_account[:4]}... Amount: {transaction_amount}")
                else:
                    risk_factors.append(f"Amount: {transaction_amount}")
        except:
            pass
            
        return 30, risk_factors 