import random
import string
import pandas as pd
import os
from django.conf import settings
from django.db import transaction

def generate_unique_account_number(existing_numbers=None):
    """
    Generate a unique 12-digit account number that doesn't exist in the database
    or in the provided list of existing numbers
    """
    if existing_numbers is None:
        existing_numbers = []
    
    # Load active customers from Excel to check against bank database
    try:
        file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'active_customers.xlsx')
        df = pd.read_excel(file_path)
        bank_account_numbers = df['account_number'].astype(str).tolist()
        existing_numbers.extend(bank_account_numbers)
    except Exception as e:
        print(f"Error loading active customers database: {e}")
    
    # Generate a unique account number
    while True:
        # Generate a 12-digit account number
        account_number = ''.join(random.choices(string.digits, k=12))
        
        # Check if it's unique
        if account_number not in existing_numbers:
            return account_number

def assign_account_number_to_user(user, full_name=None):
    """
    Assign a unique account number to a user
    If the user already has an account number in the bank database, use that instead
    """
    from .models import UserProfile
    
    # If user doesn't have a profile, create one
    profile, created = UserProfile.objects.get_or_create(user=user)
    
    # If user already has an account number, return it
    if profile.account_number:
        return profile.account_number
    
    # Try to find the user in the bank database by name
    if full_name:
        try:
            file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'active_customers.xlsx')
            df = pd.read_excel(file_path)
            
            # Look for matching name
            name_match = df[df['full_name'].str.lower() == full_name.lower()]
            if not name_match.empty:
                # Use the account number from the bank database
                account_number = str(name_match.iloc[0]['account_number'])
                bank_name = name_match.iloc[0]['bank_name']
                ifsc_code = name_match.iloc[0]['ifsc_code']
                balance = float(name_match.iloc[0]['balance'])
                
                # Update user profile with bank details
                profile.account_number = account_number
                profile.bank_name = bank_name
                profile.ifsc_code = ifsc_code
                profile.balance = balance
                profile.save()
                
                return account_number
        except Exception as e:
            print(f"Error checking bank database for user: {e}")
    
    # Get all existing account numbers from user profiles
    existing_numbers = list(UserProfile.objects.exclude(account_number__isnull=True)
                           .exclude(account_number='')
                           .values_list('account_number', flat=True))
    
    # Generate a new unique account number
    account_number = generate_unique_account_number(existing_numbers)
    
    # Assign it to the user
    profile.account_number = account_number
    
    # If no bank name is set, assign a default one
    if not profile.bank_name:
        profile.bank_name = "Union Bank"
    
    # If no IFSC code is set, generate one
    if not profile.ifsc_code:
        profile.ifsc_code = 'UNBN0' + ''.join(random.choices(string.digits, k=6))
    
    profile.save()
    
    # Add the user to the active customers database
    add_user_to_active_customers(user, account_number)
    
    return account_number

def add_user_to_active_customers(user, account_number):
    """
    Add a user to the active customers database
    """
    try:
        file_path = os.path.join(settings.BASE_DIR, 'static', 'data', 'active_customers.xlsx')
        
        # Load existing database or create new one if it doesn't exist
        try:
            df = pd.read_excel(file_path)
        except:
            df = pd.DataFrame(columns=[
                "username", "full_name", "account_number", "bank_name", 
                "ifsc_code", "balance", "is_active", "date_joined"
            ])
        
        # Check if user already exists in database
        if not df.empty:
            existing = df[df['username'] == user.username]
            if not existing.empty:
                # User already in database, update their details
                idx = existing.index[0]
                df.loc[idx, 'account_number'] = account_number
                df.loc[idx, 'bank_name'] = user.userprofile.bank_name
                df.loc[idx, 'ifsc_code'] = user.userprofile.ifsc_code
                df.loc[idx, 'balance'] = float(user.userprofile.balance)
                df.loc[idx, 'is_active'] = True
                df.to_excel(file_path, index=False)
                return
        
        # Create new entry
        full_name = f"{user.first_name} {user.last_name}" if user.first_name and user.last_name else user.username
        
        new_entry = pd.DataFrame([{
            "username": user.username,
            "full_name": full_name,
            "account_number": account_number,
            "bank_name": user.userprofile.bank_name,
            "ifsc_code": user.userprofile.ifsc_code,
            "balance": float(user.userprofile.balance),
            "is_active": True,
            "date_joined": pd.Timestamp.now().strftime("%Y-%m-%d")
        }])
        
        # Append to existing database
        updated_df = pd.concat([df, new_entry], ignore_index=True)
        
        # Save updated database
        updated_df.to_excel(file_path, index=False)
        
    except Exception as e:
        print(f"Error adding user to active customers database: {e}")

def sync_all_users_with_bank_database():
    """
    Synchronize all users with the bank database
    - Assign account numbers to users who don't have one
    - Update user details from bank database if they exist there
    """
    from django.contrib.auth.models import User
    
    # Get all users
    users = User.objects.all()
    
    for user in users:
        # Get or create user profile
        profile, created = UserProfile.objects.get_or_create(user=user)
        
        # If user doesn't have an account number, assign one
        if not profile.account_number:
            full_name = f"{user.first_name} {user.last_name}" if user.first_name and user.last_name else None
            assign_account_number_to_user(user, full_name)
        else:
            # Make sure user is in the active customers database
            add_user_to_active_customers(user, profile.account_number) 