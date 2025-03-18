"""
Script to update initial balance for all users in the system.
This updates both the Django database and the Excel file.
"""
import os
import sys
import django
from decimal import Decimal

# Set up Django environment
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'unionbank.settings')
django.setup()

# Import Django models and utils
from django.contrib.auth.models import User
from transactions.models import UserProfile
from transactions.utils import update_customer_balance

def update_all_balances(amount=10000):
    """
    Update all user balances to the specified amount.
    
    Args:
        amount: The amount to set as balance (default: 10000)
    """
    users = User.objects.all()
    updated_count = 0
    
    print(f"Updating balances for {users.count()} users...")
    
    for user in users:
        try:
            # Update UserProfile in Django database
            profile, created = UserProfile.objects.get_or_create(user=user)
            profile.balance = Decimal(amount)
            profile.save()
            
            # Update Excel file
            try:
                update_customer_balance(user.username, amount, is_credit=True, reset=True)
                print(f"✓ Updated balance for {user.username} to ₹{amount}")
                updated_count += 1
            except Exception as e:
                print(f"✗ Error updating Excel for {user.username}: {str(e)}")
                
        except Exception as e:
            print(f"✗ Error updating {user.username}: {str(e)}")
    
    print(f"\nSuccessfully updated {updated_count} out of {users.count()} users.")
    print("You can now make transfers with the updated balances.")

if __name__ == "__main__":
    # Get amount from command line if provided
    amount = 10000
    if len(sys.argv) > 1:
        try:
            amount = Decimal(sys.argv[1])
        except:
            print(f"Invalid amount: {sys.argv[1]}. Using default: 10000")
    
    print(f"Setting all account balances to ₹{amount}")
    update_all_balances(amount) 