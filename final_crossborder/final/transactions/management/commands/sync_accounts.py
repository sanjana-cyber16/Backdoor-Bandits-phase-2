from django.core.management.base import BaseCommand
from django.contrib.auth.models import User
from transactions.account_utils import assign_account_number_to_user, sync_all_users_with_bank_database

class Command(BaseCommand):
    help = 'Synchronize all users with the bank database and assign unique account numbers'

    def handle(self, *args, **options):
        self.stdout.write(self.style.SUCCESS('Starting account synchronization...'))
        
        # Synchronize all users
        sync_all_users_with_bank_database()
        
        # Count users with account numbers
        users_with_accounts = User.objects.filter(userprofile__account_number__isnull=False).exclude(userprofile__account_number='').count()
        total_users = User.objects.count()
        
        self.stdout.write(self.style.SUCCESS(f'Account synchronization complete! {users_with_accounts}/{total_users} users have account numbers.')) 