# Generated by Django 4.2 on 2025-03-17 06:01

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('transactions', '0003_remove_transaction_transaction_user_id_f5864b_idx_and_more'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='transaction',
            name='ip_address',
        ),
        migrations.RemoveField(
            model_name='transaction',
            name='user_agent',
        ),
        migrations.RemoveField(
            model_name='userprofile',
            name='account_locked_until',
        ),
        migrations.RemoveField(
            model_name='userprofile',
            name='failed_login_attempts',
        ),
        migrations.RemoveField(
            model_name='userprofile',
            name='last_login_ip',
        ),
    ]
