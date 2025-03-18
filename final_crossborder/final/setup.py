import os
import sys
import subprocess
import django
from django.core.management import execute_from_command_line

def main():
    """Run administrative tasks."""
    os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'unionbank.settings')
    try:
        from django.core.management import execute_from_command_line
    except ImportError as exc:
        raise ImportError(
            "Couldn't import Django. Are you sure it's installed?"
        ) from exc

    # Install requirements
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])

    # Make migrations
    execute_from_command_line(['manage.py', 'makemigrations'])
    execute_from_command_line(['manage.py', 'migrate'])

    # Create superuser
    print("\nCreating superuser...")
    execute_from_command_line(['manage.py', 'createsuperuser'])

    print("\nSetup complete! You can now run the server with:")
    print("python manage.py runserver")

if __name__ == '__main__':
    main() 