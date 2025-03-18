"""
WSGI config for unionbank project.
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'unionbank.settings')

application = get_wsgi_application() 