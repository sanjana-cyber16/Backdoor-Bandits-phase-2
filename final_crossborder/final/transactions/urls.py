from django.urls import path
from . import views

urlpatterns = [
    path('dashboard/', views.dashboard, name='dashboard'),
    path('transfer/', views.transfer_money, name='transfer'),
    path('domestic-transfer/', views.domestic_transfer, name='domestic-transfer'),
    path('check-account/', views.check_account, name='check_account'),
    path('get-account-info/', views.get_account_info, name='get_account_info'),  # New endpoint for account info
    path('history/', views.transaction_history, name='history'),
    path('register/', views.register, name='register'),
    path('profile/', views.profile, name='profile'),
    path('fraud-dashboard/', views.fraud_dashboard, name='fraud_dashboard'),
] 