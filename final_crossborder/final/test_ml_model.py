import os
import sys
import django
import pandas as pd
from tabulate import tabulate
from datetime import datetime

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'unionbank.settings')
django.setup()

# Now import the utilities
from transactions.utils import detect_fraud_with_ml, analyze_recipient_transactions

def test_ml_model_with_varied_inputs():
    """Test the ML model with multiple transactions with different characteristics"""
    print("Testing ML fraud detection model with varied inputs...")
    
    # Different test transaction cases
    test_transactions = [
        {
            "name": "Small Amount Domestic",
            "data": {
                'amount': 500,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'INR',
                'recipient_bank': 'SBI',
                'recipient_account': '123456789',
                'sender_account': '987654321',
                'device': 'mobile',
                'location': 'branch',
                'payment_method': 'bank_transfer',
                'risk_level': 'Low'
            }
        },
        {
            "name": "Large Amount Domestic",
            "data": {
                'amount': 95000,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'INR',
                'recipient_bank': 'HDFC',
                'recipient_account': '234567890',
                'sender_account': '987654321',
                'device': 'web',
                'location': 'online',
                'payment_method': 'bank_transfer',
                'risk_level': 'Medium'
            }
        },
        {
            "name": "Medium International Transfer",
            "data": {
                'amount': 25000,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'USD',
                'recipient_bank': 'Chase Bank',
                'recipient_account': '345678901',
                'sender_account': '987654321',
                'device': 'web',
                'location': 'online',
                'payment_method': 'card',
                'risk_level': 'Medium'
            }
        },
        {
            "name": "Large International Transfer",
            "data": {
                'amount': 150000,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'EUR',
                'recipient_bank': 'Deutsche Bank',
                'recipient_account': '456789012',
                'sender_account': '987654321',
                'device': 'web',
                'location': 'online',
                'payment_method': 'bank_transfer',
                'risk_level': 'High'
            }
        },
        {
            "name": "Suspicious Account Patterns",
            "data": {
                'amount': 75000,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'INR',
                'recipient_bank': 'Unknown Bank',
                'recipient_account': '567890123',
                'sender_account': '987654321',
                'device': 'mobile',
                'location': 'online',
                'payment_method': 'wallet',
                'risk_level': 'High'
            }
        }
    ]
    
    # Run tests and collect results
    results = []
    
    for test_case in test_transactions:
        # Run fraud detection
        is_fraud, score, reason = detect_fraud_with_ml(test_case["data"])
        
        # Store results
        results.append({
            "Test Case": test_case["name"],
            "Amount": f"₹{test_case['data']['amount']:,}",
            "Currency": f"{test_case['data']['source_currency']} -> {test_case['data']['target_currency']}",
            "Payment": test_case["data"]["payment_method"],
            "Bank": test_case["data"]["recipient_bank"],
            "Is Fraudulent": "✓" if is_fraud else "✗",
            "Risk Score": f"{score:.2f}",
            "Risk %": f"{score*100:.1f}%",
            "Risk Factors": reason if reason else "None detected"
        })
    
    # Display results in table format
    print("\nML Model Test Results:")
    print(tabulate(results, headers="keys", tablefmt="grid"))
    
    # Check if all risk scores are identical
    scores = [float(result["Risk Score"]) for result in results]
    if len(set([round(s, 2) for s in scores])) == 1:
        print("\n⚠️ WARNING: All risk scores are identical! The model may not be working correctly.")
    else:
        print(f"\nRisk score variation: Min={min(scores):.2f}, Max={max(scores):.2f}, Range={max(scores)-min(scores):.2f}")
        print("✅ The risk scoring system is now differentiating between different transaction types.")
    
    return results

def test_edge_cases():
    """Test unusual or edge cases to see how the model handles them"""
    print("\nTesting edge cases...")
    
    edge_cases = [
        {
            "name": "Very Large Amount",
            "data": {
                'amount': 1000000,  # 10 Lakh INR
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'INR',
                'recipient_bank': 'SBI',
                'recipient_account': '123456789',
                'sender_account': '987654321',
            }
        },
        {
            "name": "Late Night International",
            "data": {
                'amount': 50000,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'USD',
                'recipient_bank': 'Chase Bank',
                'recipient_account': '345678901',
                'sender_account': '987654321',
                # Force time to be late night for testing
                '_override_hour': 2  # 2 AM
            }
        },
        {
            "name": "Multiple Risk Factors",
            "data": {
                'amount': 125000,
                'transaction_type': 'send',
                'source_currency': 'INR',
                'target_currency': 'USD',
                'recipient_bank': 'Unknown Bank',
                'recipient_account': '345678901',
                'sender_account': '987654321',
                'payment_method': 'wallet',
                '_override_hour': 23  # 11 PM
            }
        }
    ]
    
    # Make a copy of datetime.now to restore later
    original_datetime_now = datetime.now
    
    # Mock datetime.now for testing late night transactions
    try:
        import unittest.mock
        from unittest.mock import patch
        
        # Run tests and collect results
        edge_results = []
        
        for test_case in edge_cases:
            # If there's a time override, apply it
            if '_override_hour' in test_case['data']:
                hour_override = test_case['data'].pop('_override_hour')
                
                # Mock datetime.now
                class MockDateTime:
                    @classmethod
                    def now(cls, *args, **kwargs):
                        real_now = original_datetime_now(*args, **kwargs)
                        return real_now.replace(hour=hour_override)
                
                # Apply the mock
                with patch('transactions.utils.datetime', MockDateTime):
                    is_fraud, score, reason = detect_fraud_with_ml(test_case["data"])
            else:
                is_fraud, score, reason = detect_fraud_with_ml(test_case["data"])
            
            # Store results
            edge_results.append({
                "Test Case": test_case["name"],
                "Amount": f"₹{test_case['data']['amount']:,}",
                "Currency": f"{test_case['data']['source_currency']} -> {test_case['data']['target_currency']}",
                "Bank": test_case["data"]["recipient_bank"],
                "Is Fraudulent": "✓" if is_fraud else "✗",
                "Risk Score": f"{score:.2f}",
                "Risk %": f"{score*100:.1f}%",
                "Risk Factors": reason if reason else "None detected"
            })
        
        # Display results in table format
        print("\nEdge Case Test Results:")
        print(tabulate(edge_results, headers="keys", tablefmt="grid"))
        
        # Check for high risk scores in edge cases
        high_risk_count = sum(1 for result in edge_results if float(result["Risk Score"]) > 0.7)
        print(f"\nDetected {high_risk_count} high-risk transactions out of {len(edge_results)} edge cases.")
        
        return edge_results
    except Exception as e:
        print(f"Error during edge case testing: {e}")
        return []

def run_basic_test():
    """Run the original basic test for backward compatibility"""
    print("\nRunning basic test...")
    
    # Sample transaction data
    test_transaction = {
        'amount': 50000,
        'transaction_type': 'send',
        'source_currency': 'INR',
        'target_currency': 'USD',
        'recipient_bank': 'Test Bank',
        'recipient_account': '123456789',
        'sender_account': '987654321'
    }
    
    # Run fraud detection
    is_fraud, score, reason = detect_fraud_with_ml(test_transaction)
    
    print(f"Basic test results:")
    print(f"- Is potentially fraudulent: {is_fraud}")
    print(f"- Risk score: {score:.4f} ({score*100:.2f}%)")
    print(f"- Reason: {reason}")
    
    # Test recipient analysis
    risk_percentage, risk_factors = analyze_recipient_transactions('123456789', 50000)
    
    print("\nRecipient analysis results:")
    print(f"- Risk percentage: {risk_percentage}%")
    print("- Risk factors:")
    for factor in risk_factors:
        print(f"  * {factor}")
    
    return is_fraud, score, reason, risk_percentage, risk_factors

if __name__ == "__main__":
    # Run basic test first
    run_basic_test()
    
    # Run comprehensive tests
    test_ml_model_with_varied_inputs()
    
    # Test edge cases
    try:
        test_edge_cases()
    except Exception as e:
        print(f"Edge case tests not supported in this environment: {e}") 