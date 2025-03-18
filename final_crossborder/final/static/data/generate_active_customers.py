import pandas as pd
import random
import string

# Generate sample active customers data
def generate_active_customers():
    data = []
    
    # Sample customer names (these would be actual users in your system)
    customer_names = [
        "Alice Johnson", "Bob Smith", "Charlie Brown", "Diana Prince", 
        "Edward Norton", "Frank Castle", "Grace Kelly", "Henry Ford",
        "Irene Adler", "Jack Ryan", "Kate Bishop", "Leo Fitz",
        "Maria Hill", "Nick Fury", "Olivia Pope", "Peter Parker",
        "Quinn Fabray", "Rachel Green", "Steve Rogers", "Tony Stark"
    ]
    
    # Generate customer data
    for i, name in enumerate(customer_names):
        first_name, last_name = name.split(' ')
        username = f"{first_name.lower()}{last_name[0].lower()}"
        
        account_number = ''.join(random.choices(string.digits, k=12))
        bank_name = random.choice([
            "State Bank of India", "HDFC Bank", "ICICI Bank", "Axis Bank",
            "Punjab National Bank", "Bank of Baroda", "Canara Bank", "Union Bank of India",
            "Bank of India", "Indian Bank", "Central Bank of India", "IndusInd Bank",
            "Yes Bank", "Kotak Mahindra Bank", "Federal Bank", "IDBI Bank"
        ])
        ifsc = ''.join(random.choices(string.ascii_uppercase, k=4)) + '0' + ''.join(random.choices(string.digits, k=6))
        balance = round(random.uniform(1000, 100000), 2)
        
        data.append({
            "username": username,
            "full_name": name,
            "account_number": account_number,
            "bank_name": bank_name,
            "ifsc_code": ifsc,
            "balance": balance,
            "is_active": True,
            "date_joined": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    
    return pd.DataFrame(data)

# Generate and save the data
customers_df = generate_active_customers()
customers_df.to_excel("static/data/active_customers.xlsx", index=False)
print(f"Generated active customers database with {len(customers_df)} entries.") 