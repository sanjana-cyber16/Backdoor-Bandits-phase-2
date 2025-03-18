import pandas as pd
import random
import string

# Generate sample fraudulent data
def generate_fraudster_data(count=50):
    data = []
    
    # Common fraudster names
    suspicious_names = [
        "John Smith", "Jane Doe", "Michael Johnson", "Robert Williams", 
        "David Brown", "Richard Davis", "Thomas Wilson", "Charles Miller",
        "Daniel Moore", "Matthew Taylor", "Anthony Anderson", "Donald Thomas",
        "Mark Jackson", "Paul White", "Steven Harris", "Andrew Martin",
        "Kenneth Thompson", "George Garcia", "Edward Martinez", "Brian Robinson",
        "Ronald Clark", "Anthony Rodriguez", "Kevin Lewis", "Jason Lee",
        "Jeff Walker", "Scott Hall", "Eric Allen", "Mary Young", "Lisa King",
        "Sarah Wright", "Karen Scott", "Nancy Green", "Jessica Baker", "Susan Nelson",
        "Margaret Hill", "Betty Adams", "Dorothy Campbell", "Sandra Mitchell",
        "Ashley Roberts", "Kimberly Carter", "Emily Phillips", "Donna Evans",
        "Michelle Turner", "Carol Parker", "Amanda Edwards", "Melissa Collins",
        "Deborah Stewart", "Stephanie Sanchez", "Rebecca Morris", "Sharon Rogers"
    ]
    
    # Generate random account numbers and bank names
    for i in range(count):
        account_number = ''.join(random.choices(string.digits, k=random.randint(10, 16)))
        name = random.choice(suspicious_names)
        bank_name = random.choice([
            "Global Bank", "United Finance", "Secure Trust", "Liberty Banking",
            "Premier Financial", "National Credit", "Apex Banking", "Metro Bank",
            "City Finance", "Royal Trust", "Capital One", "First Direct",
            "Universal Bank", "Omega Financial", "Delta Credit", "Alpha Banking"
        ])
        ifsc = ''.join(random.choices(string.ascii_uppercase, k=4)) + '0' + ''.join(random.choices(string.digits + string.ascii_uppercase, k=6))
        reason = random.choice([
            "Reported for fraud", "Suspicious activity", "Multiple complaints",
            "Unauthorized transactions", "Scam reports", "Identity theft",
            "Money laundering", "Phishing attempts"
        ])
        risk_level = random.choice(["High", "Medium", "Low"])
        
        data.append({
            "account_number": account_number,
            "recipient_name": name,
            "bank_name": bank_name,
            "ifsc_code": ifsc,
            "reason": reason,
            "risk_level": risk_level,
            "date_added": pd.Timestamp.now().strftime("%Y-%m-%d")
        })
    
    return pd.DataFrame(data)

# Generate and save the data
fraudster_df = generate_fraudster_data()
fraudster_df.to_excel("static/data/fraudster_database.xlsx", index=False)
print(f"Generated fraudster database with {len(fraudster_df)} entries.") 