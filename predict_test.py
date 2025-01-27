#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:8885/predict'

customer_id = 'xyz-123'
customer = {'status': 'a11',
 'duration': 6,
 'credit_history': 'a34',
 'purpose': 'a43',
 'credit_amount': 1169,
 'savings': 'a65',
 'employment': 'a75',
 'installment_rate': 4,
 'personal_status': 'a93',
 'other_debtors': 'a101',
 'residence_duration': 4,
 'property': 'a121',
 'age': 67,
 'other_installments': 'a143',
 'housing': 'a152',
 'existing_credits': 2,
 'job': 'a173',
 'number_dependents': 1,
 'telephone': 'a192',
 'foreign_worker': 'a201'
 }

response = requests.post(url, json=customer).json()
print(response)

if response['target'] == 0:
    print(f"The credit score for {customer_id} is good.")
else:
    print(f"The credit score for the {customer_id} is bad.")