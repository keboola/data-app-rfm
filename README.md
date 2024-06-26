# RFM Data App
A plug-and-play application that creates an RFM analysis based on industry standard definitions of segments and views. 
The RFM score calculation is customizable by the user. 
The app shows AI powered insights into your data and suggestions for possible actions.

Requirements:
- Update the bucket secret in the data with the name of the bucket where your data is located
- In your bucket, there should be a table named 'order'
- The table 'order' should have at least 4 columns: 
	- customer_id: Id of the customer
	- created_at: Timestamp of the event in the format 2022-10-11T13:49:55-04:00 
	- total_price_usd: Total price in USD
	- financial_status: paid/pending

The app is built to support the Shopify component. 
If the data should come from different component, either:
- Transform the table to have the requirements
- Add code for handling the new component in the load_data() function

Secrets used:
kbc_url - url of Keboola instance  
kbc_token - Keboola API token (one that can create buckets and tables)  
openai_token - OpenAI API token (with access to the model you would like to use), currently set up with gpt-3.5-turbo  
read_bucket - Keboola bucket from which to take the data  
write_bucket - Keboola bucket to which write the data

| Version |    Date    |       Description       |
|---------|:----------:|:-----------------------:|
| 1.0     | 2024-06-20 |  A data app for RFM analysis
