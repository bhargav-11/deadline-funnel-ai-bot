import cohere  
co = cohere.Client('YOUR_COHERE_KEY')

response = co.chat(  
	message="Please use the connector and reply what is evergreen campaign?",  
	connectors=[{"id": "intercom-t7tbxs"}]  
)

print(response)