import cohere  
co = cohere.Client('SamZ9jNB7MDCSMirvba3s9LRGkmVfep5olBV4iXV')

response = co.chat(  
	message="Please use the connector and reply what is evergreen campaign?",  
	connectors=[{"id": "intercom-t7tbxs"}]  
)

print(response)