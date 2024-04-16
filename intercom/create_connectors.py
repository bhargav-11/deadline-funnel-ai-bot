import cohere  
from cohere import CreateConnectorServiceAuth
from cohere.core import RequestOptions

co = cohere.Client('SamZ9jNB7MDCSMirvba3s9LRGkmVfep5olBV4iXV')

from cohere import AuthTokenType  # Importing the AuthTokenType

# Correct instantiation of CreateConnectorServiceAuth
service_auth = CreateConnectorServiceAuth(
    type="bearer",
    token='1234567890'
)

created_connector = co.connectors.create(
            name="intercom",
            url="https://e814-2405-201-2027-2063-9506-ff9f-f1c8-1735.ngrok-free.app/search",
            service_auth=service_auth,
            request_options=RequestOptions(timeout_in_seconds=55)
        )

print(created_connector)

# response = co.chat(  
# 	message="What is the chemical formula for glucose?",  
# 	connectors=[{"id": "intercom"}]  
# )

# print(response)