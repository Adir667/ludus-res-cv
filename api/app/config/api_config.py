API_KEY = 'mykey'
CONNECTION_STRING = 'mongodb+srv://user:pass@ludus.shsntx0.mongodb.net/?retryWrites=true&w=majority'

def get_connection_string():
    return CONNECTION_STRING

def validate_api_key(given_key):
    return given_key == API_KEY