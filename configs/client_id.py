import time
import hashlib

# when running in the real setting, flower loses the tracking of the ID of the client
def generate_client_id():
    """
    Generate a unique client ID based on the current time.
    """
    current_time = str(time.time()).encode()  # Get the current timestamp as a string
    client_id = hashlib.md5(current_time).hexdigest()  # Create a hash for uniqueness
    return client_id