from config.api_config import get_connection_string
from flask import jsonify
from pymongo import MongoClient
from time import time

from time import time

def save_responses_to_db(user_id, video_id, responses_array):
    timestamp = int(time())

    client = MongoClient(get_connection_string())
    db = client.ludus

    data = []

    for item in responses_array:
        data.append({
            "user_id": int(user_id),
            "value": float(item),
            "video_id": video_id,
            "timestamp": timestamp,
        })

    try:
        # Insert the dummy data into the collection
        result = db.field.insert_many(data)
        client.close()  # Close the MongoDB client connection
        return jsonify({"message": "Data saved successfully", "ids": result.inserted_ids}), 200
    except Exception as e:
        client.close()  # Close the MongoDB client connection
        return jsonify({"error": str(e)}), 500

#1