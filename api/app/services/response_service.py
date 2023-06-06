from repositories.response_repository import save_responses_to_db
import numpy as np

def save_responses(user_id, video_id, responses_array):
    # some test about the responses being valid, user being valid
    if responses_array is not None:
        save_responses_to_db(user_id, video_id, responses_array)

def avg_response_time(responses_array):

    hits_np = np.array(responses_array)
    return round(np.mean(hits_np), 3)

def best_response_time(responses_array):
    return min(responses_array)