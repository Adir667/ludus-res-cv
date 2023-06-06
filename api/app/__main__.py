from flask import Flask, request, jsonify, make_response, url_for
from services.video_service import is_video_file_format, save_video, analyze_video
from config.api_config import validate_api_key
from services.response_service import save_responses, avg_response_time, best_response_time
import uuid
import os
from urllib.parse import quote

app = Flask(__name__)

@app.route('/response_time', methods=['POST'])
def response_time_post():

    # Get the API key from the request headers
    if not validate_api_key(request.headers.get('x-api-key')):
       return jsonify({'error': 'Invalid API key'}), 401
     
    # Check if the request contains a file
    if 'video' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    # Extract file name and extension
    video_file = request.files['video']
    original_filename = video_file.filename

    # Check if the file format is video
    if not is_video_file_format(original_filename):
        return jsonify({'error': 'Invalid file format. Only video files are allowed.'}), 400
    
    # give uuid to the video for later refrences
    unique_id = str(uuid.uuid4().hex)
    _, extension = os.path.splitext(original_filename)

    new_video_name = unique_id + extension

    # save the video before analysis
    saved_path = save_video(video_file, new_video_name)

    # video processing logic
    save_analyzed_video = bool(request.form.get('download'))

    responses_array = analyze_video(saved_path, save_analyzed_video)

    if len(responses_array) == 0:
        return jsonify({'message': 'Did not find any responses in your video.'}), 200

    # save array to db for the user
    user_id = request.form.get('user_id')
    save_responses(user_id, new_video_name, responses_array)

    response_data = {
        'Detected hits': responses_array,
        'avg': avg_response_time(responses_array),
        'best': best_response_time(responses_array)
    }

    if save_analyzed_video:
        response_data['download_link'] = url_for('download_file', filename=unique_id, _external=True)
    
    return jsonify(response_data), 200

@app.route('/download_file')
def download_file():
    # Path to the file
    requested_file = request.args.get('filename')
    file_path = './output_vids/' + requested_file + '_analyzed.mp4'

    try:
        # Read the file content
        with open(file_path, 'rb') as file:
            file_content = file.read()

        # Set the headers for the response
        headers = {
            'Content-Disposition': f'attachment; filename="{quote(requested_file + "_analyzed.mp4")}"'
        }

        # Create the response object
        response = make_response(file_content)
        response.headers = headers

        # Return the response
        return response

    except FileNotFoundError:
        return 'File not found', 404

if __name__ == '__main__':
    app.run()
