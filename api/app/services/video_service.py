import os
from pathlib import Path
import cv2
from PIL import Image
from pathlib import Path
import torch

save_input_directory = './input_vids/'
save_output_directory = './output_vids/'

def is_video_file_format(filename):
    # Perform validation here to check if the file is in video format
    # This is just a simple example checking the file extension
    allowed_extensions = ['.mp4', '.avi', '.mov']
    return (filename.lower().endswith(ext) for ext in allowed_extensions)

def save_video(video, new_video_name):

    # Save the video file with the unique filename
    save_path = os.path.join(save_input_directory, new_video_name)
    video.save(save_path)

    return save_path

def analyze_video(video_path, save):

    # Load YOLOv5 model
    model = torch.hub.load('ultralytics/yolov5', 'custom', path='./cv_models/best_trigger.pt')

    cap = cv2.VideoCapture(str(video_path))

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #set_video_writer_output()
    if save:
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Create output video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_path_obj = Path(video_path)  # Convert video_path to Path object
        output_filename = video_path_obj.stem + "_analyzed.mp4"  # Generate output filename
        output_dir = video_path_obj.parent.parent / "output_vids"  # Output directory path
        output_path = output_dir / output_filename  # Create output path
        output_dir.mkdir(parents=True, exist_ok=True)  # Create the output directory if it doesn't exist
        out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # set initial values for counters
    frame_counter = 1
    hit_frame = 0
    prev_position = None
    trigger_frame = 0
    frames_gap = 0
    movement_threshold_for_hit = 12
    distance = 0
    hits_counter = 0
    hits_array = []
    trigger_on = False
    looking_for_hit = False
    current_position = None
    max_waiting_frames = 45
    current_response = 0.000
    paddle_min_confidence = 0.8
    trigger_min_confidence = 0.6
    min_frames_gap = 8

    # Loop through video frames
    while cap.isOpened() and frame_counter <= 200:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in frame
        img = Image.fromarray(frame[...,::-1])  # convert frame from BGR to RGB
        results = model(img, size=640)  # perform object detection

        print(f"Frame: {frame_counter}/{total_frames}")
        paddle_detected = False
        
        boxes, labels, confidences = results.xyxy[0], results.names[0], results.pred[0]
        detections = [(box, label, conf) for box, label, conf in zip(boxes, labels, confidences)]
        for box, label, conf in detections:
            x1, y1, x2, y2 = map(int, box[:4])
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2

            #position and confidence
            conf_array = conf.detach().cpu().numpy()
            confidence = conf_array[-2]
            position_confidence = f'pos: {center_x}, {center_y}. conf: {confidence:.2f}'

            #paddle
            if label == 'p' and confidence > paddle_min_confidence:
                if save:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (128, 0, 128), 2)
                    cv2.putText(frame, position_confidence, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (128, 0, 128), 2)
                    # Draw a small red dot at the center of the paddle's bounding box
                    cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
                paddle_detected = True
                paddle_box = box
                
                if prev_position is not None:
                    current_position = (center_x, center_y)
                    distance = abs(current_position[0] - prev_position[0])
                    print('distance difference for paddle:' ,distance)

                    if distance > movement_threshold_for_hit:
                        if save:
                            cv2.putText(frame, f"Movement: {distance} px!", (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        if looking_for_hit:
                            hit_frame = frame_counter
                
                #update position            
                prev_position = (center_x, center_y)
            
            #trigger
            if label == 'a' and not trigger_on and confidence > trigger_min_confidence:
                if trigger_is_within_paddle(center_x, center_y, paddle_box):
                    if save:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                        cv2.circle(frame, (center_x, center_y), 6, (0, 255, 0), 1)

                    print('Detected trigger: ', position_confidence)
                    trigger_frame = frame_counter
                    trigger_on = True
                    looking_for_hit = True

        #hit_analyzer(frame_counter, trigger_frame, max_waiting_frames, trigger_on, paddle_detected, looking_for_hit, min_frames_gap)
        # max wait time for hit            
        if frame_counter - trigger_frame > max_waiting_frames:
            trigger_on = False

        if (trigger_on and paddle_detected == False):
            print('The trigger was on and now i dont see the paddle anymore, I assume it was hit (hard)')
            hit_frame = frame_counter
            trigger_on = False

        if (hit_frame > trigger_frame and looking_for_hit):
            frames_gap = (hit_frame - trigger_frame)
            if frames_gap > min_frames_gap: 
                hits_counter += 1
                print('hit nr', hits_counter)
                current_response = round(frames_gap / fps, 2)
                hits_array.append(current_response)
                looking_for_hit = False

        # printing and debuging
        debug_print(paddle_detected, trigger_on, trigger_frame, hit_frame)

        if save:
            write_to_screen(frame, fps, frame_counter, trigger_frame, hit_frame, current_response, out)

        frame_counter += 1

    #end of while loop

    # Release video capture and writer
    cap.release()

    # print summary
    print_summary(hits_counter, hits_array)

    return hits_array

def trigger_is_within_paddle(center_x, center_y, outer_box):

    outer_box = outer_box[:4]  # extract only the first four elements
    a1, b1, a2, b2 = map(int, outer_box)

    print(center_x, center_y)
    print(outer_box)
    
    print('Checking if trigger is inside the paddle')
    print('Result:', (a1 < center_x < a2) and (b1 < center_y < b2))
    return (a1 < center_x < a2) and (b1 < center_y < b2)

def write_to_screen(frame, fps, frame_counter, trigger_frame, hit_frame, current_response, out):
    #write to the screen
    fps_indicator = f'FPS: {fps:.2f}'
    cv2.putText(frame, fps_indicator, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    timestamp = f'Time: {frame_counter/fps:.3f}s'
    cv2.putText(frame, timestamp, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    frame_indicator = f'Frame nr: {frame_counter}'
    cv2.putText(frame, frame_indicator, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    trigger_frame_indicator = f'T frame: {trigger_frame}'
    cv2.putText(frame, trigger_frame_indicator, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    if trigger_frame > hit_frame:
        hit_frame_indicator = 'H frame: waiting'
    else:
        hit_frame_indicator = f'H frame: {hit_frame}'
    cv2.putText(frame, hit_frame_indicator, (50, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    response_time_indicator = f'Response: {current_response}s'
    cv2.putText(frame, response_time_indicator, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    
    #Write frame to output video
    out.write(frame)

def debug_print(paddle_detected, trigger_on, trigger_frame, hit_frame):

    print('paddle found?:' ,paddle_detected)
    print('trigger on?', trigger_on)
    print('T frame:' , trigger_frame)
    print('H frame:' , hit_frame)
    print('---------------------')

def print_summary(hits_counter, hits_array):

    print("summary")
    print('Total hits:' ,hits_counter)
    print(hits_array)

#hit_analyzer(frame_counter, trigger_frame, max_waiting_frames, trigger_on, paddle_detected, looking_for_hit, min_frames_gap)
#1