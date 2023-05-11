import cv2
import os
import torch
from PIL import Image
from pathlib import Path
import numpy as np

# Load YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='../models/best_trigger.pt')

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

def box_is_within_box(inner_box, outer_box):
    inner_box = inner_box[:4]  # extract only the first four elements
    outer_box = outer_box[:4]  # extract only the first four elements
    x1, y1, x2, y2 = map(int, inner_box)
    a1, b1, a2, b2 = map(int, outer_box)

    # Calculate the room allowance as 10% of the outer box dimensions
    # room_x = (a2 - a1) * 0.1
    # room_y = (b2 - b1) * 0.1

    # print(inner_box)
    # print(outer_box)
    
    # Adjust the inner box edges with the room allowance
    # x1 -= room_x
    # y1 -= room_y
    # x2 += room_x
    # y2 += room_y

    print('Checking if trigger is inside the paddle')
    print('Result:', (x1 > a1 and y1 > b1 and x2 < a2 and y2 < b2))
    return (x1 > a1 and y1 > b1 and x2 < a2 and y2 < b2)


def avg_response_time(hits_array):

    hits_np = np.array(hits_array)
    return np.mean(hits_np)

def best_response_time(hits_array):
    return min(hits_array)

# Open video file
video_path = Path('../input_vids/3.mp4').resolve()
if os.path.isfile(video_path):

    cap = cv2.VideoCapture(str(video_path))
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    output_path = Path('../analyzed_vids/output_res.mp4').resolve()
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (frame_width, frame_height))

    # set initial values for counters
    frame_counter = 1
    hit_frame = 0
    prev_position = None
    trigger_frame = 0
    frames_gap = 0
    movement_threshold_for_hit = fps/2
    distance = 0
    hits_counter = 0
    hits_array = []
    distance_array = []
    trigger_on = False
    waiting_for_hit = False

    # Loop through video frames
    while cap.isOpened() and frame_counter <= 400:
        ret, frame = cap.read()
        if not ret:
            break

        # Detect objects in frame
        img = Image.fromarray(frame[...,::-1])  # convert frame from BGR to RGB
        results = model(img, size=640)  # perform object detection

        print(f"Frame: {frame_counter}/{total_frames}")
        paddle_detected = False

        for box, label, conf in zip(results.xyxy[0], results.names[0], results.pred[0]):
            x1, y1, x2, y2 = map(int, box[:4])
            center_x = int((x1 + x2) / 2)
            center_y = int((y1 + y2) / 2)

            #confidence
            conf_array = conf.detach().cpu().numpy()
            confidence = conf_array[-2]

            #print the position and the confidence for the detected object
            position_confidence = f'pos: {center_x}, {center_y}. conf: {confidence:.2f}'
            cv2.putText(frame, position_confidence, (x1, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            #paddle
            if label == 'p' and confidence > 0.8:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                paddle_detected = True
                paddle_box = box

                # Draw a small red dot at the center of the paddle's bounding box
                cv2.circle(frame, (center_x, center_y), 6, (0, 0, 255), -1)
                print('Detected paddle: ', position_confidence)
                
                if prev_position is not None:
                    current_position = (center_x, center_y)
                    distance = abs(current_position[0] - prev_position[0])
                    print('distance difference for paddle:' ,distance)
                    #add the distance travel to the array
                    distance_array.append(distance)

                    if distance > movement_threshold_for_hit:
                        cv2.putText(frame, f"Movement: {distance} px!", (x1, y2+25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        if waiting_for_hit:
                            hit_frame = frame_counter
                            waiting_for_hit = False
                
                #update position            
                prev_position = (center_x, center_y)
            
            #trigger
            if label == 'a' and not trigger_on: #and confidence > 0.3
                if (box_is_within_box(box, paddle_box)):
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.circle(frame, (center_x, center_y), 6, (0, 255, 0), 1)

                    print('Detected trigger: ', position_confidence)
                    trigger_frame = frame_counter
                    trigger_on = True
                    waiting_for_hit = True

        if (trigger_on and paddle_detected == False):
            print('The trigger was on and now i dont see the paddle anymore, I assume it was hit (hard)')
            hit_frame = frame_counter
            trigger_on = False

        if (hit_frame > trigger_frame and waiting_for_hit):
            hits_counter += 1
            print('hit nr', hits_counter)
            frames_gap = (hit_frame - trigger_frame) 
            hits_array.append(frames_gap/fps)
            waiting_for_hit = False

        # printing and debugging
        print('paddle found?:' ,paddle_detected)
        print('trigger on?', trigger_on)
        print('T frame:' ,trigger_frame)
        print('H frame:' ,hit_frame)
        print('---------------------')


        #write to the screen
        fps_indicator = f'FPS: {fps:.2f}'
        cv2.putText(frame, fps_indicator, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        timestamp = f'Time: {frame_counter/fps:.3f}s'
        cv2.putText(frame, timestamp, (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        frame_indicator = f'Frame nr: {frame_counter}'
        cv2.putText(frame, frame_indicator, (50, 300), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        trigger_frame_indicator = f'T frame: {trigger_frame}'
        cv2.putText(frame, trigger_frame_indicator, (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        hit_frame_indicator = f'H frame: {hit_frame}'
        cv2.putText(frame, hit_frame_indicator, (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        response_time_indicator = f'Response: {frames_gap/fps:.3f}s'
        cv2.putText(frame, response_time_indicator, (50, 600), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        total_hits_counter = f'Total H: {hits_counter}'
        cv2.putText(frame, total_hits_counter, (50, 700), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 5)
        
        # Write frame to output video
        out.write(frame)
        
        frame_counter += 1

    #end of while loop

    # Release video capture and writer
    cap.release()
    out.release()

    print(f'Successfully saved output video to {output_path}')

    # print summary
    print("summary")
    print('Total hits:' ,hits_counter)
    print(hits_array)
    print("avg hits")
    avg_time = avg_response_time(hits_array)
    print(f"The average response time was {avg_time:.2f} seconds")
    best_time = best_response_time(hits_array)
    print(f"The best response time was {best_time:.2f} seconds")

    print('paddle position tracker array', distance_array)

else:
    print("Could not find the input file")




