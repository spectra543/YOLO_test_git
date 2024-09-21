from collections import defaultdict
import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator, colors

# Load the segmentation model
model = YOLO("best17.pt")  # segmentation model
names = model.model.names  # Get class names from the model

# Capture the video file
cap = cv2.VideoCapture("t1.mp4")
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Store unique track IDs
track_history = set()

# Object counter variable
plastic_bottle = 0
Other_plastic_cup = 0
Other_plastic_wrapper = 0
Shoe = 0
Styrofoam_piece = 0
wood = 0

# Output video writer
out = cv2.VideoWriter("instance-segmentation.avi", cv2.VideoWriter_fourcc(*"MJPG"), fps, (w, h))

# Process the video frame by frame
while True:
    ret, im0 = cap.read()
    if not ret:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Perform predictions and tracking on the current frame
    results = model.predict(im0)
    results2 = model.track(im0, persist=True)
    annotator = Annotator(im0, line_width=2)

    # Check if there are any valid detections
    if results[0].boxes.id is not None and results[0].masks is not None:
        # Get the track IDs, classes, and masks
        track_ids = results[0].boxes.id.int().cpu().tolist()
        clss = results[0].boxes.cls.cpu().tolist()
        masks = results[0].masks.xy

        # Iterate over the detected objects and their respective masks
        for mask, cls, track_id in zip(masks, clss, track_ids):
            color = colors(int(cls), True)
            txt_color = annotator.get_txt_color(color)
            annotator.seg_bbox(mask=mask, mask_color=color, label=names[int(cls)], txt_color=txt_color)

# Garbage counter-------------------------------------------------------------------------------------------------------------------------------------------------------------------------
            if names[int(cls)] == "Plastic-bottle" and track_id not in track_history:
                plastic_bottle += 1  # Increment plastic bottle counter for new ID
                track_history.add(track_id)  # Add this track_id to history
                
            if names[int(cls)] == "Other plastic cup" and track_id not in track_history:
                Other_plastic_cup += 1  # Increment plastic bottle counter for new ID
                track_history.add(track_id)  # Add this track_id to history
                
            if names[int(cls)] == "Other plastic wrapper" and track_id not in track_history:
                Other_plastic_wrapper += 1  # Increment plastic bottle counter for new ID
                track_history.add(track_id)  # Add this track_id to history
            
            if names[int(cls)] == "Shoe" and track_id not in track_history:
                Shoe += 1  # Increment plastic bottle counter for new ID
                track_history.add(track_id)  # Add this track_id to history
                
            if names[int(cls)] == "Styrofoam piece" and track_id not in track_history:
                Styrofoam_piece += 1  # Increment plastic bottle counter for new ID
                track_history.add(track_id)  # Add this track_id to history
                
            if names[int(cls)] == "wood" and track_id not in track_history:
                wood += 1  # Increment plastic bottle counter for new ID
                track_history.add(track_id)  # Add this track_id to history
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

    # Write the processed frame to the output video
    out.write(im0)
    
    # Display the processed frame
    #cv2.imshow("instance-segmentation", im0)
    
    print(f"Total plastic bottles detected: {plastic_bottle}")
    print(f"Total Other plastic cup detected: {Other_plastic_cup}")
    print(f"Total Styrofoam piece detected: {Styrofoam_piece}")

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release video resources
out.release()
cap.release()
cv2.destroyAllWindows()

