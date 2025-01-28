
import datetime 
import requests
import os
import cv2
import numpy as np
import torch
from time import time
from ultralytics import YOLO
import supervision as sv

class ObjectDetection:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device) 
        # select CPU or GPU 

        self.model = self.load_model# calls the pretraiend pytorch model 
        self.CLASS_NAMES_DICT = self.model.model.names # Get class names from pretrained model
        
        self.box_annotator = sv.BoxAnnotator(
            color=sv.Color.WHITE,
            thickness=3,
            color_lookup=sv.ColorLookup.CLASS
        )
        #create bounding box 

        self.output_dir = 'output_images'
        os.makedirs(self.output_dir, exist_ok=True) #Create directory if directory doesn't exist alr
        self.img_counter = 0 
        self.value_map = {
            "clothes": 3,"dressshirt":3,"suit":5,"sweater":7,"trenchcoat":10,"tshirt":3,"vest":3  # Assigns the value 10 to trenchcoat whice corresponds to how many degrees a piece of clothing provides
        }
        self.detected_values = []# List to store the numerical value each piece of clothing provides
        

    def load_model(self):
        # model = YOLO("yolo11n.pt")
        model = YOLO("best.pt") #custom pretrained model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame) #run the model on webcam input frame
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        # Extract detections for the specified class
        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes.xyxy)):
                class_id = boxes.cls[i]
                conf = boxes.conf[i]
                xyxy = boxes.xyxy[i]
                if self.CLASS_NAMES_DICT[class_id] in self.value_map:  # Check against value_map
                    xyxys.append(xyxy)
                    confidences.append(conf)
                    class_ids.append(class_id)

                    #Adds to warmth currently provided 
                    self.detected_values.append(self.value_map[self.CLASS_NAMES_DICT[class_id]])

        # Setup detections for visualization
        if xyxys:  # Only proceed if we have detections
            detections = sv.Detections(
                xyxy=np.array(xyxys),
                confidence=np.array(confidences),
                class_id=np.array(class_ids),
            )

            # Annotate and display frame 
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            for i, detection in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, detection)
                label = f"{self.CLASS_NAMES_DICT[class_ids[i]]} {confidences[i]:0.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return frame
    
    def __call__(self):
        #Uses webcam as video capture and set dimensions
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        paused = False  # Variable to track the pause state
        last_detection_time = time.time()  # Initialize detection timer
        detection_interval = 15  # 15 seconds

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error reading frame")  # Ensure there is a video input
                break

            current_time = time.time()

            # Check if enough time has passed for the next detection
            if not paused and (current_time - last_detection_time >= detection_interval):
                results = self.predict(frame)  # Perform detection
                frame = self.plot_bboxes(results, frame)
                last_detection_time = current_time  # Update the last detection time

            fps = 1 / np.round(time.time() - current_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow('Clothing detection', frame)

            # Check for key presses
            key = cv2.waitKey(1)  # Short wait for key press
            if key & 0xFF == 27:  # Exit on ESC key
                print("Escape hit")
                break
            elif key & 0xFF == ord('n'):  # Toggle pause on 'n' key
                paused = not paused
                if paused:
                    print("Detection paused. Press 'n' to resume.")
                else:
                    print("Detection resumed.")


        cap.release()
        cv2.destroyAllWindows()
        return self.detected_values



def kelvinToCelsius(kelvin):
    return kelvin-273.15
#weather collection 
BASE_URL ="https://api.openweathermap.org//data/2.5/weather?"
API_KEY="input own api key from openweather map"
CITY = input("What city: ")
url = BASE_URL+"appid="+API_KEY+"&q="+CITY
response = requests.get(url).json()
temp_kelvin = response['main']['temp']
temp_celsius = kelvinToCelsius(temp_kelvin)
temp_feelKel = response['main']['feels_like']
temp_feelCel = kelvinToCelsius(temp_feelKel)
print(f"Actual temp: {temp_celsius:.2f}")  
print(f"Feels like: {temp_feelCel:.2f}")

while True:
    cold_response = input("Are you prone to feeling cold? (yes/no): ")
    if cold_response == "yes":
        tempCompare = temp_feelCel  # Set temperature for prone to cold
        break  # Break the loop if valid input is received
    elif cold_response == "no":
        tempCompare = temp_celsius  # Set temperature for not prone to cold
        break  # Break the loop if valid input is received
    else:
        print("Invalid response. Please answer with 'yes' or 'no'.")               


# Loop until valid input is received for clothing_response
while True:
    clothing_response = input("Are you wearing at least a t-shirt or a tank top? (yes/no): ").strip().lower()
    
    if clothing_response == "yes":
        print("You are ready to proceed!")
        capture_index = 0  # Change to your camera index or video file path
        detector = ObjectDetection(capture_index) #Create instance of ObjectDetection class 
        detected_values = detector()
        
        warmth = sum(detected_values)  # Total warmth from detected clothing
        clothing_value = detector.value_map
        
        # Determine how much warmth is needed based on tempCompare
        if tempCompare < 0:  # Cold environment
            warmth_needed = tempCompare + (warmth / 3)  # Calculate additional warmth needed
            if warmth_needed > 0:
                print(f"You need at least: {warmth_needed:.2f} Celsius worth of clothes!")
                # Recommendations based on warmth needed
                while warmth_needed > 0:
                    clothing_recommendations = detector.value_map
                    recommended_item = None
                    
                    for item, value in clothing_recommendations.items():
                        if warmth_needed <= value:  # Check if the clothing item can cover the warmth needed
                            recommended_item = item
                            break
                    
                    if recommended_item:
                        print(f"We recommend you put on a {recommended_item} to stay warm.")
                        warmth_needed -= clothing_recommendations[recommended_item]  # Decrease warmth needed
                    else:
                        print("No more recommendations available.")
                        break
            else:
                print("You are adequately dressed for the cold temperature.")
        
        else:  # For zero or positive temperatures
            if tempCompare <= 10:
                warmth_needed = abs(tempCompare) - (warmth / 3)  # For temperatures 10 and below
            elif 10 < tempCompare < 22:
                warmth_needed = tempCompare - (7+warmth / 3)  # For temperatures between 10 and 22
            else:  # For temperatures 22 and above
                warmth_needed = (tempCompare - 2) - (17+warmth / 3)  # Slight reduction for comfort
            
            if warmth > tempCompare:
                print("You have too much clothing on for the current temperature! You may want to remove some layers.")
            elif warmth < tempCompare:
                print(f"You need at least: {warmth_needed:.2f} Celsius worth of clothes!")
                
                # Recommendations based on warmth needed
                while warmth_needed > 0:
                    clothing_recommendations = detector.value_map
                    recommended_item = None
                    
                    for item, value in clothing_recommendations.items():
                        if warmth_needed <= value:  # Check if the clothing item can cover the warmth needed
                            recommended_item = item
                            break
                    
                    if recommended_item:
                        print(f"We recommend you put on a {recommended_item} to stay warm.")
                        warmth_needed -= clothing_recommendations[recommended_item]  # Decrease warmth needed
                    else:
                        print("No more recommendations available.")
                        break
            else:
                print("You are adequately dressed for the temperature.")
        
        break  # Exit the loop after processing the clothing response

    elif clothing_response == "no":
        print("Please put on at least a t-shirt or tank top before proceeding.")
        break  # Exit the program 

    else:
        print("Invalid response. Please answer with 'yes' or 'no'.")