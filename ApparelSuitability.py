
import datetime 
import requests
import os
import cv2
import numpy as np
import torch
import time
import logging
from ultralytics import YOLO
import supervision as sv
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt, Confirm


#logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

#rich console for better readaiblity
console = Console()


#configs

CONFIG = {
    "weather": {
        "api_key": "input own api key",
        "base_url": "https://api.openweathermap.org/data/2.5/weather?",
        "cache_duration": 600
    },
    "model": {
        "path": "best.pt",  
        "confidence_threshold": 0.3
    },
    "clothing": {
        "default_warmth": {
            "clothes": 3,
            "dressshirt": 3,
            "suit": 5,
            "sweater": 7,
            "trenchcoat": 10,
            "tshirt": 3,
            "vest": 3,
            "hoodie": 8,
            "rain_jacket": 9,
            "puffer": 12
        }
    },
    "detection": {
        "interval": 15, 
        "frame_skip": 2
    },
    "output": {
        "directory": "output_images",
        "video": "output_video.mp4",
        "results_csv": "detection_results.csv"
    }
}
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


class WeatherAPI:
    def __init__(self):
        self.base_url = CONFIG["weather"]["base_url"]
        self.api_key = CONFIG["weather"]["api_key"]
        self.cache_duration = CONFIG["weather"]["cache_duration"]
        self.cache = None
        self.last_fetch = 0

    def kelvin_to_celsius(self, kelvin):
        return kelvin - 273.15

    def fetch_weather(self, city):
        current_time = time.time()
        if self.cache and (current_time - self.last_fetch) < self.cache_duration:
            console.print("[green]Using cached weather data.[/green]")
            return self.cache

        url = f"{self.base_url}appid={self.api_key}&q={city}"
        try:
            response = requests.get(url).json()
            if response.get("cod") != 200:
                raise ValueError(f"API Error: {response.get('message', 'Invalid city name')}")
            self.cache = response
            self.last_fetch = current_time
            return response
        except Exception as e:
            logger.error(f"Failed to fetch weather: {e}")
            raise

    def get_temperatures(self, city):
        data = self.fetch_weather(city)
        temp_celsius = self.kelvin_to_celsius(data["main"]["temp"])
        temp_feel_celsius = self.kelvin_to_celsius(data["main"]["feels_like"])
        return temp_celsius, temp_feel_celsius
    
    #put weather gathering into a class and improved readiblity wiht color
    