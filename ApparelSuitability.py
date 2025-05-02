
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

class ClothingDetector:
    def __init__(self, capture_index, value_map):
        self.capture_index = capture_index
        self.value_map = value_map
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        console.print(f"[blue]Using device: {self.device}[/blue]")

        self.model = self.load_model()
        self.class_names = self.model.names
        console.print(f"[blue]Model class names: {list(self.class_names.values())}[/blue]")
        console.print(f"[blue]Expected clothing classes: {list(self.value_map.keys())}[/blue]")
        console.print("[yellow]Note: YOLO11n.pt uses COCO classes (e.g., 'person', 'tie'). For clothing detection, train a custom model on a dataset like ModaNet.[/yellow]")

        self.confidence_threshold = CONFIG["model"]["confidence_threshold"]

        self.box_annotator = sv.BoxAnnotator(
            color=sv.Color.WHITE,
            thickness=3,
            color_lookup=sv.ColorLookup.CLASS
        )

        self.output_dir = CONFIG["output"]["directory"]
        os.makedirs(self.output_dir, exist_ok=True)
        self.detected_values = []
        self.results = []

    def load_model(self):
        try:
            model = YOLO(CONFIG["model"]["path"])
            model.fuse()
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise

    def predict(self, frame):
        try:
            results = self.model(frame, conf=self.confidence_threshold)
            console.print(f"[yellow]Raw detections: {len(results[0].boxes)} objects detected[/yellow]")
            return results
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return []

    def plot_bboxes(self, results, frame, frame_count):
        xyxys, confidences, class_ids = [], [], []

        for result in results:
            boxes = result.boxes.cpu().numpy()
            for i in range(len(boxes.xyxy)):
                class_id = int(boxes.cls[i])
                conf = boxes.conf[i]
                xyxy = boxes.xyxy[i]
                class_name = self.class_names[class_id]
                console.print(f"[cyan]Detection[/cyan]: {class_name}, Confidence: {conf:.2f}")
                xyxys.append(xyxy)
                confidences.append(conf)
                class_ids.append(class_id)
                if class_name in self.value_map:
                    self.detected_values.append(self.value_map[class_name])
                    self.results.append({
                        "frame": frame_count,
                        "class": class_name,
                        "confidence": conf,
                        "warmth": self.value_map[class_name],
                        "timestamp": datetime.datetime.now().isoformat()
                    })
                else:
                    console.print(f"[yellow]Class '{class_name}' not in value_map. Ignored for warmth calculation.[/yellow]")

        if xyxys:
            detections = sv.Detections(
                xyxy=np.array(xyxys),
                confidence=np.array(confidences),
                class_id=np.array(class_ids),
            )
            frame = self.box_annotator.annotate(scene=frame, detections=detections)
            for i, detection in enumerate(detections.xyxy):
                x1, y1, x2, y2 = map(int, detection)
                label = f"{self.class_names[int(class_ids[i])]} {confidences[i]:.2f}"
                cv2.putText(
                    frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2
                )
        else:
            console.print("[yellow]No detections in this frame[/yellow]")

        return frame

    def save_results(self):
        df = pd.DataFrame(self.results)
        if not df.empty:
            df.to_csv(os.path.join(self.output_dir, CONFIG["output"]["results_csv"]), index=False)
            console.print(f"[green]Detection results saved to {CONFIG['output']['results_csv']}[/green]")

    def run(self):
        cap = cv2.VideoCapture(self.capture_index)
        if not cap.isOpened():
            logger.error("Error: Could not open video capture.")
            raise ValueError("Invalid video capture source")
        console.print("[green]Webcam opened successfully[/green]")

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(
            os.path.join(self.output_dir, CONFIG["output"]["video"]),
            fourcc,
            30.0,
            (1280, 720)
        )

        paused = False
        last_detection_time = time.time()
        detection_interval = CONFIG["detection"]["interval"]
        frame_skip = CONFIG["detection"]["frame_skip"]
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                console.print("[red]Error reading frame[/red]")
                break

            frame_count += 1
            if frame_count % frame_skip != 0:
                continue

            current_time = time.time()
            if not paused and (current_time - last_detection_time >= detection_interval):
                results = self.predict(frame)
                frame = self.plot_bboxes(results, frame, frame_count)
                last_detection_time = current_time

            fps = 1 / max(np.round(current_time - last_detection_time, 2), 0.01)
            cv2.putText(frame, f"FPS: {int(fps)}", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)

            cv2.imshow("Object Detection", frame)
            out.write(frame)

            key = cv2.waitKey(1)
            if key & 0xFF == 27:
                console.print("[yellow]Escape key pressed. Exiting...[/yellow]")
                break
            elif key & 0xFF == ord("p"):
                paused = not paused
                console.print(f"[blue]Detection {'paused' if paused else 'resumed'}[/blue]")

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        self.save_results()
        return self.detected_values
    
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
