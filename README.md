# 🧥 Apparel Suitability

<div align="center">


**Smart clothing recommendations based on real-time weather and what you're wearing**

[![Python 3.x](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)
[![OpenCV](https://img.shields.io/badge/opencv-4.x-green.svg)](https://opencv.org/)
[![PyTorch](https://img.shields.io/badge/pytorch-2.x-orange.svg)](https://pytorch.org/)
[![YOLO](https://img.shields.io/badge/YOLO-v12-yellow.svg)](https://github.com/ultralytics/ultralytics)

</div>

## ✨ Features

- 📷 **Real-time clothing detection** using a custom YOLO model
- 🌤️ **Weather data integration** via OpenWeatherMap API
- 🌡️ **Personalized temperature adjustments** based on your cold sensitivity
- 👕 **Smart clothing recommendations** based on current attire and conditions

## 📋 Requirements

- Python 3.x
- OpenCV
- PyTorch
- Ultralytics YOLO
- Supervision
- NumPy
- Requests

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/apparel-suitability.git
cd apparel-suitability

# Install dependencies
pip install opencv-python torch ultralytics supervision numpy requests rich pandas
```

### 2. Configuration

- Obtain an API key from [OpenWeatherMap](https://openweathermap.org/)
- Replace the placeholder in `ApparelSuitability.py`:
  ```python
  API_KEY = "your_api_key_here"
  ```
- Download the custom YOLO model file (`best.pt`) and place it in the project directory
- Ensure a webcam is connected (default capture_index=0). Adjust if needed:
detector = ClothingDetector(capture_index=0, value_map=value_map)

### 3. Running the Application

```bash
python ApparelSuitability.py
```

## 🔍 How It Works


1. **Weather Data Collection**
   - Enter your city name
   - Retrieves current temperature and "feels like" data, cached for 10 minutes.

2. **User Preferences**
   - Indicate if you’re prone to feeling cold.
   - Confirm you’re wearing a t-shirt or tank top.
   - Customize warmth values for clothing items.

3. **Clothing Detection**
   - AI model detects clothing items in real-time.
   - Persists bounding boxes for 3 seconds for smooth visualization.
   - Assigns warmth values to detected items.

4. **Temperature Analysis**
   - Compares clothing warmth to required warmth based on temperature, season, and activity level.
   - Adjusts for cold sensitivity.
5. **Smart Recommendations**
   - Suggests specific clothing items to add or remove.
   - Displays recommendations in a formatted table.

## 🧠 The AI Model

The project uses a custom-trained YOLO (You Only Look Once) model designed to detect various clothing items:

| Clothing Item | Warmth Value |
|---------------|--------------|
| T-shirt       | 3            |
| Dress shirt   | 3            |
| Vest          | 3            |
| Suit          | 5            |
| Sweater       | 7            |
| Hoodie        | 8            |
| Rain Jacket   | 9            |
| Trench coat   | 10           |
| Puffer        | 12           |

## ⌨️ Controls

- **ESC**: Exit the application
- **P**: Pause/resume detection
- Warmth values can be customized during runtime.

## 📁 Output

- Images: Saved to output_images/ directory.
- Video: Annotated video saved as output_images/output_video.mp4.
- CSV: Detection results saved to output_images/detection_results.csv.

## 📝 Example Output

```
Enter your city: New York
[green]Actual temp: 15.25°C[/green]
[green]Feels like: 13.78°C[/green]
Are you prone to feeling cold? [y/N]: yes
Are you wearing at least a t-shirt or tank top? [Y/n]: yes
[bold cyan]Customize warmth values (press Enter for default)[/bold cyan]
Enter warmth value for hoodie [8]: 9
...
[blue]Using device: cuda[/blue]
[green]Webcam opened successfully[/green]
[yellow]Raw detections: 2 objects detected[/yellow]

Clothing Recommendation
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Status        ┃ Details               ┃ Action        ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ Underdressed  │ Need 2.5 more warmth  │ Add: sweater  │
└───────────────┴───────────────────────┴───────────────┘
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
