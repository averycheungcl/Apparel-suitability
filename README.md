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
pip install opencv-python torch ultralytics supervision numpy requests
```

### 2. Configuration

- Obtain an API key from [OpenWeatherMap](https://openweathermap.org/)
- Replace the placeholder in `ApparelSuitability.py`:
  ```python
  API_KEY = "your_api_key_here"
  ```
- Download the custom YOLO model file (`best.pt`) and place it in the project directory

### 3. Running the Application

```bash
python ApparelSuitability.py
```

## 🔍 How It Works


1. **Weather Data Collection**
   - Enter your city name
   - Current temperature and "feels like" data retrieved

2. **User Preferences**
   - Indicate whether you're sensitive to cold
   - Confirm you're wearing basic clothing (t-shirt/tank top)

3. **Clothing Detection**
   - AI model identifies clothing items you're wearing
   - Each item is assigned a "warmth value"

4. **Temperature Analysis**
   - The application compares your clothing's warmth to the current temperature
   - Takes into account environmental conditions and personal preferences

5. **Smart Recommendations**
   - Suggests specific clothing items if you need more warmth
   - Advises if you're overdressed for the conditions

## 🧠 The AI Model

The project uses a custom-trained YOLO (You Only Look Once) model designed to detect various clothing items:

| Clothing Item | Warmth Value |
|---------------|--------------|
| T-shirt       | 3            |
| Dress shirt   | 3            |
| Vest          | 3            |
| Suit          | 5            |
| Sweater       | 7            |
| Trench coat   | 10           |

## ⌨️ Controls

- **ESC**: Exit the application
- **N**: Pause/resume detection

## 📁 Output

Detected clothing images are saved to an `output_images` directory created in the project folder.

## 📝 Example Output

```
What city: New York
Actual temp: 15.25
Feels like: 13.78
Are you prone to feeling cold? (yes/no): yes
Are you wearing at least a t-shirt or a tank top? (yes/no): yes
You are ready to proceed!
Using Device: cpu
You need at least: 2.74 Celsius worth of clothes!
We recommend you put on a sweater to stay warm.
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---
