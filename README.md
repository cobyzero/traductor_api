# Traductor API

A Eel web application that recognizes sign language letters from the MNIST dataset. The app allows users to either take a photo using their device's camera or upload an image for recognition.

## Features

- Image upload functionality
- Responsive design with Bootstrap 5
- Simple and intuitive user interface
- Displays prediction confidence score

## Prerequisites

- Python 3.9.6
- pip (Python package manager)

## Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd traductor_api
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: .\venv\Scripts\activate
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Eel

1. Run the Eel application:
   ```bash
   python app.py
   ```

2. Open your web browser and navigate to:
   ```
   http://127.0.0.1:27000/
   ```

3. Choose between using your camera or uploading an image:
   - **Camera Tab**: Click "Capture & Predict" to take a photo and get a prediction
   - **Upload Tab**: Click "Choose an image" to select an image file, then click "Predict"

## How It Works

1. The application uses a pre-trained CNN model based on the MNIST dataset.
2. When an image is captured or uploaded, it's preprocessed to match the MNIST format:
   - Converted to grayscale
   - Resized to 28x28 pixels
   - Inverted (MNIST uses white digits on black background)
   - Normalized
3. The preprocessed image is then passed to the model for prediction.
4. The predicted letter and confidence score are displayed to the user.

## Compile Executable

1. Compile the Eel application:
   ```bash
   pyinstaller --noconfirm --windowed app.py --add-data "templates;templates"
   ```

2. The executable will be created in the `dist` folder.

## Notes

- This is a demo application using the MNIST dataset, which contains handwritten digits (0-9).
- The mapping from digits to letters is simplified for demonstration purposes.
- For production use, consider training on an actual sign language dataset.

## License

This project is open source and available under the MIT License.
