# IMDb Movie Review Sentiment Analyzer

AI-powered sentiment classification with IMDb-style star ratings.

## Prerequisites

- Python 3.8+
- A pretrained Keras model file (`.h5` format)

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Add Your Model

Place your pretrained model file in the project root:
```
project/
├── imdb_sentiment_model.h5    # Your model here
├── app.py
├── index.html
├── styles.css
└── script.js
```

Or update the model path in `app.py` (line 26):
```python
MODEL_PATH = 'path/to/your/model.h5'
```

## Running the Application

### Start Backend (Flask)

```bash
python app.py
```

Backend will run on: `http://localhost:5000`

### Start Frontend

Open `index.html` in your browser:

**Option 1: Direct Open**
```bash
# macOS
open index.html

# Windows
start index.html

# Linux
xdg-open index.html
```

**Option 2: Local Server (Recommended)**
```bash
# Python 3
python -m http.server 8000

# Then open: http://localhost:8000
```

**Option 3: VS Code Live Server**
- Install "Live Server" extension
- Right-click `index.html` → "Open with Live Server"

## Usage

1. Write a movie review in the text area
2. Click "Analyze Sentiment" or press `Ctrl/Cmd + Enter`
3. View results:
   - Sentiment (Positive/Negative)
   - IMDb star rating (0-10 scale)
   - Review preview

## File Structure

```
project/
├── app.py                      # Flask backend
├── requirements.txt            # Python dependencies
├── index.html                  # Frontend HTML
├── styles.css                  # Styling
├── script.js                   # Frontend logic
├── imdb_sentiment_model.h5    # Your trained model (not included)
└── README.md                   # This file
```

## Troubleshooting

### "Connection Error" in Frontend

**Problem**: Frontend can't connect to backend

**Solution**: Ensure Flask is running and CORS is enabled
```bash
# Check if Flask is running
curl http://localhost:5000/health

# Should return: {"status": "healthy", ...}
```

### Model Not Found

**Problem**: `FileNotFoundError: imdb_sentiment_model.h5`

**Solution**: 
- Verify model file exists in project root
- Check `MODEL_PATH` in `app.py` points to correct location

### Port Already in Use

**Problem**: `Address already in use: 5000`

**Solution**: Kill the process or use a different port
```bash
# Kill process on port 5000 (macOS/Linux)
lsof -ti:5000 | xargs kill -9

# Or change port in app.py
app.run(debug=True, host='0.0.0.0', port=5001)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Service status |
| `/health` | GET | Detailed health check |
| `/predict` | POST | Analyze sentiment |

### Example API Request

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"review": "This movie was amazing!"}'
```

### Example API Response

```json
{
  "status": "success",
  "review": "This movie was amazing!",
  "sentiment": "Positive",
  "confidence": 0.9542
}
```

## Technologies

- **Backend**: Flask, TensorFlow/Keras
- **Frontend**: HTML, CSS, JavaScript (Vanilla)
- **Styling**: IBM Plex Sans/Mono fonts
- **Theme**: Dark (IMDb-inspired)

## Notes

- Model is loaded once at startup (not per request)
- Star rating = confidence × 10 (e.g., 0.85 → 8.5/10)
- Frontend works with any CORS-enabled Flask backend
