# 🎯 SentiTune: BERT Fine-Tuning for Sentiment Analysis

> Production-ready sentiment analysis system using BERT with FastAPI backend and React frontend

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.0+-orange.svg)](https://huggingface.co/transformers/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.68+-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-18+-61DAFB.svg)](https://reactjs.org/)

## 📖 Overview

SentiTune is a complete end-to-end sentiment analysis application that demonstrates production-level transformer fine-tuning and deployment. The project fine-tunes BERT on the IMDb dataset and deploys it through a FastAPI backend with a modern React frontend.

### Key Features

- ✅ BERT fine-tuned on 25,000 IMDb movie reviews
- ✅ FastAPI REST API for real-time inference
- ✅ React-based web interface
- ✅ Mixed precision training (FP16) for efficiency
- ✅ Production-ready model serving
- ✅ Complete training notebook included

---

## 🏗️ Project Structure

```
FINETUNNING_WITH_BERT/
├── frontend/                      # React web application
│   ├── src/
│   │   ├── App.js
│   │   ├── App.test.js
│   │   ├── index.css
│   │   ├── index.js
│   │   └── reportWebVitals.js
│   ├── package.json
│   └── package-lock.json
│
├── sentiment_model/               # Fine-tuned BERT model
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer_config.json
│   ├── tokenizer.json
│   └── training_args.bin
│
├── main.py                        # FastAPI backend server
├── model.py                       # Model loading & inference logic
└── SentiTune_BERT_Fine_Tuning_for_Sentiment_Analysis.ipynb
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- Node.js 14+ and npm
- CUDA-compatible GPU (for training, optional for inference)
- 8GB+ RAM

### Backend Setup

```bash
# Install Python dependencies
pip install transformers torch fastapi uvicorn pydantic

# Start the FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

The web app will open at `http://localhost:3000`

---

## 🎯 Model Training

### Training Configuration

The model was trained using the following setup:

| Parameter | Value |
|-----------|-------|
| Base Model | `bert-base-uncased` |
| Dataset | IMDb (25,000 training samples) |
| Learning Rate | 2e-5 |
| Batch Size | 8 per device |
| Epochs | 3 |
| Max Sequence Length | 128 tokens |
| Precision | Mixed (FP16) |
| Weight Decay | 0.01 |

### Training Results

```
Training Loss: 0.2451
Training Time: 866 seconds (~14.5 minutes)
Training Speed: 86.6 samples/second
GPU Used: Tesla T4
Total Steps: 9,375
```

### Running the Complete Notebook

To train the model from scratch:

```bash
# Open the Jupyter notebook
jupyter notebook SentiTune_BERT_Fine_Tuning_for_Sentiment_Analysis.ipynb

# Or use Google Colab (recommended for GPU access)
# Upload the notebook to Google Colab
# Runtime > Change runtime type > GPU (T4)
```

The notebook covers:
1. Dataset loading (IMDb)
2. Tokenization with BERT tokenizer
3. Model initialization
4. Training with Trainer API
5. Model saving
6. Inference testing

---

## 🔧 API Usage

### Health Check

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "message": "SentiTune API is running 🚀"
}
```

### Sentiment Prediction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was absolutely fantastic!"}'
```

Response:
```json
{
  "text": "This movie was absolutely fantastic!",
  "sentiment": "Positive"
}
```

### Using Python

```python
import requests

response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "I loved this movie!"}
)

print(response.json())
# Output: {'text': 'I loved this movie!', 'sentiment': 'Positive'}
```

---

## 💻 Frontend Features

The React frontend provides:

- Clean, modern UI for sentiment analysis
- Real-time API integration
- Input validation
- Responsive design
- Sentiment visualization

### Using the Web App

1. Open `http://localhost:3000` in your browser
2. Enter any text in the input field
3. Click "Analyze Sentiment"
4. View the predicted sentiment (Positive/Negative)

---

## 📊 Training Pipeline

The complete training workflow follows this architecture:

```
IMDb Dataset (25K reviews)
         ↓
   Tokenization
   (max_length=128)
         ↓
BERT Base Uncased Model
         ↓
  Fine-Tuning (3 epochs)
         ↓
   Model Checkpoint
         ↓
    Deployment
```

### Core Components

| Component | Implementation |
|-----------|----------------|
| Dataset | IMDb movie reviews (Hugging Face Datasets) |
| Tokenizer | `bert-base-uncased` AutoTokenizer |
| Model | BERT for Sequence Classification (2 labels) |
| Training | Hugging Face Trainer API |
| Optimization | AdamW with FP16 mixed precision |
| Deployment | FastAPI REST API |

---

## 🎓 Key Concepts

### 1. Tokenization

Converts text into numerical tokens that BERT understands:
- Truncates long sequences to 128 tokens
- Adds padding to maintain consistent length
- Creates attention masks (1 = real token, 0 = padding)

### 2. Mixed Precision Training (FP16)

Reduces memory usage and speeds up training:
- 50% less VRAM consumption
- 1.5-2x faster training
- Enables larger batch sizes

### 3. Binary Sentiment Classification

- Label 0: Negative sentiment
- Label 1: Positive sentiment
- Output: Logits converted to class prediction

---

## 🛠️ Dependencies

### Backend
```
transformers>=4.30.0
torch>=2.0.0
fastapi>=0.100.0
uvicorn>=0.23.0
pydantic>=2.0.0
```

### Frontend
```
react>=18.0.0
axios or fetch API
```

### Training (Notebook)
```
datasets>=2.12.0
evaluate>=0.4.0
accelerate>=0.20.0
```

---

## 📈 Performance Optimization

### Inference Speed

- CPU Inference: ~100-200ms per request
- GPU Inference: ~20-50ms per request

### Memory Requirements

- Model Size: ~440MB (BERT-base)
- Runtime Memory: ~2GB RAM (CPU inference)
- GPU Memory: ~1-2GB VRAM (if using GPU)

### Optimization Tips

1. **Batch Processing**: Process multiple texts together
2. **Model Quantization**: Use INT8 quantization for faster inference
3. **ONNX Export**: Convert to ONNX for production deployment
4. **Caching**: Implement Redis for frequently analyzed texts

---

## 🔍 Testing the Model

### Example Test Cases

```python
# Positive examples
"This movie was absolutely fantastic!"
"I loved every moment of it!"
"Best film I've seen this year!"

# Negative examples
"Terrible waste of time"
"I hated this movie"
"Worst experience ever"

# Neutral/Edge cases
"It was okay, nothing special"
"The movie exists"
```

---

## 🚀 Deployment Options

### 1. Local Deployment (Development)
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### 2. Production Deployment

**Using Docker:**
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Using Gunicorn (Production Server):**
```bash
pip install gunicorn
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### 3. Cloud Deployment

- **AWS**: Deploy on EC2 with Auto Scaling
- **Google Cloud**: Use Cloud Run for serverless deployment
- **Azure**: Deploy on Azure App Service
- **Heroku**: Simple deployment with Procfile

---

## 🐛 Troubleshooting

### Backend Issues

**Model not loading:**
```bash
# Ensure the sentiment_model directory exists
ls -la sentiment_model/

# Verify all required files are present:
# - config.json
# - model.safetensors
# - tokenizer_config.json
# - tokenizer.json
```

**Port already in use:**
```bash
# Use a different port
uvicorn main:app --port 8001
```

**CORS errors:**
- Check that CORS middleware is properly configured in `main.py`
- Ensure frontend origin is allowed

### Frontend Issues

**npm install fails:**
```bash
# Clear cache and retry
npm cache clean --force
rm -rf node_modules package-lock.json
npm install
```

**API connection errors:**
- Verify backend is running on port 8000
- Check network firewall settings
- Ensure correct API endpoint in frontend code

---

## 📚 Resources

### Documentation
- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [React Documentation](https://reactjs.org/docs)

### Research Papers
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Datasets
- [IMDb Dataset on Hugging Face](https://huggingface.co/datasets/imdb)

---

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

- [ ] Add evaluation metrics visualization
- [ ] Implement model confidence scores
- [ ] Add support for batch predictions
- [ ] Create Docker compose setup
- [ ] Add unit tests for API endpoints
- [ ] Implement model versioning
- [ ] Add CI/CD pipeline

---

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Stanford AI Lab for the IMDb dataset
- FastAPI team for the excellent web framework
- Google Colab for providing free GPU access

---

## 📧 Contact

For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Built with ❤️ using Hugging Face Transformers, FastAPI, and React**

*SentiTune - Making sentiment analysis simple and accessible*
