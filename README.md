# 🎯 SentiTune: BERT Fine-Tuning for Sentiment Analysis

> Production-ready sentiment analysis system using BERT with FastAPI backend and React frontend

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
0
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

## 📝 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

- Hugging Face for the Transformers library
- Stanford AI Lab for the IMDb dataset
- FastAPI team for the excellent web framework
- Google Colab for providing free GPU access

---


For questions, issues, or collaboration opportunities, please open an issue on GitHub.

---

**Built with ❤️ using Hugging Face Transformers, FastAPI, and React**

*SentiTune - Making sentiment analysis simple and accessible*
