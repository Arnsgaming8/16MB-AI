# 🤖 Mini AI - Pre-trained 16MB Neural Network

A compact transformer-based language model that fits within 16MB with pre-trained weights.

## 🌐 Live Demo

Deploy to GitHub Pages to see the AI in action!

## 📁 Project Structure

```
16MB AI/
├── web/
│   └── index.html          # Web interface (GitHub Pages)
├── mini_ai/
│   ├── __init__.py         # Package initialization
│   ├── model.py            # Model architecture
│   ├── train.py            # Full training script
│   ├── create_pretrained.py  # Quick pre-training
│   ├── inference.py        # Inference script
│   └── pretrained/         # Pre-trained weights
│       ├── model.pt        # Model weights (3.1 MB)
│       └── tokenizer.json  # Tokenizer
├── .github/
│   └── workflows/
│       └── deploy.yml      # GitHub Pages deployment
└── README.md
```

## 🚀 Quick Start

### Run Locally (Python)

```bash
cd mini_ai
python inference.py --demo          # See examples
python inference.py --interactive   # Chat mode
python inference.py --prompt "Your text"
```

### Deploy to GitHub Pages

1. **Push to GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR_USERNAME/16MB-AI.git
   git push -u origin main
   ```

2. **Enable GitHub Pages**
   - Go to your repository on GitHub
   - Settings → Pages
   - Source: Select "Deploy from a branch"
   - Branch: `gh-pages` / `(root)`
   - Click Save

3. **Automatic Deployment** (with workflow)
   - The workflow file is already set up
   - Push to trigger automatic deployment
   - Or go to Actions → Deploy to GitHub Pages → Run workflow

### Alternative: Manual Deployment

```bash
# Build the web folder
git checkout -b gh-pages
git push origin gh-pages
```

Then go to: `https://YOUR_USERNAME.github.io/16MB-AI/`

## 💻 Local Web Preview

```bash
cd web
python -m http.server 8000
# Open http://localhost:8000
```

## 🔧 Model Details

- **Architecture**: Transformer (4 layers, 4 heads, 128 hidden)
- **Parameters**: 800,512
- **Size**: 3.13 MB (well under 16MB)
- **Type**: Character-level language model

## 📝 Usage Examples

### Python
```python
import sys
sys.path.insert(0, 'mini_ai')
from inference import load_model, generate_text

model, tokenizer = load_model('mini_ai/pretrained')
result = generate_text(model, tokenizer, "Machine learning")
print(result)
```

### Web Interface
- Open the deployed page
- Type a message
- The AI generates a response

## 📦 Files

| File | Description |
|------|-------------|
| `mini_ai/model.py` | PyTorch transformer model |
| `mini_ai/train.py` | Full training pipeline |
| `mini_ai/inference.py` | CLI inference tool |
| `web/index.html` | Browser-based demo |
| `.github/workflows/deploy.yml` | Auto-deploy to Pages |

## 🤝 Contributing

1. Fork the repo
2. Create a branch
3. Make changes
4. Push and create PR

## 📄 License

MIT License
