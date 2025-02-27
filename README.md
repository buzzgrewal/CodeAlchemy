
# CodeAlchemy

**Transmuting Pseudocode into C++ Mastery**

## Overview

CodeAlchemy is a Transformer-based sequence-to-sequence model built from scratch in PyTorch that converts pseudocode into C++ code. This project demonstrates how to design, train, evaluate, and deploy a custom transformer model for code generation without relying on pretrained models.

## Features

- **Custom Transformer Model:** Implemented from scratch using PyTorch.
- **Vocabulary Building & Tokenization:** Efficiently constructs vocabularies for pseudocode and C++.
- **GPU-Optimized Training:** Uses gradient clipping and learning rate scheduling.
- **Evaluation:** Computes BLEU scores for assessing translation quality.
- **Testing & Inference:** Separate scripts for testing and interactive inference.
- **Interactive Streamlit UI:** Upload model files and get instant pseudocode-to-C++ conversion.

## Project Structure

```plaintext
CodeAlchemy/
├── model_train.py         # Transformer model, training routines, and utility functions.
├── eval.py                # Evaluation script that computes BLEU score.
├── test.py                # Testing script to generate predictions on test data.
├── streamlit_app.py       # Streamlit UI for interactive pseudocode-to-C++ generation.
├── README.md              # This file.
├── requirements.txt       # Python dependencies.
└── data/
    ├── train.tsv          # Training data (TSV format: pseudocode and C++ code).
    ├── eval.tsv           # Evaluation data.
    └── test.tsv           # Test data.
```

## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/CodeAlchemy.git
   cd CodeAlchemy
   ```

2. **Install Dependencies:**

   Ensure you have Python 3.8+ installed. Then, install the required packages:

   ```bash
   pip install -r requirements.txt
   ```

   *Example `requirements.txt`:*
   ```
   torch
   pandas
   nltk
   streamlit
   ```

3. **Download NLTK Data:**

   In a Python shell, run:

   ```python
   import nltk
   nltk.download('punkt')
   ```

## Data Preparation

- **Input Files:** Prepare your training, evaluation, and testing files in TSV (or CSV) format with at least these columns:
  - `text` – The pseudocode input.
  - `code` – The target C++ code.
- Place these files in the `data/` directory and adjust file paths in the scripts if necessary.

## Training the Model

Run the training script:

```bash
python model_train.py
```

This script will:
- Build the source and target vocabularies.
- Train the Transformer model using your training data.
- Save the model's state dictionary as `transformer_model.pth`.
- Save the source and target vocabularies as `src_vocab.pkl` and `trg_vocab.pkl`, respectively.

## Evaluation

Evaluate the trained model by running:

```bash
python eval.py
```

This script:
- Loads the model (state dictionary only) and the saved vocabularies.
- Runs inference on the evaluation data.
- Computes and prints the BLEU score.

## Testing

Generate predictions on your test data with:

```bash
python test.py
```

The script loads the model and vocabularies, translates the pseudocode inputs into C++ code, and saves the predictions to `test_predictions.txt`.

## Interactive Streamlit Interface

Try out the model interactively with the Streamlit app. The app allows you to upload your model and vocabulary files via the sidebar and generate C++ code from pseudocode inputs.

Run the app with:

```bash
streamlit run streamlit_app.py
```

**Sidebar Instructions:**
- Upload the `transformer_model.pth` file.
- Upload the `src_vocab.pkl` file.
- Upload the `trg_vocab.pkl` file.
- Enter your pseudocode in the main area and click **Generate Code** to see the output.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the model, code, or documentation.

## License

This project is licensed under the [MIT License](LICENSE).

## Acknowledgements

- Built with [PyTorch](https://pytorch.org/) and [Streamlit](https://streamlit.io/).
- Inspired by the growing field of AI-powered code generation.

