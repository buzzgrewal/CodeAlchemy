import streamlit as st
import torch
import pickle
from model_train import TransformerSeq2Seq, PositionalEncoding, Vocabulary

def load_vocab(file_bytes):
    return pickle.load(file_bytes)


def translate_input(model, pseudocode, src_vocab, trg_vocab, device, max_len=50):
    tokens = [src_vocab.stoi["<sos>"]] + src_vocab.numericalize(pseudocode) + [src_vocab.stoi["<eos>"]]
    src_tensor = torch.LongTensor(tokens).unsqueeze(0).to(device)
    
    trg_init = torch.LongTensor([trg_vocab.stoi["<sos>"]]).unsqueeze(0).to(device)
    preds = trg_init

    for _ in range(max_len):
        output = model(src_tensor, preds)
        next_token_logits = output[:, -1, :]
        next_tokens = next_token_logits.argmax(dim=-1, keepdim=True)
        preds = torch.cat([preds, next_tokens], dim=1)
        if (next_tokens == trg_vocab.stoi["<eos>"]).all():
            break

    output_tokens = []
    for token in preds[0]:
        token = token.item()
        if token == trg_vocab.stoi["<sos>"]:
            continue
        if token == trg_vocab.stoi["<eos>"]:
            break
        output_tokens.append(trg_vocab.itos[token])
    return " ".join(output_tokens)

st.title("CodeAlchemy ⚗️")
st.subheader("Transmuting Pseudocode into C++ Mastery 💻➡️🎯")

st.sidebar.header("Upload Required Files")
model_file = st.sidebar.file_uploader("Transformer Model (.pth)", type=["pth"])
src_vocab_file = st.sidebar.file_uploader("Source Vocabulary (.pkl)", type=["pkl"])
trg_vocab_file = st.sidebar.file_uploader("Target Vocabulary (.pkl)", type=["pkl"])

pseudocode_input = st.text_area("Enter your pseudocode below:")

if st.button("Generate C++ Code"):
    if model_file and src_vocab_file and trg_vocab_file and pseudocode_input.strip() != "":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        st.write("Using device:", device)
        
        src_vocab = load_vocab(src_vocab_file)
        trg_vocab = load_vocab(trg_vocab_file)
        
        state_dict = torch.load(model_file, map_location=device)
        model = TransformerSeq2Seq(len(src_vocab.stoi), len(trg_vocab.stoi)).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        
        generated_code = translate_input(model, pseudocode_input, src_vocab, trg_vocab, device)
        
        st.subheader("Generated C++ Code")
        st.code(generated_code, language="cpp")
    else:
        st.error("Please upload all required files and enter your pseudocode.")
