import os
import urllib.request
import torch
from ai_otso import AI_Otso, device, encode, decode, vocab_size

MODEL_PATH = "full_model.pth"
MODEL_URL = "https://github.com/cmdman-urlman/Ai-otso/raw/refs/heads/main/full_model.pth"


def download_model():
    print("⬇️ Ladataan AI Otso -malli verkosta...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("✅ Malli ladattu onnistuneesti!")
    except Exception as e:
        print("❌ Mallin lataus epäonnistui:", e)
        raise


def load_ai_otso():
    # tarkista onko mallia
    if not os.path.exists(MODEL_PATH):
        print("⚠️ full_model.pth puuttuu.")
        download_model()

    # yritä ladata
    try:
        print("🔍 Ladataan mallia...")
        model = torch.load(MODEL_PATH, map_location=device)
        model.eval()
        print("🐻 AI Otso on nyt valmis!")
        return model
    except Exception as e:
        print("❌ Mallin lataus epäonnistui:", e)
        print("Yritetään ladata malli uudelleen verkosta...")
        download_model()
        model = torch.load(MODEL_PATH, map_location=device)
        model.eval()
        return model
