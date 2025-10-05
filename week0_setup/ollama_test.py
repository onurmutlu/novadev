"""
Hafta 0: Ollama Lokal LLM Test
Basit prompt gönderme denemesi (HTTP API üzerinden).
"""


try:
    import requests

    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  'requests' paketi bulunamadı. Kurulum için:")
    print("   pip install requests")


def test_ollama(model: str = "qwen2.5:7b", prompt: str = "Explain PyTorch tensors in 2 sentences."):
    """Test Ollama API with a simple prompt."""
    if not REQUESTS_AVAILABLE:
        return

    url = "http://localhost:11434/api/generate"
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
    }

    print(f"🧠 Testing Ollama with model: {model}")
    print(f"📝 Prompt: {prompt}\n")

    try:
        response = requests.post(url, json=payload, timeout=60)
        response.raise_for_status()

        result = response.json()
        answer = result.get("response", "")

        print(f"✅ Response:\n{answer}\n")
        print(f"⏱️  Total duration: {result.get('total_duration', 0) / 1e9:.2f}s")
        print(f"📊 Tokens generated: {result.get('eval_count', 0)}")

    except requests.exceptions.ConnectionError:
        print("❌ Ollama servisine bağlanılamadı!")
        print("   Çözüm: Terminal'de 'ollama serve' komutunu çalıştır.")
    except requests.exceptions.Timeout:
        print("❌ İstek zaman aşımına uğradı (60s). Model çok büyük olabilir.")
    except requests.exceptions.HTTPError as e:
        print(f"❌ HTTP hatası: {e}")
        print("   Model indirilmiş mi? Kontrol: ollama list")
    except Exception as e:
        print(f"❌ Beklenmeyen hata: {e}")


def list_ollama_models():
    """List available Ollama models."""
    if not REQUESTS_AVAILABLE:
        return

    url = "http://localhost:11434/api/tags"

    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()

        models = response.json().get("models", [])
        if models:
            print("📚 İndirilen Ollama modelleri:")
            for model in models:
                name = model.get("name", "unknown")
                size_gb = model.get("size", 0) / (1024**3)
                print(f"   - {name} ({size_gb:.2f} GB)")
            print()
        else:
            print("⚠️  Henüz model indirilmemiş. Örnek:")
            print("   ollama pull qwen2.5:7b")
            print()

    except requests.exceptions.ConnectionError:
        print("❌ Ollama servisine bağlanılamadı.")
        print("   Çözüm: ollama serve\n")


if __name__ == "__main__":
    print("=" * 60)
    print("  Ollama Lokal LLM Test")
    print("=" * 60 + "\n")

    list_ollama_models()
    test_ollama()

    print("\n💡 Farklı model denemek için:")
    print("   python ollama_test.py")
    print("   # Kod içinde model='llama3.2:7b' olarak değiştir\n")
