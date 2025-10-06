#!/usr/bin/env python3
"""
RPC Health Check
Week 0: Verify RPC connection and latency
"""
import os
import sys
import time
import json
from pathlib import Path

try:
    import requests
    from dotenv import load_dotenv
except ImportError:
    print("❌ Missing dependencies. Install:")
    print("   pip install -e '.[crypto]'")
    sys.exit(1)

HERE = Path(__file__).parent
load_dotenv(HERE / ".env")

RPC_URL = os.getenv("RPC_URL")
CHAIN_ID = int(os.getenv("CHAIN_ID", 11155111))


def rpc(method, params=None):
    """Call JSON-RPC method"""
    t0 = time.perf_counter()
    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": params or []
    }
    r = requests.post(RPC_URL, json=payload, timeout=10)
    dt = (time.perf_counter() - t0) * 1000  # ms
    r.raise_for_status()
    return r.json(), dt


def main():
    print("=== RPC Health Check ===\n")
    
    if not RPC_URL:
        print("❌ RPC_URL yok. .env doldur.")
        print("   cp .env.example .env")
        print("   vim .env")
        return False
    
    if "YOUR_API_KEY" in RPC_URL:
        print("❌ RPC_URL placeholder içeriyor.")
        print("   Gerçek API key ekle: https://dashboard.alchemy.com")
        return False
    
    # Mask API key in output
    masked = RPC_URL
    if "/v2/" in masked:
        parts = masked.split("/v2/")
        masked = parts[0] + "/v2/***" + parts[1][-4:]
    print(f"RPC: {masked}")
    print(f"Chain ID: {CHAIN_ID}\n")
    
    # Test: Latest block
    try:
        res, ms = rpc("eth_blockNumber")
        block_hex = res.get("result")
        if not block_hex:
            print(f"❌ Unexpected response: {res}")
            return False
        
        bn = int(block_hex, 16)
        
        # Latency check
        threshold_ms = 300
        if ms > threshold_ms:
            print(f"⚠️  RPC OK but SLOW | latest block: {bn} | {ms:.1f} ms (> {threshold_ms}ms)")
            print("   Consider using a different provider or region")
        else:
            print(f"✅ RPC OK | latest block: {bn} | {ms:.1f} ms")
        
        return True
        
    except requests.exceptions.Timeout:
        print("❌ Request timeout. Check RPC URL or network.")
        return False
    except requests.exceptions.ConnectionError as e:
        print(f"❌ Connection error: {e}")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
