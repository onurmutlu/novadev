#!/usr/bin/env python3
"""
RPC Health Check Script
Week 0: Verify RPC connection and latency
"""
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from web3 import Web3
    from dotenv import load_dotenv
except ImportError:
    print("❌ Missing dependencies. Install:")
    print("   pip install web3 python-dotenv")
    sys.exit(1)


def check_rpc_health():
    """Run RPC health checks"""
    print("=== RPC Health Check ===\n")
    
    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    if not env_path.exists():
        print(f"❌ .env file not found: {env_path}")
        print("   Copy .env.example to .env and configure")
        return False
    
    load_dotenv(env_path)
    
    # Get config
    rpc_url = os.getenv("RPC_URL")
    chain_id = int(os.getenv("CHAIN_ID", 11155111))
    network_name = os.getenv("NETWORK_NAME", "Sepolia")
    
    if not rpc_url or "YOUR_API_KEY" in rpc_url:
        print("❌ RPC_URL not configured in .env")
        print("   Get API key from: https://dashboard.alchemy.com")
        return False
    
    # Mask API key in output
    masked_url = rpc_url.split("/")
    if len(masked_url) > 2:
        masked_url[-1] = "***" + masked_url[-1][-4:]
    
    print(f"RPC URL: {'/'.join(masked_url)}")
    print(f"Network: {network_name} ({chain_id})\n")
    
    # Connect
    try:
        w3 = Web3(Web3.HTTPProvider(rpc_url))
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False
    
    # Test 1: Connection
    print("[Testing] Connection...")
    if not w3.is_connected():
        print("❌ Not connected")
        return False
    print("[✓] Connection OK")
    
    # Test 2: Chain ID
    print("[Testing] Chain ID...")
    try:
        actual_chain_id = w3.eth.chain_id
        if actual_chain_id != chain_id:
            print(f"⚠️  Chain ID mismatch: expected {chain_id}, got {actual_chain_id}")
            return False
        print(f"[✓] Chain ID: {actual_chain_id} ✓")
    except Exception as e:
        print(f"❌ Chain ID check failed: {e}")
        return False
    
    # Test 3: Latest block
    print("[Testing] Latest block...")
    try:
        start = time.time()
        block_number = w3.eth.block_number
        latency_ms = (time.time() - start) * 1000
        print(f"[✓] Latest block: {block_number}")
    except Exception as e:
        print(f"❌ Block number fetch failed: {e}")
        return False
    
    # Test 4: Latency
    print("[Testing] Latency...")
    threshold_ms = 300
    if latency_ms > threshold_ms:
        print(f"⚠️  Latency: {latency_ms:.0f}ms (> {threshold_ms}ms)")
        print("   Consider using a different RPC provider")
    else:
        print(f"[✓] Latency: {latency_ms:.0f}ms (< {threshold_ms}ms) ✓")
    
    # Test 5: Get block (verify data access)
    print("[Testing] Block data access...")
    try:
        block = w3.eth.get_block("latest")
        tx_count = len(block["transactions"])
        print(f"[✓] Block data OK (timestamp: {block['timestamp']}, txs: {tx_count})")
    except Exception as e:
        print(f"❌ Block data access failed: {e}")
        return False
    
    # Summary
    print("\n" + "="*40)
    print("Status: HEALTHY ✓")
    print("="*40)
    print(f"""
Next steps:
  1. Test event capture:
     python collector/event_capture.py --blocks 100
  
  2. Setup database:
     duckdb crypto/db/crypto.db < crypto/db/schema.sql
  
  3. Test wallet report:
     python -m crypto.api.wallet_report --address 0x742d35Cc6634C0532925a3b844Bc9e7595f0bEb
""")
    
    return True


if __name__ == "__main__":
    success = check_rpc_health()
    sys.exit(0 if success else 1)
