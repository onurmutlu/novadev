#!/usr/bin/env python3
"""
Event Capture Script
Week 0: Test capturing Transfer events from last N blocks
"""
import os
import sys
import argparse
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from web3 import Web3
    from dotenv import load_dotenv
except ImportError:
    print("❌ Missing dependencies. Install:")
    print("   pip install web3 python-dotenv")
    sys.exit(1)

# ERC20 Transfer event signature
# Transfer(address indexed from, address indexed to, uint256 value)
TRANSFER_TOPIC = "0xddf252ad1be2c89b69c2b068fc378daa952ba7f163c4a11628f55a4df523b3ef"


def capture_events(blocks=100, verbose=False):
    """Capture Transfer events from last N blocks"""
    print("=== Event Capture Test ===\n")
    
    # Load .env
    env_path = Path(__file__).parent.parent / ".env"
    load_dotenv(env_path)
    
    rpc_url = os.getenv("RPC_URL")
    if not rpc_url or "YOUR_API_KEY" in rpc_url:
        print("❌ RPC_URL not configured")
        return False
    
    # Connect
    print("Connecting to RPC...")
    w3 = Web3(Web3.HTTPProvider(rpc_url))
    if not w3.is_connected():
        print("❌ Connection failed")
        return False
    
    # Get block range
    latest = w3.eth.block_number
    from_block = max(0, latest - blocks)
    to_block = latest
    
    print(f"Scanning last {blocks} blocks...")
    print(f"Block range: {from_block} → {to_block}\n")
    
    # Filter for Transfer events
    print("Filter: Transfer(address,address,uint256)")
    print(f"Topic: {TRANSFER_TOPIC}\n")
    
    # Get logs
    print("Fetching logs...")
    try:
        logs = w3.eth.get_logs({
            "fromBlock": from_block,
            "toBlock": to_block,
            "topics": [TRANSFER_TOPIC]
        })
    except Exception as e:
        print(f"❌ Failed to fetch logs: {e}")
        return False
    
    # Count by block
    block_counts = {}
    for log in logs:
        block_num = log["blockNumber"]
        block_counts[block_num] = block_counts.get(block_num, 0) + 1
    
    # Display results
    total = len(logs)
    blocks_with_events = len(block_counts)
    
    if verbose and total > 0:
        for block_num in sorted(block_counts.keys())[:10]:  # First 10
            count = block_counts[block_num]
            print(f"[✓] Block {block_num}: {count} transfer(s)")
        if len(block_counts) > 10:
            print(f"... ({len(block_counts) - 10} more blocks)")
        print()
    
    print(f"Total: {total} transfers captured")
    print(f"Blocks with events: {blocks_with_events}/{blocks}")
    print(f"Capture rate: {blocks_with_events/blocks*100:.1f}%")
    
    if total > 0:
        # Show sample event
        sample = logs[0]
        print(f"\nSample event:")
        print(f"  Block: {sample['blockNumber']}")
        print(f"  Tx: {sample['transactionHash'].hex()}")
        print(f"  Token: {sample['address']}")
        print(f"  Topics: {len(sample['topics'])}")
    
    # Summary
    print("\n" + "="*40)
    if total > 0:
        print("Status: OK ✓")
        print("Event capture working!")
    else:
        print("Status: ⚠️  No events found")
        print("(Normal for low-activity testnet)")
        print("Try increasing --blocks or use mainnet")
    print("="*40)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Capture Transfer events")
    parser.add_argument(
        "--blocks",
        type=int,
        default=100,
        help="Number of recent blocks to scan (default: 100)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    success = capture_events(blocks=args.blocks, verbose=args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
