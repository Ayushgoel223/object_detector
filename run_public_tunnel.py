"""
BlindAid Public HTTPS Tunnel Generator
========================================
Creates a free public HTTPS URL (e.g. https://xxxx.loca.lt or https://xxxx.serveo.net)
so ANY smartphone can access the BlindAid camera & voice web app securely.

Usage:
    python run_public_tunnel.py
"""

import subprocess
import time
import sys
import os

def start_tunnel():
    print("=" * 60)
    print("  BlindAid — Public HTTPS Mobile Deployment")
    print("=" * 60)
    print("  Creating a secure public HTTPS link for your smartphone...")
    print("=" * 60)

    # Method 1: Try npx localtunnel
    try:
        proc = subprocess.Popen(
            ["npx", "-y", "localtunnel", "--port", "5000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        print("\n[Tunnel] LocalTunnel starting on port 5000...")
        time.sleep(3)
        for line in proc.stdout:
            if "url is" in line.lower() or "https://" in line.lower():
                print("\n" + "🌟" * 30)
                print(f"  MOBILE PUBLIC URL: {line.strip()}")
                print("🌟" * 30 + "\n")
                print("Open this link on your smartphone browser (Chrome/Safari)!")
                print("Camera and Bluetooth Audio speech output are enabled.\n")
                break
        proc.wait()
        return
    except Exception as e:
        print(f"[Tunnel] localtunnel notice: {e}")

    # Method 2: Fallback to SSH Serveo tunnel
    try:
        print("\n[Tunnel] Trying Serveo SSH tunnel...")
        proc = subprocess.Popen(
            ["ssh", "-R", "80:localhost:5000", "serveo.net"],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        for line in proc.stdout:
            print(f"  [Serveo] {line.strip()}")
            if "https://" in line.lower():
                print("\n" + "🌟" * 30)
                print(f"  MOBILE PUBLIC URL: {line.strip()}")
                print("🌟" * 30 + "\n")
                break
        proc.wait()
    except Exception as e:
        print(f"[Tunnel] Serveo notice: {e}")

if __name__ == "__main__":
    start_tunnel()
