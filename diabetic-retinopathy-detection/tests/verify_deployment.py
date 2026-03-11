
import requests
import cv2
import numpy as np
import time
import sys

def test_api():
    print("--- Starting Deployment Verification ---")
    
    # 1. create dummy image
    print("[1/3] Creating synthetic fundus image...")
    dummy = np.zeros((512, 512, 3), dtype=np.uint8)
    cv2.circle(dummy, (256, 256), 200, (0, 0, 128), -1) # Reddish circle
    cv2.imwrite("synthetic_eye.jpg", dummy)
    
    # 2. Health check
    print("[2/3] Checking System Health...")
    try:
        r = requests.get("http://localhost:8000/health")
        if r.status_code == 200:
            print("   ✅ API is Online")
            print(f"   ℹ️  Device: {r.json().get('device')}")
        else:
            print(f"   ❌ Health Check Failed: {r.status_code} - {r.text}")
            sys.exit(1)
    except Exception as e:
        print(f"   ❌ Connection Failed: {e}")
        sys.exit(1)
        
    # 3. Prediction Test
    print("[3/3] Testing Prediction Endpoint...")
    try:
        t0 = time.time()
        files = {'file': open('synthetic_eye.jpg', 'rb')}
        r = requests.post("http://localhost:8000/predict", files=files)
        dt = time.time() - t0
        
        if r.status_code == 200:
            data = r.json()
            print("   ✅ Prediction Successful")
            print(f"   ⏱️  Total Latency: {dt:.2f}s")
            print(f"   🔍 Class: {data['predicted_label']}")
            print(f"   📊 Confidence: {data['confidence']:.2%}")
            
            if data.get('gradcam_base64'):
                print("   🎨 Grad-CAM Visualization: Present")
            else:
                print("   ⚠️  Grad-CAM Visualization: MISSING")
                
            print("--- Verification Complete: SYSTEM IS OPERATIONAL ---")
        else:
            print(f"   ❌ Prediction Failed: {r.status_code} - {r.text}")
            sys.exit(1)
            
    except Exception as e:
        print(f"   ❌ Prediction Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    test_api()
