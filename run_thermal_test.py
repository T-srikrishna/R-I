"""
Thermal Emotion Recognition - Quick Launcher
============================================

Simple launcher for testing thermal emotion recognition with your laptop camera.
Choose between basic emotion detection or thermal simulation mode.
"""

import sys
import subprocess
import os

def print_header():
    print("🌡️  THERMAL EMOTION RECOGNITION")
    print("="*50)
    print("🎯 Testing thermal emotion recognition")
    print("🎥 Using laptop camera for simulation")
    print("🧠 Enhanced model: 88.93% accuracy")
    print("="*50)

def print_menu():
    print("\n📋 Choose mode:")
    print("1. 🎭 Basic Emotion Detection")
    print("2. 🌡️  Thermal Camera Simulator")
    print("3. ❌ Quit")
    print()

def run_basic_emotion():
    print("\n🚀 Starting Basic Emotion Detection...")
    print("📹 Press 'q' to quit")
    try:
        subprocess.run([sys.executable, "live_emotion_camera.py"])
    except FileNotFoundError:
        print("❌ Error: live_emotion_camera.py not found")
    except Exception as e:
        print(f"❌ Error: {e}")

def run_thermal_simulator():
    print("\n🚀 Starting Thermal Camera Simulator...")
    print("📹 Controls: 'q'=quit, 's'=screenshot, 'e'=export, SPACE=pause")
    try:
        subprocess.run([sys.executable, "thermal_simulator.py"])
    except FileNotFoundError:
        print("❌ Error: thermal_simulator.py not found")
    except Exception as e:
        print(f"❌ Error: {e}")

def check_model():
    """Check if the enhanced model exists"""
    if not os.path.exists("thermal_emotion_model_enhanced.h5"):
        print("❌ Error: thermal_emotion_model_enhanced.h5 not found")
        print("🔧 Please run the notebook to train the enhanced model first")
        return False
    return True

def main():
    print_header()
    
    # Check if model exists
    if not check_model():
        input("\nPress Enter to exit...")
        return
    
    while True:
        print_menu()
        choice = input("Enter your choice (1-3): ").strip()
        
        if choice == "1":
            run_basic_emotion()
        elif choice == "2":
            run_thermal_simulator()
        elif choice == "3":
            print("\n👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")
        
        input("\n⏸️  Press Enter to return to menu...")

if __name__ == "__main__":
    main()