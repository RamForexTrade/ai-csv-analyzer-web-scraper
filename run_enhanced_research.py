#!/usr/bin/env python3
"""
Enhanced Research Runner
Combines all features for maximum business research success
"""

import subprocess
import sys
import time
import os
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("=" * 70)
    print("🎭 ENHANCED BUSINESS RESEARCH")
    print("   AI-Powered CSV Analysis + Web Scraping")
    print("=" * 70)

def check_prerequisites():
    """Check that all required files exist"""
    required_files = [
        "ai_csv_analyzer.py",
        "data_explorer.py",
        "requirements.txt"
    ]
    
    missing_files = []
    for file in required_files:
        if not Path(file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required files:")
        for file in missing_files:
            print(f"   - {file}")
        return False
    
    print("✅ All required files found")
    return True

def run_research():
    """Run the main research application"""
    print("\n🔬 Starting AI-powered research application...")
    
    try:
        # Run the main CSV analyzer
        subprocess.run([sys.executable, "-m", "streamlit", "run", "ai_csv_analyzer.py"], check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Research application failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Error running research: {e}")
        return False

def main():
    """Main execution flow"""
    print_banner()
    
    # Step 1: Check prerequisites
    print("\n📋 Step 1: Checking prerequisites...")
    if not check_prerequisites():
        print("\n❌ Prerequisites check failed!")
        print("Please ensure all required files are present.")
        input("Press Enter to exit...")
        return
    
    # Step 2: Run research
    print("\n📋 Step 2: Starting research...")
    print("This will launch the main research application with:")
    print("✅ AI-powered CSV analysis")
    print("✅ Smart data filtering") 
    print("✅ Business contact research")
    print("✅ Web scraping capabilities")
    print("✅ Export functionality")
    
    proceed = input("\nStart research application? (y/n): ").strip().lower()
    if proceed == 'y':
        if run_research():
            print("\n🎉 Research completed successfully!")
        else:
            print("\n❌ Research encountered issues")
    
    print("\n" + "=" * 70)
    print("🎭 Enhanced Business Research Session Complete")
    print("=" * 70)
    input("Press Enter to exit...")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Session interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        input("Press Enter to exit...")
