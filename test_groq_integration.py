"""
Test Script for Groq Integration
Tests the Groq AI provider functionality for business research
"""

import os
import sys
import asyncio
import pandas as pd
from datetime import datetime

# Add modules to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

def test_groq_api_connection():
    """Test Groq API connection"""
    print("🧪 Testing Groq API Connection...")
    
    try:
        from streamlit_business_researcher import StreamlitBusinessResearcher
        
        # Test with Groq provider
        researcher = StreamlitBusinessResearcher(ai_provider="groq")
        
        # Test APIs
        api_ok, api_message = researcher.test_apis()
        
        if api_ok:
            print("✅ Groq API connection successful!")
            print(f"   Message: {api_message}")
            return True
        else:
            print(f"❌ Groq API connection failed: {api_message}")
            return False
            
    except ValueError as e:
        if "GROQ_API_KEY" in str(e):
            print("❌ GROQ_API_KEY not found in environment variables")
            print("💡 Add GROQ_API_KEY to your .env file")
        else:
            print(f"❌ Configuration error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def check_env_file():
    """Check .env file configuration"""
    print("\n📋 Checking .env File Configuration...")
    
    required_keys = {
        'TAVILY_API_KEY': 'tvly-',
        'OPENAI_API_KEY': 'sk-',
        'GROQ_API_KEY': 'gsk_'
    }
    
    env_status = {}
    
    for key, prefix in required_keys.items():
        value = os.getenv(key)
        
        if value:
            if value.startswith(prefix):
                env_status[key] = "✅ Configured"
            else:
                env_status[key] = f"⚠️  Invalid format (should start with {prefix})"
        else:
            env_status[key] = "❌ Missing"
    
    # Display table
    print("┌─────────────────┬─────────────────────────────────────┐")
    print("│ API Key         │ Status                              │")
    print("├─────────────────┼─────────────────────────────────────┤")
    
    for key, status in env_status.items():
        print(f"│ {key:<15} │ {status:<35} │")
    
    print("└─────────────────┴─────────────────────────────────────┘")
    
    # Check if Groq is ready
    groq_ready = env_status['GROQ_API_KEY'] == "✅ Configured" and env_status['TAVILY_API_KEY'] == "✅ Configured"
    
    if groq_ready:
        print("\n🎉 Groq setup is complete! You can use Groq for business research.")
    else:
        print("\n💡 To use Groq, add these to your .env file:")
        if env_status['GROQ_API_KEY'] != "✅ Configured":
            print("   GROQ_API_KEY=gsk_your_groq_api_key_here")
        if env_status['TAVILY_API_KEY'] != "✅ Configured":
            print("   TAVILY_API_KEY=tvly-your_tavily_api_key_here")
        print("\n   Get Groq API key from: https://console.groq.com/keys")
        print("   Get Tavily API key from: https://tavily.com")
    
    return groq_ready

def main():
    """Run all Groq integration tests"""
    print("🚀 Groq Integration Test Suite")
    print("=" * 60)
    
    # Check environment
    env_ready = check_env_file()
    
    if env_ready:
        # Test API connections
        groq_working = test_groq_api_connection()
        
        # Final summary
        print("\n" + "=" * 60)
        print("🎯 Test Summary:")
        
        if groq_working:
            print("✅ Groq integration is working!")
            print("✅ You can now select Groq in the web scraping interface")
            print("✅ Groq will use the Llama-3.3-70b-versatile model")
            
            # Usage instructions
            print("\n📝 How to use Groq:")
            print("1. Run your Streamlit app: streamlit run ai_csv_analyzer.py")
            print("2. Go to Data Explorer > Research Options")
            print("3. Select 'Groq (Llama-3.3-70b)' as your AI provider")
            print("4. Start research to see Groq in action!")
            
        else:
            print("❌ Groq integration needs setup")
            print("💡 Configure your GROQ_API_KEY in the .env file")
    
    else:
        print("\n⚠️  Environment setup required before testing")

if __name__ == "__main__":
    main()
