"""
Web Scraping Module for Business Contact Research
Simplified version optimized for Railway deployment
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import asyncio
import requests
import json

def get_env_var(key, default=None):
    """Get environment variable from .env file (local) or Railway environment (deployment)"""
    value = os.getenv(key)
    if value:
        return value

    # Try Streamlit secrets as fallback
    try:
        if hasattr(st, 'secrets') and key in st.secrets:
            return st.secrets[key]
    except Exception:
        pass

    return default

def perform_web_scraping(filtered_df):
    """Perform web scraping of business contact information from filtered data"""
    
    # Check if DataFrame is empty
    if len(filtered_df) == 0:
        st.error("❌ No data to scrape. Please adjust your filters.")
        return

    # Find suitable columns for business names
    potential_name_columns = []
    for col in filtered_df.columns:
        col_lower = col.lower()
        if any(keyword in col_lower for keyword in ['consignee', 'name', 'company', 'business', 'shipper', 'supplier']):
            potential_name_columns.append(col)

    if not potential_name_columns:
        st.error("❌ No suitable business name columns found. Need columns like 'Consignee Name', 'Company Name', etc.")
        return

    # User selects which column to use for business names
    st.write("🏷️ **Select Business Name Column:**")
    selected_column = st.selectbox(
        "Choose the column containing business names:",
        potential_name_columns,
        help="Select the column that contains the business names you want to research",
        key="business_name_column_selector"
    )

    # Check unique business count
    unique_businesses = filtered_df[selected_column].dropna().nunique()
    if unique_businesses == 0:
        st.error(f"❌ No business names found in column '{selected_column}'")
        return

    st.info(f"📊 Found {unique_businesses} unique businesses to research in '{selected_column}'")

    # Research limit selection
    if 'business_range_from' not in st.session_state:
        st.session_state.business_range_from = 1
    if 'business_range_to' not in st.session_state:
        st.session_state.business_range_to = min(5, unique_businesses)

    st.write("🎯 **Business Research Range:**")
    col_from, col_to = st.columns(2)
    
    with col_from:
        range_from = st.number_input(
            "From:",
            min_value=1,
            max_value=min(20, unique_businesses),
            value=st.session_state.business_range_from,
            help="Starting business number",
            key="business_range_from_input"
        )
    
    with col_to:
        range_to = st.number_input(
            "To:",
            min_value=range_from,
            max_value=min(20, unique_businesses),
            value=max(st.session_state.business_range_to, range_from),
            help="Ending business number",
            key="business_range_to_input"
        )
    
    # Calculate number of businesses to research
    max_businesses = range_to - range_from + 1
    
    # Update session state
    st.session_state.business_range_from = range_from
    st.session_state.business_range_to = range_to
    
    # Show summary
    st.info(f"📊 Will research businesses {range_from} to {range_to} ({max_businesses} total businesses)")

    # Cost estimation
    estimated_cost = max_businesses * 0.03
    st.warning(f"💰 **Estimated API Cost:** ~${estimated_cost:.2f} (approx $0.03 per business)")

    # API Configuration check
    st.write("🔧 **API Configuration:**")

    groq_key = get_env_var('GROQ_API_KEY')
    tavily_key = get_env_var('TAVILY_API_KEY')

    # Key validation function
    def is_valid_key(key, key_type):
        if not key or key.strip() == '':
            return False, "Key is empty or missing"
        if key.strip() in ['your_groq_key_here', 'your_tavily_key_here', 'gsk_...', 'tvly-...']:
            return False, "Key is a placeholder value"
        if key_type == 'groq' and not key.startswith('gsk_'):
            return False, "Groq key should start with 'gsk_'"
        if key_type == 'tavily' and not key.startswith('tvly-'):
            return False, "Tavily key should start with 'tvly-'"
        return True, "Key format is valid"

    groq_valid, groq_reason = is_valid_key(groq_key, 'groq')
    tavily_valid, tavily_reason = is_valid_key(tavily_key, 'tavily')

    # Display API status
    col_api1, col_api2 = st.columns(2)

    with col_api1:
        if groq_valid:
            st.success("✅ Groq API Key: Configured")
            masked_key = f"{groq_key[:10]}...{groq_key[-4:]}" if len(groq_key) > 14 else f"{groq_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Groq API Key: {groq_reason}")
            st.caption("Add GROQ_API_KEY to environment variables")

    with col_api2:
        if tavily_valid:
            st.success("✅ Tavily API Key: Configured")
            masked_key = f"{tavily_key[:10]}...{tavily_key[-4:]}" if len(tavily_key) > 14 else f"{tavily_key[:6]}..."
            st.caption(f"Key: {masked_key}")
        else:
            st.error(f"❌ Tavily API Key: {tavily_reason}")
            st.caption("Add TAVILY_API_KEY to environment variables")

    # Show setup instructions if keys are invalid
    if not groq_valid or not tavily_valid:
        st.warning("⚠️ **Setup Required**: Please configure both API keys before starting research.")

        with st.expander("📝 Setup Instructions", expanded=False):
            st.markdown("""
            **For Railway Deployment:**
            
            1. **Go to your Railway project dashboard**
            2. **Navigate to Variables tab**
            3. **Add environment variables:**
               ```
               GROQ_API_KEY=gsk_your_actual_groq_key_here
               TAVILY_API_KEY=tvly-your_actual_tavily_key_here
               ```
            4. **Redeploy your application**
            
            **Get API keys from:**
            - [Groq API Keys](https://console.groq.com/keys) (Fast, cost-effective)
            - [Tavily API](https://tavily.com) (Web search)

            **For Local Development:**
            Create a `.env` file with:
            ```
            GROQ_API_KEY=gsk_your_groq_key_here
            TAVILY_API_KEY=tvly-your_tavily_key_here
            ```
            """)

    # Test API connectivity if both keys are valid
    both_apis_configured = groq_valid and tavily_valid

    if both_apis_configured:
        st.info("🟢 **Both API keys configured!** You can proceed with web scraping.")

        # Test API connection button
        if st.button("🧪 Test API Connection", help="Test if APIs are working correctly", key="test_api_button"):
            with st.spinner("Testing API connections..."):
                api_ok, api_message = test_apis(groq_key, tavily_key)
                
                if api_ok:
                    st.success(f"✅ API Test Successful: {api_message}")
                else:
                    st.error(f"❌ API Test Failed: {api_message}")

    # Research button
    st.markdown("---")
    button_disabled = not both_apis_configured
    button_help = f"Research {max_businesses} businesses using AI web scraping" if both_apis_configured else "Configure both API keys first"

    if st.button(
        f"🚀 Start Research ({max_businesses} businesses)",
        type="primary",
        disabled=button_disabled,
        help=button_help,
        key="start_research_button"
    ):
        if not both_apis_configured:
            st.error("❌ Cannot start research: API keys not properly configured")
            return

        # Show starting message
        st.info("🔄 Starting business research...")

        try:
            # Get businesses to research
            unique_businesses_list = filtered_df[selected_column].dropna().unique()
            start_idx = range_from - 1
            end_idx = range_to
            businesses_to_research = unique_businesses_list[start_idx:end_idx]

            research_df = filtered_df[filtered_df[selected_column].isin(businesses_to_research)]

            st.info(f"🎯 **Researching businesses {range_from} to {range_to}:**")
            for i, business in enumerate(businesses_to_research, start=range_from):
                st.write(f"   {i}. {business}")

            # Run the research
            progress_bar = st.progress(0)
            status_text = st.empty()

            status_text.info("🚀 Initializing research system...")
            progress_bar.progress(10)

            # Execute research
            results_df, summary = research_businesses_simplified(
                businesses_to_research, 
                groq_key, 
                tavily_key,
                progress_bar,
                status_text
            )

            progress_bar.progress(100)
            status_text.success("✅ Research completed!")

            if results_df is not None and not results_df.empty:
                # Display summary
                st.success(f"🎉 **Research Summary:**")
                col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)

                with col_sum1:
                    st.metric("Total Processed", summary['total_processed'])
                with col_sum2:
                    st.metric("Successful", summary['successful'])
                with col_sum3:
                    st.metric("Manual Required", summary['manual_required'])
                with col_sum4:
                    st.metric("Success Rate", f"{summary['success_rate']:.1f}%")

                # Display results table
                st.subheader("📈 Research Results")
                st.dataframe(results_df, use_container_width=True, height=400)

                # Download results
                st.subheader("📥 Download Research Results")
                csv_data = results_df.to_csv(index=False)

                col_down1, col_down2 = st.columns(2)
                with col_down1:
                    st.download_button(
                        label="📄 Download Research Results CSV",
                        data=csv_data,
                        file_name=f"business_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

                with col_down2:
                    # Create enhanced dataset
                    if 'business_name' in results_df.columns:
                        try:
                            results_df_unique = results_df.drop_duplicates(subset=['business_name'], keep='first')
                            research_mapping = results_df_unique.set_index('business_name')[['phone', 'email', 'website', 'address']].to_dict('index')
                            
                            enhanced_df = research_df.copy()
                            enhanced_df['research_phone'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('phone', ''))
                            enhanced_df['research_email'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('email', ''))
                            enhanced_df['research_website'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('website', ''))
                            enhanced_df['research_address'] = enhanced_df[selected_column].map(lambda x: research_mapping.get(x, {}).get('address', ''))

                            enhanced_csv = enhanced_df.to_csv(index=False)

                            st.download_button(
                                label="🔗 Download Enhanced Dataset",
                                data=enhanced_csv,
                                file_name=f"enhanced_dataset_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                mime="text/csv",
                                help="Original data + research results combined"
                            )
                        except Exception as e:
                            st.warning(f"Could not create enhanced dataset: {e}")

                st.balloons()
                st.success(f"🎉 Successfully researched {summary['successful']} businesses!")

            else:
                st.error("❌ No research results obtained. Please check your API configuration and try again.")

        except Exception as e:
            st.error(f"❌ Research error: {str(e)}")


def test_apis(groq_key, tavily_key):
    """Test both APIs before starting research"""
    try:
        # Test Groq
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": "Say 'API test successful'"}],
                "max_tokens": 10,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return False, f"Groq API error: {response.status_code}"

        # Test Tavily
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_key,
                "query": "test search",
                "max_results": 1
            },
            timeout=30
        )
        
        if response.status_code != 200:
            return False, f"Tavily API error: {response.status_code}"

        return True, "Both APIs are working correctly"
        
    except Exception as e:
        return False, f"API test failed: {str(e)}"


def research_businesses_simplified(business_names, groq_key, tavily_key, progress_bar, status_text):
    """Simplified business research function"""
    
    results = []
    total_businesses = len(business_names)
    
    for i, business_name in enumerate(business_names):
        try:
            progress = 20 + (i / total_businesses) * 70  # Progress from 20% to 90%
            progress_bar.progress(int(progress))
            status_text.info(f"🔍 Researching {i+1}/{total_businesses}: {business_name}")
            
            # Search for business information
            search_results = search_business_info(business_name, tavily_key)
            
            if search_results:
                # Extract contact info using Groq
                contact_info = extract_contacts_with_groq(business_name, search_results, groq_key)
                results.append(contact_info)
            else:
                # Create manual fallback
                results.append(create_manual_fallback(business_name))
                
            # Small delay to avoid rate limits
            import time
            time.sleep(1)
            
        except Exception as e:
            st.error(f"Error researching {business_name}: {str(e)}")
            results.append(create_manual_fallback(business_name))
    
    # Convert results to DataFrame
    results_df = pd.DataFrame(results)
    
    # Create summary
    total_processed = len(results)
    successful = len([r for r in results if r.get('status') == 'success'])
    manual_required = total_processed - successful
    
    summary = {
        'total_processed': total_processed,
        'successful': successful,
        'manual_required': manual_required,
        'success_rate': (successful / total_processed * 100) if total_processed > 0 else 0
    }
    
    return results_df, summary


def search_business_info(business_name, tavily_key):
    """Search for business information using Tavily"""
    try:
        response = requests.post(
            "https://api.tavily.com/search",
            headers={"Content-Type": "application/json"},
            json={
                "api_key": tavily_key,
                "query": f"{business_name} contact information phone email address website",
                "max_results": 3,
                "search_depth": "basic"
            },
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return data.get('results', [])
        else:
            return []
            
    except Exception as e:
        print(f"Tavily search error: {e}")
        return []


def extract_contacts_with_groq(business_name, search_results, groq_key):
    """Extract contact information using Groq AI"""
    try:
        # Format search results for Groq
        results_text = format_search_results(search_results)
        
        prompt = f"""Extract contact information for the business "{business_name}" from the following search results:

{results_text}

Extract and return ONLY the following information in this exact format:
BUSINESS_NAME: {business_name}
PHONE: [phone number if found, or "Not found"]
EMAIL: [email address if found, or "Not found"]
WEBSITE: [website URL if found, or "Not found"]
ADDRESS: [business address if found, or "Not found"]
CONFIDENCE: [rate 1-10 how confident you are this information is correct]

Rules:
- Only extract clearly stated information
- Do not make up or guess information
- Keep the exact format shown above
"""
        
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {groq_key}",
                "Content-Type": "application/json"
            },
            json={
                "model": "llama-3.3-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 500,
                "temperature": 0.1
            },
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            extracted_info = result['choices'][0]['message']['content']
            
            # Parse the extracted information
            return parse_extracted_info(business_name, extracted_info)
        else:
            return create_manual_fallback(business_name)
            
    except Exception as e:
        print(f"Groq extraction error: {e}")
        return create_manual_fallback(business_name)


def format_search_results(search_results):
    """Format search results for AI processing"""
    formatted_results = []
    
    for i, result in enumerate(search_results[:3], 1):
        formatted_result = f"""
        RESULT {i}:
        Title: {result.get('title', 'No title')}
        URL: {result.get('url', 'No URL')}
        Content: {result.get('content', 'No content')[:400]}...
        """
        formatted_results.append(formatted_result)
    
    return '\n'.join(formatted_results)


def parse_extracted_info(business_name, extracted_info):
    """Parse the extracted information into a structured format"""
    
    info_dict = {
        'business_name': business_name,
        'phone': extract_field_value(extracted_info, 'PHONE:'),
        'email': extract_field_value(extracted_info, 'EMAIL:'),
        'website': extract_field_value(extracted_info, 'WEBSITE:'),
        'address': extract_field_value(extracted_info, 'ADDRESS:'),
        'confidence': extract_field_value(extracted_info, 'CONFIDENCE:'),
        'status': 'success',
        'research_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Groq + Tavily'
    }
    
    return info_dict


def extract_field_value(text, field_name):
    """Extract field value from formatted text"""
    try:
        lines = text.split('\n')
        for line in lines:
            if line.strip().startswith(field_name):
                value = line.replace(field_name, '').strip()
                return value if value and value not in ["Not found", ""] else ""
        return ""
    except:
        return ""


def create_manual_fallback(business_name):
    """Create fallback result when automated research fails"""
    
    return {
        'business_name': business_name,
        'phone': 'Research required',
        'email': 'Research required',
        'website': 'Research required',
        'address': 'Research required',
        'confidence': '1',
        'status': 'manual_required',
        'research_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'method': 'Manual research needed'
    }
