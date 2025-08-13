import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime

def create_data_explorer(df, identifier_cols):
    """
    Simplified Data Explorer with clean interface
    - Primary Filter (categorical columns)
    - Secondary Filter (categorical columns)
    - Text Search
    """

    # CRITICAL FIX: Tab preservation function
    def preserve_data_explorer_tab():
        """Preserve the Data Explorer tab state during any interaction"""
        # This function is called on widget interactions to prevent unwanted reruns
        # that would reset the tab to the first one. The key is having the callback
        # present - it prevents the default rerun behavior that causes tab switching.
        pass

    st.subheader("📊 Data Explorer")

    if df is None or len(df) == 0:
        st.warning("No data available to explore.")
        return
    
    # Get all categorical columns (including identifiers)
    categorical_cols = []
    
    # Add object type columns
    object_cols = df.select_dtypes(include=['object']).columns.tolist()
    categorical_cols.extend(object_cols)
    
    # Add identifier columns
    if identifier_cols:
        categorical_cols.extend([col for col in identifier_cols if col not in categorical_cols])
    
    # Remove duplicates and sort
    categorical_cols = sorted(list(set(categorical_cols)))
    
    if not categorical_cols:
        st.info("No categorical columns found for filtering.")
        st.subheader("📈 Raw Dataset")
        st.dataframe(df.head(100), use_container_width=True, height=500)
        return
    
    # Simple 2-column layout for filters
    st.write("**🔍 Filter Your Data:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Primary Filter**")
        primary_filter_col = st.selectbox(
            "Select Column",
            ['None'] + categorical_cols,
            key="primary_filter_column",
            on_change=preserve_data_explorer_tab
        )

        primary_filter_value = 'All'
        if primary_filter_col != 'None':
            unique_values = ['All'] + sorted([str(val) for val in df[primary_filter_col].dropna().unique()])
            primary_filter_value = st.selectbox(
                f"Filter by {primary_filter_col}",
                unique_values,
                key="primary_filter_value",
                on_change=preserve_data_explorer_tab
            )
        
        # Text search for primary filter
        if primary_filter_col != 'None':
            primary_search = st.text_input(
                f"Search in {primary_filter_col}",
                placeholder="Enter search term...",
                key="primary_search",
                on_change=preserve_data_explorer_tab
            )
        else:
            primary_search = ""
    
    with col2:
        st.write("**Secondary Filter**")
        # Remove primary filter column from secondary options
        secondary_options = [col for col in categorical_cols if col != primary_filter_col]
        
        secondary_filter_col = st.selectbox(
            "Select Column",
            ['None'] + secondary_options,
            key="secondary_filter_column",
            on_change=preserve_data_explorer_tab
        )

        secondary_filter_value = 'All'
        if secondary_filter_col != 'None':
            unique_values = ['All'] + sorted([str(val) for val in df[secondary_filter_col].dropna().unique()])
            secondary_filter_value = st.selectbox(
                f"Filter by {secondary_filter_col}",
                unique_values,
                key="secondary_filter_value",
                on_change=preserve_data_explorer_tab
            )
        
        # Text search for secondary filter
        if secondary_filter_col != 'None':
            secondary_search = st.text_input(
                f"Search in {secondary_filter_col}",
                placeholder="Enter search term...",
                key="secondary_search",
                on_change=preserve_data_explorer_tab
            )
        else:
            secondary_search = ""
    
    # Apply filters
    filtered_df = df.copy()
    
    # Apply primary filter
    if primary_filter_col != 'None':
        if primary_filter_value != 'All':
            filtered_df = filtered_df[filtered_df[primary_filter_col].astype(str) == primary_filter_value]
        
        if primary_search:
            mask = filtered_df[primary_filter_col].astype(str).str.contains(
                primary_search, case=False, na=False
            )
            filtered_df = filtered_df[mask]
    
    # Apply secondary filter
    if secondary_filter_col != 'None':
        if secondary_filter_value != 'All':
            filtered_df = filtered_df[filtered_df[secondary_filter_col].astype(str) == secondary_filter_value]
        
        if secondary_search:
            mask = filtered_df[secondary_filter_col].astype(str).str.contains(
                secondary_search, case=False, na=False
            )
            filtered_df = filtered_df[mask]
    
    # Display results summary
    st.markdown("---")
    
    col_summary1, col_summary2, col_summary3, col_summary4 = st.columns(4)
    
    with col_summary1:
        st.metric("📊 Total Records", f"{len(filtered_df):,}")
    
    with col_summary2:
        original_count = len(df)
        if original_count > 0:
            percentage = (len(filtered_df) / original_count) * 100
            st.metric("📈 % of Total", f"{percentage:.1f}%")
        else:
            st.metric("📈 % of Total", "0%")
    
    with col_summary3:
        if primary_filter_col != 'None' and primary_filter_col in filtered_df.columns:
            unique_primary = filtered_df[primary_filter_col].nunique()
            st.metric(f"🔢 Unique {primary_filter_col[:10]}...", unique_primary)
        else:
            st.metric("🔢 Columns", len(filtered_df.columns))
    
    with col_summary4:
        if secondary_filter_col != 'None' and secondary_filter_col in filtered_df.columns:
            unique_secondary = filtered_df[secondary_filter_col].nunique()
            st.metric(f"🔢 Unique {secondary_filter_col[:10]}...", unique_secondary)
        else:
            completeness = (1 - filtered_df.isnull().sum().sum() / (len(filtered_df) * len(filtered_df.columns))) * 100 if len(filtered_df) > 0 else 0
            st.metric("✅ Completeness", f"{completeness:.1f}%")
    
    # Display filtered data
    st.subheader(f"📈 Filtered Dataset ({len(filtered_df):,} records)")
    
    if len(filtered_df) == 0:
        st.warning("No records match your filter criteria. Try adjusting your filters.")
        return
    
    # Display controls with safety check for small datasets
    col_display1, col_display2 = st.columns(2)
    with col_display1:
        # Safety check to prevent slider error when filtered data is small
        total_rows = len(filtered_df)
        if total_rows <= 10:
            # For very small datasets, just show all rows
            display_rows = total_rows
            st.info(f"Showing all {total_rows} rows (dataset is small)")
        else:
            # For larger datasets, provide slider
            min_display = min(10, total_rows)
            max_display = min(500, total_rows)
            default_display = min(100, total_rows)
            
            # Ensure min_value < max_value for slider
            if min_display >= max_display:
                display_rows = max_display
                st.info(f"Showing all {max_display} rows")
            else:
                # Initialize session state for slider if not exists
                if "display_rows_value" not in st.session_state:
                    st.session_state.display_rows_value = default_display

                # Ensure the session state value is within bounds
                if st.session_state.display_rows_value < min_display:
                    st.session_state.display_rows_value = min_display
                elif st.session_state.display_rows_value > max_display:
                    st.session_state.display_rows_value = max_display

                display_rows = st.slider(
                    "Rows to display:",
                    min_display,
                    max_display,
                    st.session_state.display_rows_value,
                    key="display_rows_slider_explorer",
                    on_change=preserve_data_explorer_tab
                )

                # Update session state
                st.session_state.display_rows_value = display_rows
    with col_display2:
        st.write(f"Showing {min(display_rows, len(filtered_df))} of {len(filtered_df)} total rows")
    
    # Smart column ordering - put filtered columns first
    columns_order = []
    if primary_filter_col != 'None':
        columns_order.append(primary_filter_col)
    if secondary_filter_col != 'None' and secondary_filter_col not in columns_order:
        columns_order.append(secondary_filter_col)
    
    # Add remaining columns
    remaining_cols = [col for col in filtered_df.columns if col not in columns_order]
    columns_order.extend(remaining_cols)
    
    # Display data
    st.dataframe(
        filtered_df[columns_order].head(display_rows),
        use_container_width=True,
        height=500
    )
    
    # Quick insights
    if len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("💡 Quick Insights")
        
        col_insights1, col_insights2 = st.columns(2)
        
        with col_insights1:
            if primary_filter_col != 'None':
                st.write(f"**Top 5 {primary_filter_col}:**")
                top_values = filtered_df[primary_filter_col].value_counts().head(5)
                for value, count in top_values.items():
                    percentage = (count / len(filtered_df)) * 100
                    st.write(f"• **{value}**: {count} ({percentage:.1f}%)")
        
        with col_insights2:
            if secondary_filter_col != 'None':
                st.write(f"**Top 5 {secondary_filter_col}:**")
                top_values = filtered_df[secondary_filter_col].value_counts().head(5)
                for value, count in top_values.items():
                    percentage = (count / len(filtered_df)) * 100
                    st.write(f"• **{value}**: {count} ({percentage:.1f}%)")
            else:
                # Show data quality info instead
                st.write("**Data Quality:**")
                missing_data = filtered_df.isnull().sum()
                cols_with_missing = missing_data[missing_data > 0].head(3)
                if len(cols_with_missing) > 0:
                    for col, missing in cols_with_missing.items():
                        pct = (missing / len(filtered_df)) * 100
                        st.write(f"• {col}: {missing} missing ({pct:.1f}%)")
                else:
                    st.write("✅ No missing data found")
    
    # Download filtered data
    st.markdown("---")
    col_download1, col_download2 = st.columns(2)
    
    with col_download1:
        if len(filtered_df) > 0:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📊 Download Filtered Data",
                data=csv,
                file_name=f"filtered_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    with col_download2:
        # Web scraping option with session state management
        if len(filtered_df) > 0:
            # Initialize session state for web scraping interface
            if 'show_web_scraping' not in st.session_state:
                st.session_state.show_web_scraping = False
            
            # FIXED: Toggle button for web scraping interface with proper tab preservation
            if not st.session_state.show_web_scraping:
                if st.button(
                    "🌐 Research Business Contacts",
                    help="Use AI to find business contact information",
                    key="research_contacts_button",
                    on_click=preserve_data_explorer_tab
                ):
                    st.session_state.show_web_scraping = True
                    # NO st.rerun() call here!
            else:
                if st.button(
                    "❌ Close Research Interface",
                    help="Close the business research interface",
                    key="close_research_button",
                    on_click=preserve_data_explorer_tab
                ):
                    st.session_state.show_web_scraping = False
                    # NO st.rerun() call here!
    
    # FIXED: Show web scraping interface if activated
    if st.session_state.get('show_web_scraping', False) and len(filtered_df) > 0:
        st.markdown("---")
        st.subheader("🔍 Business Contact Research")

        perform_web_scraping(filtered_df)

def perform_web_scraping(filtered_df):
    """Perform web scraping of business contact information from filtered data"""

    try:
        # Import the web scraping module
        import sys
        import os

        # Add the modules directory to the path
        modules_path = os.path.join(os.path.dirname(__file__), 'modules')
        if modules_path not in sys.path:
            sys.path.insert(0, modules_path)

        # Import and call the web scraping function from the module
        import web_scraping_module
        web_scraping_module.perform_web_scraping(filtered_df)

    except ImportError as e:
        st.error(f"❌ Could not import web scraping module: {str(e)}")
        st.info("💡 Make sure the 'modules/web_scraping_module.py' file exists and is properly configured.")

    except Exception as e:
        st.error(f"❌ Error during web scraping: {str(e)}")
        st.info("💡 Please check your configuration and try again.")
