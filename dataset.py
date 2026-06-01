"""
Dataset Module for Online Retail II
Handles data loading and RFM (Recency, Frequency, Monetary) feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Tuple


def load_data(file_path: str = "dataset/online_retail_II.xlsx") -> pd.DataFrame:
    """
    Load the Online Retail II dataset from Excel file.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with all transactions
    """
    print("Loading dataset...")
    
    # Read both sheets (Year 2009-2010 and Year 2010-2011)
    df1 = pd.read_excel(file_path, sheet_name="Year 2009-2010")
    df2 = pd.read_excel(file_path, sheet_name="Year 2010-2011")
    
    # Combine both sheets
    df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"Total records loaded: {len(df):,}")
    print(f"Columns: {list(df.columns)}")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the dataset by removing invalid records.
    
    Args:
        df: Raw DataFrame
        
    Returns:
        Cleaned DataFrame
    """
    print("\nCleaning data...")
    initial_count = len(df)
    
    # Remove rows with missing Customer ID
    df = df.dropna(subset=['Customer ID'])
    
    # Remove cancelled orders (Invoice starts with 'C')
    df = df[~df['Invoice'].astype(str).str.startswith('C')]
    
    # Remove rows with negative or zero quantities
    df = df[df['Quantity'] > 0]
    
    # Remove rows with negative or zero prices
    df = df[df['Price'] > 0]
    
    # Convert Customer ID to integer
    df['Customer ID'] = df['Customer ID'].astype(int)
    
    # Ensure InvoiceDate is datetime
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    
    # Calculate total amount for each transaction line
    df['TotalAmount'] = df['Quantity'] * df['Price']
    
    final_count = len(df)
    print(f"Records removed: {initial_count - final_count:,}")
    print(f"Clean records: {final_count:,}")
    
    return df


def calculate_rfm(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate RFM (Recency, Frequency, Monetary) values for each customer.
    
    RFM Analysis:
    - Recency: Days since last purchase (lower is better)
    - Frequency: Number of unique transactions (higher is better)
    - Monetary: Total spending amount (higher is better)
    
    Args:
        df: Cleaned transaction DataFrame
        
    Returns:
        DataFrame with Customer ID and RFM values
    """
    print("\nCalculating RFM values...")
    
    # Reference date: the day after the last transaction in the dataset
    reference_date = df['InvoiceDate'].max() + pd.Timedelta(days=1)
    print(f"Reference date for Recency: {reference_date.date()}")
    
    # Group by Customer ID and calculate RFM
    rfm = df.groupby('Customer ID').agg({
        'InvoiceDate': lambda x: (reference_date - x.max()).days,  # Recency
        'Invoice': 'nunique',  # Frequency (unique invoices)
        'TotalAmount': 'sum'   # Monetary (total spending)
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
    
    print(f"Number of unique customers: {len(rfm):,}")
    print(f"\nRFM Statistics:")
    print(rfm[['Recency', 'Frequency', 'Monetary']].describe())
    
    return rfm


def get_rfm_data(file_path: str = "dataset/online_retail_II.xlsx") -> pd.DataFrame:
    """
    Main function to load data and calculate RFM values.
    
    Args:
        file_path: Path to the Excel file
        
    Returns:
        DataFrame with RFM values for each customer
    """
    df = load_data(file_path)
    df = clean_data(df)
    rfm = calculate_rfm(df)
    
    return rfm


if __name__ == "__main__":
    # Test the module
    rfm_df = get_rfm_data()
    print("\nSample RFM data:")
    print(rfm_df.head(10))
