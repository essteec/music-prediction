import pandas as pd
import sys

def print_csv_pretty(csv_file, rows=None):
    """
    Print a CSV file in a pretty format to the terminal
    
    Args:
        csv_file: Path to the CSV file
        rows: Number of rows to display (None = all rows)
    """
    try:
        # First, get total row count efficiently (just count lines)
        with open(csv_file, 'r') as f:
            total_rows = sum(1 for line in f) - 1  # -1 for header
        
        # Read the CSV file (only specified number of rows if provided)
        if rows:
            df = pd.read_csv(csv_file, nrows=rows)
        else:
            df = pd.read_csv(csv_file)
        
        # Filter to only show specific columns if they exist
        columns_to_show = ['id', 'popularity', 'genre', 'explicit', 'year', 'valence', 'energy', 'danceability']
        available_columns = [col for col in columns_to_show if col in df.columns]
        if available_columns:
            df = df[available_columns]
        
        # Print basic info
        print(f"\n{'='*80}")
        print(f"File: {csv_file}")
        print(f"{'='*80}")
        if rows and rows < total_rows:
            print(f"Showing: {len(df)} rows | Total in file: {total_rows} | Columns: {len(df.columns)}")
        else:
            print(f"Rows: {len(df)} | Columns: {len(df.columns)}")
        print(f"{'='*80}\n")
        
        # Print the dataframe in a pretty format
        # Set display options for better visibility
        pd.set_option('display.max_columns', None)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.width', None)
        pd.set_option('display.max_colwidth', 50)
        
        print(df.to_string(index=False))
        
        print(f"\n{'='*80}")
        if rows and rows < total_rows:
            print(f"Displayed: {len(df)} of {total_rows} rows")
        else:
            print(f"Total: {len(df)} rows")
        print(f"{'='*80}\n")
        
    except FileNotFoundError:
        print(f"Error: File '{csv_file}' not found!")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

def main():
    """
    Usage: 
        python print_csv.py [csv_file] [rows]
        
    Examples:
        python print_csv.py                          # prints genre_mappings.csv (all rows)
        python print_csv.py genre_mappings.csv       # prints all rows
        python print_csv.py genre_mappings.csv 10    # prints first 10 rows only
        python print_csv.py songs.csv 50             # prints first 50 rows of songs.csv
    """
    # Default to genre_mappings.csv if no argument provided
    csv_file = 'genre_mappings.csv'
    rows = None
    
    if len(sys.argv) > 1:
        csv_file = sys.argv[1]
    
    if len(sys.argv) > 2:
        try:
            rows = int(sys.argv[2])
        except ValueError:
            print(f"Error: rows parameter must be an integer, got '{sys.argv[2]}'")
            sys.exit(1)
    
    print_csv_pretty(csv_file, rows)

if __name__ == "__main__":
    main()
