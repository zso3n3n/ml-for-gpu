import os
import zipfile
import pandas as pd
import json
import re

def parse_homberger_file(file_path):
    """
    Parse a Homberger VRPTW instance file (.txt format) and return structured data.
    
    Args:
        file_path: Path to the .txt file
        
    Returns:
        tuple: (customers_df, params_dict)
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    
    # Parse header information
    # Line 1: Instance name
    instance_name = lines[0]
    
    # Line 5: NUMBER CAPACITY
    header_parts = lines[4].split()
    num_customers = int(header_parts[0]) - 1  # Exclude depot
    vehicle_capacity = int(header_parts[1])
    
    # Extract customer data starting from line 10 (0-indexed line 9)
    customers = []
    depot = None
    
    for i in range(9, 9 + int(header_parts[0])):  # Include depot + customers
        if i < len(lines):
            parts = lines[i].split()
            customer_data = {
                'customer_id': int(parts[0]),
                'x': float(parts[1]),
                'y': float(parts[2]),
                'demand': int(parts[3]),
                'tw_start': int(parts[4]),
                'tw_end': int(parts[5]),
                'service_time': int(parts[6])
            }
            
            if customer_data['customer_id'] == 0:
                depot = customer_data
            else:
                customers.append(customer_data)
    
    # Create DataFrames with appropriate dtypes
    customers_df = pd.DataFrame(customers).astype({
        'customer_id': 'int32',
        'x': 'float32',
        'y': 'float32',
        'demand': 'int16',
        'tw_start': 'int32',
        'tw_end': 'int32',
        'service_time': 'int16'
    })
    
    # Estimate number of vehicles (heuristic: total demand / capacity * 1.2)
    total_demand = customers_df['demand'].sum()
    estimated_vehicles = max(1, int((total_demand / vehicle_capacity) * 1.2))
    
    # Create parameters dictionary
    params = {
        'instance': instance_name,
        'K': estimated_vehicles,
        'Q': vehicle_capacity,
        'depot': {
            'x': float(depot['x']) if depot else 50.0,
            'y': float(depot['y']) if depot else 50.0,
            'tw_start': int(depot['tw_start']) if depot else 0,
            'tw_end': int(depot['tw_end']) if depot else 1000,
            'service_time': int(depot['service_time']) if depot else 0
        }
    }
    
    return customers_df, params

def convert_homberger_to_parquet(zip_path, instance_pattern="rc2.*\\.txt", output_dir="../data"):
    """
    Extract a Homberger instance from ZIP, convert to Parquet + JSON format.
    
    Args:
        zip_path: Path to the ZIP file containing instances
        instance_pattern: Regex pattern to match instance files
        output_dir: Directory to save output files
        
    Returns:
        tuple: (customers_parquet_path, params_json_path)
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        # Find matching instance files
        matching_files = [f for f in zip_ref.namelist() if re.match(instance_pattern, os.path.basename(f))]
        
        if not matching_files:
            raise ValueError(f"No files matching pattern '{instance_pattern}' found in {zip_path}")
        
        # Use the first matching file
        instance_file = matching_files[0]
        print(f"Converting instance: {instance_file}")
        
        # Extract to temporary location
        temp_path = os.path.join(output_dir, "temp_instance.txt")
        with zip_ref.open(instance_file) as source:
            with open(temp_path, 'wb') as target:
                target.write(source.read())
        
        try:
            # Parse the instance
            customers_df, params = parse_homberger_file(temp_path)
            
            # Generate output filenames
            instance_name = params['instance'].replace('.', '_').replace(' ', '_')
            customers_path = os.path.join(output_dir, f"{instance_name}_customers.parquet")
            params_path = os.path.join(output_dir, f"{instance_name}_params.json")
            
            # Save to Parquet and JSON
            customers_df.to_parquet(customers_path, index=False)
            
            with open(params_path, 'w') as f:
                json.dump(params, f, indent=2)
            
            print(f"✅ Converted successfully:")
            print(f"   Customers: {customers_path} ({len(customers_df)} customers)")
            print(f"   Parameters: {params_path}")
            print(f"   Vehicles: {params['K']}, Capacity: {params['Q']}")
            
            return customers_path, params_path
            
        finally:
            # Clean up temp file
            if os.path.exists(temp_path):
                os.remove(temp_path)

def main():
    """Example usage of the converter"""
    # Look for RC2 ZIP files in the data directory
    data_dir = "../data"
    zip_files = [f for f in os.listdir(data_dir) if f.endswith('.zip') and 'rc2' in f.lower()]
    
    if not zip_files:
        print("No RC2 ZIP files found in data directory")
        return
    
    # Convert the first RC2 file found
    zip_path = os.path.join(data_dir, zip_files[0])
    print(f"Converting from: {zip_path}")
    
    try:
        customers_path, params_path = convert_homberger_to_parquet(
            zip_path, 
            instance_pattern=r"rc2.*\.txt",
            output_dir=data_dir
        )
        print(f"\n🎉 Conversion complete!")
        
        # Display sample data
        customers_df = pd.read_parquet(customers_path)
        print(f"\nSample customer data:")
        print(customers_df.head())
        
        with open(params_path, 'r') as f:
            params = json.load(f)
        print(f"\nInstance parameters:")
        print(json.dumps(params, indent=2))
        
    except Exception as e:
        print(f"❌ Error during conversion: {e}")

if __name__ == "__main__":
    main()
