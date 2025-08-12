import os
import zipfile
import pandas as pd
import json
import re

def parse_homberger_file(file_path):
    """
    Parse a Homberger VRPTW instance file (.txt format) and return structured data.
    """
    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Find the "CUSTOMER" header and its index
    for idx, line in enumerate(lines):
        if line.startswith("CUSTOMER"):
            customer_header_idx = idx
            break
    else:
        raise ValueError("CUSTOMER header not found in file")

    # Parse instance name from first line
    instance_name = lines[0]

    # Parse number of customers and vehicle capacity from the line before "CUSTOMER"
    header_parts = lines[customer_header_idx - 1].split()
    num_customers = int(header_parts[0]) - 1  # Exclude depot
    vehicle_capacity = int(header_parts[1])

    # Parse customer data (lines after "CUSTOMER")
    customers = []
    depot = None
    for line in lines[customer_header_idx + 1:]:
        parts = line.split()
        if len(parts) != 7:
            continue  # skip malformed lines
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

    customers_df = pd.DataFrame(customers).astype({
        'customer_id': 'int32',
        'x': 'float32',
        'y': 'float32',
        'demand': 'int16',
        'tw_start': 'int32',
        'tw_end': 'int32',
        'service_time': 'int16'
    })

    total_demand = customers_df['demand'].sum()
    estimated_vehicles = max(1, int((total_demand / vehicle_capacity) * 1.2))

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

def convert_homberger_to_parquet(zip_path, instance_pattern="rc2.*\\.txt", output_dir="../data/vrptw/homberger"):
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
    """Convert Homberger instances from the organized data structure"""
    
    # Define data paths for the organized structure
    base_data_dir = "../data/vrptw/homberger"
    
    # Available datasets
    datasets = [
        {
            'name': 'C2 Series (200 customers)',
            'zip_path': os.path.join(base_data_dir, "c2/homberger_200_customer_instances.zip"),
            'pattern': r"c2.*\.txt",
            'series': 'c2'
        },
        {
            'name': 'RC2 Series (1000 customers)', 
            'zip_path': os.path.join(base_data_dir, "rc2/homberger_1000_customer_instances.zip"),
            'pattern': r"rc2.*\.txt",
            'series': 'rc2'
        }
    ]
    
    print("🔧 Homberger VRPTW Instance Converter")
    print("=" * 50)
    
    for i, dataset in enumerate(datasets):
        print(f"\n{i+1}. {dataset['name']}")
        print(f"   ZIP: {dataset['zip_path']}")
        print(f"   Available: {'✅' if os.path.exists(dataset['zip_path']) else '❌'}")
    
    # Convert available datasets
    converted_any = False
    
    for dataset in datasets:
        if not os.path.exists(dataset['zip_path']):
            print(f"\n⚠️  Skipping {dataset['name']} - ZIP file not found")
            continue
            
        print(f"\n🔄 Converting {dataset['name']}...")
        
        try:
            # Create series-specific output directory
            output_dir = os.path.join(base_data_dir, dataset['series'])
            
            customers_path, params_path = convert_homberger_to_parquet(
                dataset['zip_path'], 
                instance_pattern=dataset['pattern'],
                output_dir=output_dir
            )
            
            # Display sample data
            customers_df = pd.read_parquet(customers_path)
            print(f"\n📊 Sample customer data from {dataset['series'].upper()}:")
            print(customers_df.head(3))
            
            with open(params_path, 'r') as f:
                params = json.load(f)
            print(f"\n⚙️  Instance parameters:")
            print(f"   Instance: {params['instance']}")
            print(f"   Customers: {len(customers_df)}")
            print(f"   Vehicles: {params['K']}")
            print(f"   Capacity: {params['Q']}")
            print(f"   Depot: ({params['depot']['x']}, {params['depot']['y']})")
            
            converted_any = True
            
        except Exception as e:
            print(f"❌ Error converting {dataset['name']}: {e}")
    
    if converted_any:
        print(f"\n🎉 Conversion complete!")
        print(f"\n📁 To use in notebooks:")
        print(f"   # For C2 series (sample):")
        print(f"   USE_SAMPLE = True")
        print(f"   ")
        print(f"   # For RC2 series (full):")
        print(f"   USE_SAMPLE = False")
    else:
        print(f"\n❌ No datasets were converted. Please check that ZIP files exist in:")
        for dataset in datasets:
            print(f"   {dataset['zip_path']}")

if __name__ == "__main__":
    main()