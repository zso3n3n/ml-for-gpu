import pandas as pd

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
            continue
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