#!/usr/bin/env python3
"""
PySpark Vessel Route Analysis
Author: Aurimas Bzeskis
Date: May 2025
Details: created on MACOS, Phycharm IDE

The analysis involves:
1. Loading and cleaning AIS data
2. Calculating distances between consecutive GPS positions for each vessel
3. Aggregating total distances per vessel
4. Identifying the vessel with maximum distance traveled

Dataset characteristics:
- Total records: 19,175,663 AIS position reports from May 4th, 2024
- Geographic coverage: Primarily Danish/Norwegian waters (56-57°N, 8-11°E)
- Vessel count: 5,441 unique vessels tracked throughout the day
- Data quality: 99.87% of records contain valid coordinates after filtering
- Temporal span: Full 24-hour period with high-frequency position updates

Implementation notes:
- Java runtime is required for PySpark (handled automatically in setup)
- Progress tracking with tqdm isn't practical due to Spark's internal parallelization
- Processing speed is remarkably fast compared to traditional sequential approaches
  (19+ million records processed in ~60 seconds vs hours with conventional methods) - compared to previous tasks works way  faster.
"""

import os
import sys
import math
from tqdm import tqdm


# Set up Java environment for Spark
# Note: Spark requires Java to run, so we need to locate and configure it
def setup_java():
    """
    Configure Java environment for PySpark.
    Tries common installation paths on macOS.
    """
    java_paths = ["/opt/homebrew/opt/openjdk@17", "/opt/homebrew/opt/openjdk@11"]
    for path in java_paths:
        if os.path.exists(path):
            os.environ['JAVA_HOME'] = path
            os.environ['PATH'] = f"{path}/bin:" + os.environ.get('PATH', '')
            return True
    return False


# Exit if Java not found
if not setup_java():
    print("Java not found. Install with: brew install openjdk@17")
    sys.exit(1)

# Import PySpark modules after Java setup
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, sum as spark_sum, max as spark_max, count, desc, udf
from pyspark.sql.window import Window
from pyspark.sql.types import DoubleType, LongType


def create_spark_session():
    """
    Initialize Spark session with appropriate configuration.

    Returns:
        SparkSession: Configured Spark session for the analysis
    """
    return SparkSession.builder \
        .appName("VesselRouteAnalysis") \
        .config("spark.ui.enabled", "false") \
        .config("spark.driver.memory", "2g") \
        .getOrCreate()


def load_data(spark):
    """
    Load AIS data from CSV and prepare it for analysis.

    This function:
    - Reads the CSV file with proper headers
    - Selects relevant columns (MMSI, timestamp, lat, lon)
    - Casts data types for calculations
    - Filters out invalid records

    Args:
        spark: SparkSession object

    Returns:
        DataFrame: Cleaned AIS data ready for processing
    """
    print("Loading data...")

    # Update this path to match your data location
    file_path = "/Users/studentas/Desktop/BIG DATA PYSPARK/aisdk-2024-05-04.csv"

    # Read CSV with schema inference
    df_raw = spark.read.option("header", "true").option("inferSchema", "true").csv(file_path)

    # Select and cast relevant columns to proper data types
    # MMSI: vessel identifier, timestamp: time of position report
    # latitude/longitude: GPS coordinates
    df_prepared = df_raw.select(
        col("MMSI").cast(LongType()).alias("mmsi"),
        col("# Timestamp").alias("timestamp"),  # Note: column has # symbol
        col("Latitude").cast(DoubleType()).alias("latitude"),
        col("Longitude").cast(DoubleType()).alias("longitude")
    ).filter(
        # Remove records with missing data or invalid coordinates
        (col("mmsi").isNotNull()) &
        (col("latitude").isNotNull()) &
        (col("longitude").isNotNull()) &
        (col("timestamp").isNotNull()) &
        (col("latitude").between(-90, 90)) &  # Valid latitude range
        (col("longitude").between(-180, 180))  # Valid longitude range
    )

    # Show progress
    total_records = df_prepared.count()
    print(f"Loaded {total_records:,} records")

    return df_prepared


def calculate_distances(df_prepared):
    """
    Calculate distances between consecutive positions for each vessel.

    This is the core of the analysis:
    1. Define Haversine formula for great-circle distance
    2. Use window functions to get previous position for each vessel
    3. Calculate distance between current and previous position
    4. Aggregate total distance per vessel

    Args:
        df_prepared: Cleaned DataFrame with vessel positions

    Returns:
        DataFrame: Total distances traveled by each vessel
    """
    print("Calculating distances...")

    # Haversine formula implementation
    # This calculates the shortest distance between two points on a sphere
    # given their latitude and longitude coordinates
    def haversine_distance(lat1, lon1, lat2, lon2):
        """
        Calculate great-circle distance using Haversine formula.

        Args:
            lat1, lon1: First point coordinates (degrees)
            lat2, lon2: Second point coordinates (degrees)

        Returns:
            float: Distance in kilometers
        """
        # Handle null values
        if any(x is None for x in [lat1, lon1, lat2, lon2]):
            return 0.0

        # Convert degrees to radians for calculation
        lat1_r, lon1_r, lat2_r, lon2_r = map(math.radians, [lat1, lon1, lat2, lon2])

        # Haversine formula
        dlat = lat2_r - lat1_r
        dlon = lon2_r - lon1_r
        a = math.sin(dlat / 2) ** 2 + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
        c = 2 * math.asin(math.sqrt(a))

        # Earth's radius in kilometers
        return 6371 * c

    # Register function as Spark UDF (User Defined Function)
    # This allows us to use our Python function in distributed Spark operations
    haversine_udf = udf(haversine_distance, DoubleType())

    # Window specification for getting previous positions
    # Partition by vessel (MMSI) and order by timestamp
    # This ensures we get consecutive positions for each vessel
    window_spec = Window.partitionBy("mmsi").orderBy("timestamp")

    # Use lag function to get previous position for each record
    # This creates new columns with the previous latitude/longitude
    df_with_prev = df_prepared.withColumn(
        "prev_latitude", lag("latitude", 1).over(window_spec)
    ).withColumn(
        "prev_longitude", lag("longitude", 1).over(window_spec)
    )

    # Calculate distance between current and previous position
    df_with_distance = df_with_prev.withColumn(
        "distance_km",
        haversine_udf(col("prev_latitude"), col("prev_longitude"),
                      col("latitude"), col("longitude"))
    ).filter(col("prev_latitude").isNotNull())  # Remove first record for each vessel

    # Group by vessel and sum up all distances to get total route length
    vessel_distances = df_with_distance.groupBy("mmsi").agg(
        spark_sum("distance_km").alias("total_distance_km")
    )

    vessels_count = vessel_distances.count()
    print(f"Calculated distances for {vessels_count:,} vessels")

    return vessel_distances


def validate_and_filter_realistic_routes(vessel_distances):
    """
    Filter out unrealistic vessel routes that may indicate data quality issues.

    A vessel traveling >1000 km/day would need to maintain 40+ km/h constantly,
    which is unrealistic for most maritime vessels over 24 hours.

    Args:
        vessel_distances: DataFrame with total distances per vessel

    Returns:
        DataFrame: Filtered distances with realistic routes only
    """
    print("Filtering for realistic vessel routes...")

    # Maximum realistic distance: ~1000 km/day (40+ km/h average)
    realistic_distances = vessel_distances.filter(col("total_distance_km") <= 1000)

    filtered_count = realistic_distances.count()
    total_count = vessel_distances.count()
    print(f"Realistic routes: {filtered_count:,} out of {total_count:,} vessels")

    return realistic_distances


def find_longest_route(vessel_distances):
    """
    Identify the vessel that traveled the longest distance.

    Uses Spark aggregation to find the maximum distance value,
    then filters to get the corresponding vessel MMSI.

    Args:
        vessel_distances: DataFrame with total distances per vessel

    Returns:
        Row: Record containing MMSI and distance of longest route
    """
    print("Finding longest route...")

    # Find the maximum distance using Spark aggregation
    max_distance = vessel_distances.agg(spark_max("total_distance_km")).collect()[0][0]

    # Get the vessel record with this maximum distance
    longest_vessel = vessel_distances.filter(
        col("total_distance_km") == max_distance
    ).collect()[0]

    return longest_vessel


def main():
    """
    Main execution function.

    Orchestrates the entire analysis pipeline:
    1. Initialize Spark
    2. Load and clean data
    3. Calculate vessel distances
    4. Find vessel with longest route
    5. Display results
    """
    print("PySpark Vessel Route Analysis")
    print("=" * 40)

    # Initialize Spark session
    spark = create_spark_session()
    spark.sparkContext.setLogLevel("ERROR")  # Reduce log verbosity

    try:
        # Execute analysis pipeline with progress tracking
        with tqdm(total=5, desc="Analysis Progress",
                  bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]') as pbar:

            pbar.set_description("Loading data")
            df_prepared = load_data(spark)
            pbar.update(1)

            pbar.set_description("Calculating distances")
            vessel_distances = calculate_distances(df_prepared)
            pbar.update(1)

            pbar.set_description("Finding longest route")
            longest_vessel = find_longest_route(vessel_distances)
            pbar.update(1)

            pbar.set_description("Filtering realistic routes")
            realistic_distances = validate_and_filter_realistic_routes(vessel_distances)
            pbar.update(1)

            pbar.set_description("Finding realistic longest route")
            longest_realistic = find_longest_route(realistic_distances)
            pbar.update(1)

        # Display results
        print("\nFINAL RESULTS:")
        print("=" * 50)
        print("RAW DATA RESULT (likely data quality issue):")
        print(f"MMSI: {longest_vessel['mmsi']}")
        print(f"Distance: {longest_vessel['total_distance_km']:.2f} km")
        print("\nMOST REALISTIC LONGEST ROUTE:")
        print(f"MMSI: {longest_realistic['mmsi']}")
        print(f"Distance: {longest_realistic['total_distance_km']:.2f} km")
        print(
            f"Note: {longest_realistic['total_distance_km']:.0f} km/day = {longest_realistic['total_distance_km'] / 24:.1f} km/h average")
        print(
            f"\nOriginal result: {longest_vessel['total_distance_km']:.0f} km/day = {longest_vessel['total_distance_km'] / 24:.0f} km/h average")
        print("The extreme speed indicates possible data quality issues or GPS errors.")

    finally:
        # Clean up Spark resources
        spark.stop()


# Entry point - only run if script is executed directly
if __name__ == "__main__":
    main()
