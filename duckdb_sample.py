import duckdb
import pandas as pd

INPUT_FILE = "data/training_data.parquet"


def diagnose_and_sample():
    con = duckdb.connect()

    print("--- 1. DIAGNOSTIC: Checking Time Range ---")
    try:
        # Check the actual time range covered by the dataset
        stats_query = f"""
        SELECT
            count(*) as total_rows,
            min(event_time) as start_time,
            max(event_time) as end_time,
            (max(event_time) - min(event_time)) as duration_interval
        FROM '{INPUT_FILE}'
        """
        stats = con.sql(stats_query).df()
        print(stats.to_string(index=False))

        # Verify the dataset has reasonable duration (expected: 28 days, 2025-06-24 to 2025-07-22)
        duration = stats.iloc[0]["duration_interval"]
        if pd.isnull(duration) or duration.days < 1:
            print("\n[WARNING] Dataset duration is less than 1 day!")
            print("Check your aggregation script: did it detect input files correctly?")
        else:
            print(f"\n[OK] Dataset covers {duration.days} days.")

    except Exception as e:
        print(f"Error reading file: {e}")
        return

    print("\n--- 2. SAMPLING: True Global Random Sequences ---")

    total_rows = stats.iloc[0]["total_rows"]
    if total_rows == 0:
        return

    for i in range(1, 2):
        print(f"### Random Sequence {i} ###")

        # This query forces a full table scan by ordering by random().
        # This ensures we pick rows uniformly across the entire time range (2025-06-24 to 2025-07-22).
        query = f"""
        WITH random_sample AS (
            SELECT
                event_time,
                msm_id,
                src_addr,
                dst_addr,
                rtt,
                packet_error_count as err,
                random() as rand
            FROM '{INPUT_FILE}'
            ORDER BY rand
            LIMIT 25
        )
        SELECT event_time, msm_id, src_addr, dst_addr, rtt, err
        FROM random_sample
        ORDER BY event_time ASC
        """

        df = con.sql(query).df()

        # Clean formatting
        pd.set_option("display.max_colwidth", 40)
        pd.set_option("display.width", 1000)

        print(df.to_string(index=False))
        print("\n" + "-" * 100 + "\n")


if __name__ == "__main__":
    diagnose_and_sample()
