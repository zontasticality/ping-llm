"""Query Modal workspace billing usage.

Usage:
  python scripts/modal_usage.py                  # Last 7 days, daily
  python scripts/modal_usage.py --days 30        # Last 30 days
  python scripts/modal_usage.py --hours 24       # Last 24 hours, hourly
"""

import argparse
from datetime import datetime, timedelta, timezone
from decimal import Decimal

from modal.billing import workspace_billing_report


def main():
    parser = argparse.ArgumentParser(description="Query Modal billing usage")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--days", type=int, default=7, help="Lookback in days (default: 7)")
    group.add_argument("--hours", type=int, help="Lookback in hours (uses hourly resolution)")
    parser.add_argument("--app", type=str, help="Filter to app name (substring match)")
    args = parser.parse_args()

    now = datetime.now(timezone.utc)
    if args.hours:
        if args.hours > 168:
            parser.error("Hourly resolution limited to 7 days (168 hours) by Modal API")
        start = now - timedelta(hours=args.hours)
        resolution = "h"
    else:
        if args.days > 31:
            parser.error("Daily resolution limited to 31 days by Modal API")
        start = now - timedelta(days=args.days)
        resolution = "d"

    rows = workspace_billing_report(start=start, end=now, resolution=resolution)

    if args.app:
        rows = [r for r in rows if args.app.lower() in r["description"].lower()]

    # Aggregate by app
    by_app: dict[str, Decimal] = {}
    total = Decimal("0")
    for r in rows:
        name = r["description"]
        cost = r["cost"]
        by_app[name] = by_app.get(name, Decimal("0")) + cost
        total += cost

    # Print summary
    lookback = f"{args.hours}h" if args.hours else f"{args.days}d"
    print(f"Modal usage (last {lookback}, {resolution}={'hourly' if resolution == 'h' else 'daily'}):\n")

    for name, cost in sorted(by_app.items(), key=lambda x: x[1], reverse=True):
        print(f"  {name:<30s}  ${cost:.2f}")

    print(f"\n  {'TOTAL':<30s}  ${total:.2f}")

    # If hourly, also show recent line items
    if resolution == "h" and rows:
        print(f"\nRecent line items:")
        for r in sorted(rows, key=lambda x: x["interval_start"], reverse=True)[:15]:
            ts = r["interval_start"].strftime("%Y-%m-%d %H:%M")
            print(f"  {ts}  {r['description']:<30s}  ${r['cost']:.4f}")


if __name__ == "__main__":
    main()
