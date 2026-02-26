from datetime import datetime
from dateutil.relativedelta import relativedelta

def get_partitions_from_date(start_date):
    partitions = []

    current = datetime(start_date.year, start_date.month, 1)

    now = datetime.utcnow()
    end = datetime(now.year, now.month, 1)

    while current <= end:
        partitions.append((current.year, f"{current.month:02d}"))
        current += relativedelta(months=1)

    return partitions