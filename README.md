***run the project

uv run --env-file .env main.py




Tool: aggregate_by_time
   Returns aggregate counts or sentiment by time.
The date interval cannot be greater than 365 days.
The end date of the interval must not exceed yesterday's date

Params:
- collection: The collection name
- aggregation_level: One of [day, month]
- startdate: The start date (YYYY-MM-DD)
- enddate: The end date (YYYY-MM-DD)
- sentiment: If true, aggregate sentiment instead of counts
- request: The FastMCP request object containing headers.

Response: list of buckets with time period and value(s).