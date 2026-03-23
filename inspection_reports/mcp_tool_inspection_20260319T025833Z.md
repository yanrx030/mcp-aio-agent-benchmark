# MCP Tool Inspection Report

- Generated at: `2026-03-19T02:58:33.632545Z`
- Toolset: `toolsets\aio_mcp_toolset_v2.json`
- Collection context: `reddit`
- Safe end date: `2026-03-18`
- Chosen term: `CallToolResult`

## Per-tool Summary

### `get_api_version`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: not tested
- Success args: `{}`
- Success text sample: `1.0.4-api`

### `get_collections`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: not tested
- Success args: `{}`
- Success text sample: `{"collections":["bluesky","gdelt","flickr","mastodon","reddit","twitter","youtube"]}`

### `get_collection_summary`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit"}`
- Failure args: `{}`
- Success text sample: `{"startDate":"2019-10-03","endDate":"2026-03-19","count":7814482}`
- Failure text sample: `Input validation error: 'collection' is a required property`

### `aggregate_by_time`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=400; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "aggregation_level": "month", "startdate": "2026-02-17", "enddate": "2026-03-18", "sentiment": false}`
- Failure args: `{"collection": "reddit", "aggregation_level": "month", "startdate": "2019-01-01", "enddate": "2026-03-18", "sentiment": false}`
- Success text sample: `[{"time":"2026-02","count":53740},{"time":"2026-03","count":73820}]`
- Failure text sample: `Output validation error: {'error': '400 Client Error: Bad Request for url: https://api.aio.eresearch.unimelb.edu.au/analysis/aggregate/collections/reddit/aggregation?aggregationLevel=month&startdate=2019-01-01&enddate=20`

### `aggregate_seasonality`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=400; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "aggregation_level": "dayofweek", "startdate": "2026-02-17", "enddate": "2026-03-18", "sentiment": false}`
- Failure args: `{"collection": "reddit", "aggregation_level": "invalid_level", "startdate": "2026-02-17", "enddate": "2026-03-18", "sentiment": false}`
- Success text sample: `[{"time":"friday","count":16540},{"time":"monday","count":17164},{"time":"saturday","count":16350},{"time":"sunday","count":15143},{"time":"thursday","count":16706},{"time":"tuesday","count":22794},{"time":"wednesday","c`
- Failure text sample: `Output validation error: {'error': '400 Client Error: Bad Request for url: https://api.aio.eresearch.unimelb.edu.au/analysis/aggregate/collections/reddit/seasonality?aggregationLevel=invalid_level&startdate=2026-02-17&en`

### `analyze_terms_in_collection`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "startdate": "2026-03-12", "enddate": "2026-03-18", "limit": 10}`
- Failure args: `{}`
- Success text sample: `{"terms":{"issu":208,"govern":225,"someth":579,"week":703,"lane":112,"area":364,"work":996,"someon":654,"line":224,"amp":1009,"road":376,"way":1094,"http":622,"fuel":492,"price":766,"day":1401,"car":1206,"thing":1253,"ho`
- Failure text sample: `Input validation error: 'collection' is a required property`

### `get_all_terms`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: ok; is_error=False; http_status_code=400; content_type=list; structured_content_type=dict
- Success args: `{"collection": "reddit", "startdate": "2026-03-12", "enddate": "2026-03-18"}`
- Failure args: `{"collection": "reddit", "startdate": "2019-01-01", "enddate": "2026-03-18"}`
- Success text sample: `{"terms":{"issu":208,"govern":225,"someth":579,"week":703,"lane":112,"area":364,"work":996,"someon":654,"line":224,"amp":1009,"road":376,"way":1094,"http":622,"fuel":492,"price":766,"day":1401,"car":1206,"thing":1253,"ho`
- Failure text sample: `{"error":"400 Client Error: Bad Request for url: https://api.aio.eresearch.unimelb.edu.au/analysis/terms/collections/reddit?startdate=2019-01-01&enddate=2026-03-18"}`

### `get_term_daily_counts`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "terms": "CallToolResult", "startdate": "2026-03-12", "enddate": "2026-03-18"}`
- Failure args: `{"collection": "reddit", "terms": 123}`
- Success text sample: `{"CallToolResult":[]}`
- Failure text sample: `Input validation error: 123 is not of type 'string'`

### `get_nlp_terms_for_day`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: ok; is_error=False; http_status_code=404; content_type=list; structured_content_type=dict
- Success args: `{"collection": "reddit", "day": "2026-03-18", "limit": 10}`
- Failure args: `{"collection": "reddit", "day": "1900-01-01"}`
- Success text sample: `{"terms":["space","thank","everyon","issu","hous","lot","fuel","point","anyon","week","area","amp","driver","seat","bu","price","someth","http","hour","kid","work","someon","place","school","car","way","train","year","da`
- Failure text sample: `{"error":"404 Client Error: Not Found for url: https://api.aio.eresearch.unimelb.edu.au/analysis/nlp/collections/reddit/days/1900-01-01/terms"}`

### `get_nlp_term_analysis`
- Success case: ok; is_error=False; http_status_code=404; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "day": "2026-03-18", "term": "CallToolResult", "limit": 10}`
- Failure args: `{"collection": "reddit", "day": "2026-03-18", "term": 123}`
- Success text sample: `{"error":"404 Client Error: Not Found for url: https://api.aio.eresearch.unimelb.edu.au/analysis/nlp/collections/reddit/days/2026-03-18/terms/CallToolResult?limit=10"}`
- Failure text sample: `Input validation error: 123 is not of type 'string'`

### `get_nlp_topics`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Success args: `{"collection": "reddit", "startdate": "2026-03-12", "enddate": "2026-03-18"}`
- Failure args: `{"collection": "reddit", "startdate": "2019-01-01", "enddate": "2026-03-18"}`
- Success text sample: `[{"time":"2026-03-12","topics":[{"size":115,"terms":"peopl,year,time,hous,thing,car,price,fuel,day,road,lane,way,govern,work,someon,amp,http,issu,week,someth,area,bank,citi,place,everyon,cost,home,australia,compani,point`

### `get_topic_groupings`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "startdate": "2026-03-12", "enddate": "2026-03-18", "threshold": 1}`
- Failure args: `{"collection": "reddit", "startdate": "2019-01-01", "enddate": "2026-03-18", "threshold": "bad"}`
- Success text sample: `{"directed":true,"graph":{},"links":[{"common_terms":["work","year","time","day"],"sequences":[],"source":"20260312-0","target":"20260313-0","weight":4},{"common_terms":["thing","govern","year","car","someon","hous","amp`
- Failure text sample: `Input validation error: 'bad' is not of type 'integer'`

### `get_nlp_metadata`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "startdate": "2026-03-12", "enddate": "2026-03-18"}`
- Failure args: `{}`
- Success text sample: `[{"id":"reddit-20260312","corpus":{"start":"2026-03-12T00:00:00+11:00","end":"2026-03-12T23:59:59+11:00","days":1,"ndocuments":3663},"processing":{"start":"2026-03-13T02:00:05+11:00","end":"2026-03-13T02:01:02+11:00","pr`
- Failure text sample: `Input validation error: 'collection' is a required property`

### `text_search`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"collection": "reddit", "query": "text:covid", "limit": 10}`
- Failure args: `{"collection": "reddit", "query": 123}`
- Success text sample: `72248`
- Failure text sample: `Input validation error: 123 is not of type 'string'`

### `generate_chart`
- Success case: ok; is_error=False; http_status_code=None; content_type=list; structured_content_type=dict
- Failure case: mcp_error; is_error=True; http_status_code=None; content_type=list; structured_content_type=None
- Success args: `{"chart": {"type": "bar", "data": {"labels": ["A", "B"], "datasets": [{"label": "Counts", "data": [1, 2]}]}}, "width": 300, "height": 200, "format": "png"}`
- Failure args: `{"chart": "not-an-object"}`
- Success text sample: `{"status":"success","url":"https://quickchart.io/chart/render/zf-1ea81a8d-cd7e-44b0-bd79-512b7d30fcec","chart_config":{"type":"bar","data":{"labels":["A","B"],"datasets":[{"label":"Counts","data":[1,2]}]}},"dimensions":{`
- Failure text sample: `Input validation error: 'not-an-object' is not of type 'object'`
