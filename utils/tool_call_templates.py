# Tool Call Templates

# This file contains example templates for calling each tool defined in the tool_summary.json file.
# Each template includes example parameters for the tool.

TOOLS = {
    "get_api_version": {
        "description": "Returns the semantic version of the API.",
        "example": {}
    },
    "get_collections": {
        "description": "Lists available data collections.",
        "example": {}
    },
    "get_collection_summary": {
        "description": "Returns the number of posts and harvesting start/end dates for a collection.",
        "example": {
            "collection": "reddit"
        }
    },
    "aggregate_by_time": {
        "description": "Returns aggregate counts or sentiment by time.",
        "example": {
            "collection": "reddit",
            "aggregation_level": "month",
            "startdate": "2024-01-01",
            "enddate": "2024-12-31",
            "sentiment": False
        }
    },
    "aggregate_seasonality": {
        "description": "Returns aggregate counts or sentiment by seasonality (day of week or hour of day).",
        "example": {
            "collection": "reddit",
            "aggregation_level": "dayofweek",
            "startdate": "2024-01-01",
            "enddate": "2024-12-31",
            "sentiment": True
        }
    },
    "analyze_terms_in_collection": {
        "description": "Returns an analysis of terms in a collection.",
        "example": {
            "collection": "reddit",
            "startdate": "2024-01-01",
            "enddate": "2024-01-28",
            "limit": 100,
            "bookmark": None
        }
    },
    "get_all_terms": {
        "description": "Returns available stemmed terms and aggregated counts in a date range.",
        "example": {
            "collection": "reddit",
            "startdate": "2024-01-01",
            "enddate": "2024-01-28"
        }
    },
    "get_term_daily_counts": {
        "description": "Returns daily counts for specific terms in a collection.",
        "example": {
            "collection": "reddit",
            "terms": "climate,change",
            "startdate": "2024-01-01",
            "enddate": "2024-01-28"
        }
    },
    "get_nlp_terms_for_day": {
        "description": "Returns NLP terms analysis for a specific date.",
        "example": {
            "collection": "reddit",
            "day": "2024-01-15",
            "limit": 50,
            "bookmark": None
        }
    },
    "get_nlp_term_analysis": {
        "description": "Returns detailed NLP analysis for a specific term on a specific date.",
        "example": {
            "collection": "reddit",
            "day": "2024-01-15",
            "term": "climate",
            "limit": 10,
            "bookmark": None
        }
    },
    "get_nlp_topics": {
        "description": "Returns topic clusters with top terms in a date range.",
        "example": {
            "collection": "reddit",
            "startdate": "2024-01-01",
            "enddate": "2024-01-28"
        }
    },
    "get_topic_groupings": {
        "description": "Returns topic groupings network for a date range with optional threshold.",
        "example": {
            "collection": "reddit",
            "startdate": "2024-01-01",
            "enddate": "2024-01-28",
            "threshold": 5
        }
    },
    "get_nlp_metadata": {
        "description": "Returns metadata for NLP models used in a collection, optionally scoped to a date range.",
        "example": {
            "collection": "reddit",
            "startdate": "2024-01-01",
            "enddate": "2024-01-28"
        }
    },
    "text_search": {
        "description": "Performs full-text search on a collection using a Lucene query.",
        "example": {
            "collection": "reddit",
            "query": "hashtags:\"#auspol\" AND date:[2024-02-01T00:00:00Z TO 2024-02-29T23:59:59Z]",
            "limit": None,
            "bookmark": None
        }
    },
    "generate_chart": {
        "description": "Draws a chart using the JSON QuickChart schema and returns an URL to its bitmap image.",
        "example": {
            "chart": {
                "type": "bar",
                "data": {
                    "labels": ["January", "February", "March"],
                    "datasets": [
                        {
                            "label": "Dataset 1",
                            "data": [10, 20, 30]
                        }
                    ]
                }
            },
            "width": 400,
            "height": 300,
            "format": "png",
            "backgroundColor": "#ffffff"
        }
    }
}