"""
Reports API for the Ollama RAG system.
This module provides endpoints for retrieving usage statistics and analytics.
"""

from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text

from ..utils.database import SessionLocal
from .auth_middleware import verify_token

# Define a router for reports
router = APIRouter()


@router.get("/reports/users")
async def get_user_count(user_email: str = Depends(verify_token)):
    """Get the count of unique users who have made queries"""

    db = SessionLocal()
    try:
        # Query to count unique users
        result = db.execute(text("SELECT COUNT(DISTINCT user_email) FROM audit")).scalar()

        return {"user_count": result}
    finally:
        db.close()


@router.get("/reports/queries")
async def get_query_count(
    days: Optional[int] = Query(30, description="Number of days to include in the report"),
    user_email: str = Depends(verify_token),
):
    """Get the count of queries made in the past X days"""

    db = SessionLocal()
    try:
        # Query to count queries in the specified time period
        result = db.execute(
            text(
                "SELECT COUNT(*) FROM audit WHERE event_time > CURRENT_TIMESTAMP - make_interval(days => :days)"
            ),
            {"days": days},
        ).scalar()

        return {"query_count": result, "days": days}
    finally:
        db.close()


@router.get("/reports/top_documents")
async def get_top_documents(
    limit: Optional[int] = Query(10, description="Number of documents to return"),
    user_email: str = Depends(verify_token),
):
    """Get the most frequently referenced documents in responses"""

    db = SessionLocal()
    try:
        # Query to find the most commonly referenced documents
        results = db.execute(
            text(
                """
            SELECT d.class_id, c.class_name, c.authors, COUNT(*) as reference_count
            FROM (
                SELECT UNNEST(document_ids) as document_id
                FROM audit
            ) a
            JOIN document d ON a.document_id = d.document_id
            JOIN class c ON d.class_id = c.class_id
            GROUP BY d.class_id, c.class_name, c.authors
            ORDER BY reference_count DESC
            LIMIT :limit
            """
            ),
            {"limit": limit},
        ).fetchall()

        # Format the results
        formatted_results = [
            {"class_id": row[0], "class_name": row[1], "authors": row[2], "reference_count": row[3]}
            for row in results
        ]

        return formatted_results
    finally:
        db.close()


@router.get("/reports/query_activity")
async def get_query_activity(
    days: Optional[int] = Query(30, description="Number of days to include in the report"),
    user_email: str = Depends(verify_token),
):
    """Get daily query activity for the past X days"""

    db = SessionLocal()
    try:
        # Query to get daily activity
        results = db.execute(
            text(
                """
            SELECT DATE(event_time) as date, COUNT(*) as query_count
            FROM audit
            WHERE event_time > CURRENT_TIMESTAMP - make_interval(days => :days)
            GROUP BY DATE(event_time)
            ORDER BY date
            """
            ),
            {"days": days},
        ).fetchall()

        # Format the results
        formatted_results = [{"date": row[0].isoformat(), "query_count": row[1]} for row in results]

        return formatted_results
    finally:
        db.close()


@router.get("/reports/top_keywords")
async def get_top_keywords(
    limit: Optional[int] = Query(20, description="Number of keywords to return"),
    min_length: Optional[int] = Query(4, description="Minimum keyword length"),
    user_email: str = Depends(verify_token),
):
    """Get the most frequently used keywords in user queries"""

    # Common words to exclude from analysis
    exclude_words = [
        "what",
        "when",
        "where",
        "which",
        "who",
        "how",
        "why",
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "can",
        "you",
        "have",
        "are",
        "not",
        "from",
        "your",
    ]

    # Convert the list to a format suitable for PostgreSQL's ANY operator
    "{" + ",".join(exclude_words) + "}"

    db = SessionLocal()
    try:
        # Use PostgreSQL's text search capabilities to extract keywords
        results = db.execute(
            text(
                """
            WITH keywords AS (
                SELECT
                    word,
                    COUNT(*) as count
                FROM (
                    SELECT regexp_split_to_table(lower(query), E'\\s+') as word
                    FROM audit
                ) t
                WHERE
                    length(word) >= :min_length
                    AND word NOT IN (
                        'what', 'when', 'where', 'which', 'who', 'how', 'why', 'the', 'and', 'for',
                        'that', 'this', 'with', 'can', 'you', 'have', 'are', 'not', 'from', 'your'
                    )
                    AND word ~ '^[a-z0-9]+$'  -- only include alphanumeric words
                GROUP BY word
                ORDER BY count DESC
                LIMIT :limit
            )
            SELECT word, count FROM keywords
            """
            ),
            {"limit": limit, "min_length": min_length},
        ).fetchall()

        # Format the results
        formatted_results = [{"keyword": row[0], "count": row[1]} for row in results]

        return formatted_results
    finally:
        db.close()


@router.get("/reports/top_phrases")
async def get_top_phrases(
    limit: Optional[int] = Query(10, description="Number of phrases to return"),
    user_email: str = Depends(verify_token),
):
    """Get the most frequently used phrases in user queries"""

    db = SessionLocal()
    try:
        # Extract common 2-word phrases with a simpler approach
        results = db.execute(
            text(
                """
            WITH words AS (
                SELECT
                    audit_id,
                    word,
                    row_number() OVER (PARTITION BY audit_id ORDER BY position) as position
                FROM (
                    SELECT
                        audit_id,
                        regexp_split_to_table(lower(query), E'\\s+') as word,
                        regexp_split_to_table(lower(query), E'\\s+') with ordinality as position
                    FROM audit
                ) t
                WHERE length(word) >= 3
                AND word NOT IN (
                    'what', 'when', 'where', 'which', 'who', 'how', 'why', 'the', 'and', 'for',
                    'that', 'this', 'with', 'can', 'you', 'have', 'are', 'not', 'from', 'your'
                )
            ),
            phrases AS (
                SELECT
                    w1.audit_id,
                    w1.word || ' ' || w2.word AS phrase
                FROM words w1
                JOIN words w2 ON
                    w1.audit_id = w2.audit_id AND
                    w1.position + 1 = w2.position
                WHERE length(w2.word) >= 3
                AND w2.word NOT IN (
                    'what', 'when', 'where', 'which', 'who', 'how', 'why', 'the', 'and', 'for',
                    'that', 'this', 'with', 'can', 'you', 'have', 'are', 'not', 'from', 'your'
                )
            )
            SELECT phrase, COUNT(*) as count
            FROM phrases
            GROUP BY phrase
            ORDER BY count DESC
            LIMIT :limit
            """
            ),
            {"limit": limit},
        ).fetchall()

        # Format the results
        formatted_results = [{"phrase": row[0], "count": row[1]} for row in results]

        return formatted_results
    finally:
        db.close()


@router.get("/reports/user_activity")
async def get_user_activity(
    limit: Optional[int] = Query(10, description="Number of users to return"),
    user_email: str = Depends(verify_token),
):
    """Get the most active users by query count"""

    db = SessionLocal()
    try:
        # Query to get most active users
        results = db.execute(
            text(
                """
            SELECT
                user_email,
                COUNT(*) as query_count,
                MIN(event_time) as first_query,
                MAX(event_time) as last_query,
                COUNT(DISTINCT DATE(event_time)) as active_days
            FROM audit
            WHERE user_email IS NOT NULL
            GROUP BY user_email
            ORDER BY query_count DESC
            LIMIT :limit
            """
            ),
            {"limit": limit},
        ).fetchall()

        # Format the results
        formatted_results = [
            {
                "user_email": row[0],
                "query_count": row[1],
                "first_query": row[2].isoformat() if row[2] else None,
                "last_query": row[3].isoformat() if row[3] else None,
                "active_days": row[4],
            }
            for row in results
        ]

        return formatted_results
    finally:
        db.close()


@router.get("/reports/daily_active_users")
async def get_daily_active_users(
    days: Optional[int] = Query(30, description="Number of days to include in the report"),
    user_email: str = Depends(verify_token),
):
    """Get daily active users for the past X days"""

    db = SessionLocal()
    try:
        # Query to get daily active users
        results = db.execute(
            text(
                """
            SELECT
                DATE(event_time) as date,
                COUNT(DISTINCT user_email) as user_count
            FROM audit
            WHERE
                event_time > CURRENT_TIMESTAMP - make_interval(days => :days)
                AND user_email IS NOT NULL
            GROUP BY DATE(event_time)
            ORDER BY date
            """
            ),
            {"days": days},
        ).fetchall()

        # Format the results
        formatted_results = [{"date": row[0].isoformat(), "user_count": row[1]} for row in results]

        return formatted_results
    finally:
        db.close()


@router.get("/reports/system_stats")
async def get_system_stats(user_email: str = Depends(verify_token)):
    """Get overall system statistics"""

    db = SessionLocal()
    try:
        # Query various system metrics
        stats = {}

        # Total users
        stats["total_users"] = db.execute(
            text("SELECT COUNT(DISTINCT user_email) FROM audit")
        ).scalar()

        # Total queries
        stats["total_queries"] = db.execute(text("SELECT COUNT(*) FROM audit")).scalar()

        # Total documents
        stats["total_documents"] = db.execute(text("SELECT COUNT(*) FROM document")).scalar()

        # Total classes
        stats["total_classes"] = db.execute(text("SELECT COUNT(*) FROM class")).scalar()

        # Total chunks
        stats["total_chunks"] = db.execute(text("SELECT COUNT(*) FROM chunk")).scalar()

        # Queries in last 24 hours
        stats["queries_last_24h"] = db.execute(
            text(
                "SELECT COUNT(*) FROM audit WHERE event_time > CURRENT_TIMESTAMP - interval '1 day'"
            )
        ).scalar()

        # Active users in last 24 hours
        stats["active_users_last_24h"] = db.execute(
            text(
                "SELECT COUNT(DISTINCT user_email) FROM audit WHERE event_time > CURRENT_TIMESTAMP - interval '1 day'"
            )
        ).scalar()

        # Average queries per day (last 30 days)
        stats["avg_queries_per_day"] = db.execute(
            text(
                """
            SELECT AVG(query_count) FROM (
                SELECT DATE(event_time) as day, COUNT(*) as query_count
                FROM audit
                WHERE event_time > CURRENT_TIMESTAMP - make_interval(days => 30)
                GROUP BY day
            ) daily_counts
            """
            )
        ).scalar()

        return stats
    finally:
        db.close()
