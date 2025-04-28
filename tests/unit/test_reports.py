from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest

# Import the functions to test
from src.api.routers.reports import (
    get_daily_active_users,
    get_query_activity,
    get_query_count,
    get_system_stats,
    get_top_documents,
    get_top_keywords,
    get_top_phrases,
    get_user_activity,
    get_user_count,
)


class TestReportsAPI:
    """Tests for the reports.py module"""

    @pytest.fixture
    def mock_db(self):
        """Create a mock database session"""
        mock_db = MagicMock()
        mock_db.close = MagicMock()
        return mock_db

    @pytest.mark.asyncio
    async def test_get_user_count(self, mock_db):
        """Test getting the count of unique users"""
        # Mock the database response
        mock_result = 42
        mock_db.execute.return_value.scalar.return_value = mock_result

        # Test with mocked SessionLocal
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_user_count(user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        assert "COUNT(DISTINCT user_email)" in str(sql)
        assert "FROM audit" in str(sql)

        # Verify result
        assert result["user_count"] == mock_result

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_query_count(self, mock_db):
        """Test getting the count of queries within a time period"""
        # Mock the database response
        mock_result = 100
        mock_db.execute.return_value.scalar.return_value = mock_result

        # Test with mocked SessionLocal and default days (30)
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_query_count(days=30, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "COUNT(*) FROM audit" in str(sql)
        assert "INTERVAL ':days days'" in str(sql)
        assert params["days"] == 30

        # Verify result
        assert result["query_count"] == mock_result
        assert result["days"] == 30

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_top_documents(self, mock_db):
        """Test getting the most frequently referenced documents"""
        # Mock the database response
        mock_results = [
            (1, "Machine Learning Basics", "John Doe", 15),
            (2, "Advanced NLP", "Jane Smith", 10),
            (3, "Data Science", "Bob Johnson", 5),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_results

        # Test with mocked SessionLocal and default limit (10)
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_top_documents(limit=10, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "SELECT" in str(sql)
        assert "UNNEST(document_ids)" in str(sql)
        assert "ORDER BY reference_count DESC" in str(sql)
        assert "LIMIT :limit" in str(sql)
        assert params["limit"] == 10

        # Verify result format
        assert len(result) == 3
        assert result[0]["class_id"] == 1
        assert result[0]["class_name"] == "Machine Learning Basics"
        assert result[0]["authors"] == "John Doe"
        assert result[0]["reference_count"] == 15

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_query_activity(self, mock_db):
        """Test getting daily query activity"""
        # Mock the database response
        today = date.today()
        yesterday = date.today().replace(day=today.day - 1)
        mock_results = [(yesterday, 10), (today, 15)]
        mock_db.execute.return_value.fetchall.return_value = mock_results

        # Test with mocked SessionLocal and default days (30)
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_query_activity(days=30, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "DATE(event_time) as date" in str(sql)
        assert "COUNT(*) as query_count" in str(sql)
        assert "GROUP BY DATE(event_time)" in str(sql)
        assert params["days"] == 30

        # Verify result format
        assert len(result) == 2
        assert result[0]["date"] == yesterday.isoformat()
        assert result[0]["query_count"] == 10
        assert result[1]["date"] == today.isoformat()
        assert result[1]["query_count"] == 15

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_top_keywords(self, mock_db):
        """Test getting the most frequently used keywords"""
        # Mock the database response
        mock_results = [("machine", 20), ("learning", 15), ("python", 10)]
        mock_db.execute.return_value.fetchall.return_value = mock_results

        # Test with mocked SessionLocal and default parameters
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_top_keywords(limit=20, min_length=4, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "regexp_split_to_table" in str(sql)
        assert "length(word) >= :min_length" in str(sql)
        assert "word NOT IN :exclude_words" in str(sql)
        assert "LIMIT :limit" in str(sql)
        assert params["limit"] == 20
        assert params["min_length"] == 4
        assert "what" in params["exclude_words"]  # Check that exclude words are passed

        # Verify result format
        assert len(result) == 3
        assert result[0]["keyword"] == "machine"
        assert result[0]["count"] == 20
        assert result[1]["keyword"] == "learning"
        assert result[1]["count"] == 15

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_top_phrases(self, mock_db):
        """Test getting the most frequently used phrases"""
        # Mock the database response
        mock_results = [("machine learning", 15), ("deep learning", 10), ("data science", 5)]
        mock_db.execute.return_value.fetchall.return_value = mock_results

        # Test with mocked SessionLocal and default limit (10)
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_top_phrases(limit=10, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "regexp_split_to_table" in str(sql)
        assert "phrase" in str(sql)
        assert "COUNT(*) as count" in str(sql)
        assert "LIMIT :limit" in str(sql)
        assert params["limit"] == 10

        # Verify result format
        assert len(result) == 3
        assert result[0]["phrase"] == "machine learning"
        assert result[0]["count"] == 15
        assert result[1]["phrase"] == "deep learning"
        assert result[1]["count"] == 10

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_user_activity(self, mock_db):
        """Test getting the most active users"""
        # Mock the database response
        now = datetime.now()
        week_ago = datetime.now().replace(day=now.day - 7)
        mock_results = [
            ("user1@example.com", 50, week_ago, now, 5),
            ("user2@example.com", 30, week_ago, now, 3),
        ]
        mock_db.execute.return_value.fetchall.return_value = mock_results

        # Test with mocked SessionLocal and default limit (10)
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_user_activity(limit=10, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "user_email" in str(sql)
        assert "COUNT(*) as query_count" in str(sql)
        assert "active_days" in str(sql)
        assert "ORDER BY query_count DESC" in str(sql)
        assert "LIMIT :limit" in str(sql)
        assert params["limit"] == 10

        # Verify result format
        assert len(result) == 2
        assert result[0]["user_email"] == "user1@example.com"
        assert result[0]["query_count"] == 50
        assert result[0]["first_query"] == week_ago.isoformat()
        assert result[0]["last_query"] == now.isoformat()
        assert result[0]["active_days"] == 5

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_daily_active_users(self, mock_db):
        """Test getting daily active users"""
        # Mock the database response
        today = date.today()
        yesterday = date.today().replace(day=today.day - 1)
        mock_results = [(yesterday, 5), (today, 8)]
        mock_db.execute.return_value.fetchall.return_value = mock_results

        # Test with mocked SessionLocal and default days (30)
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_daily_active_users(days=30, user_email="test@example.com")

        # Verify database query
        mock_db.execute.assert_called_once()
        sql = mock_db.execute.call_args[0][0]
        params = mock_db.execute.call_args[0][1]
        assert "DATE(event_time) as date" in str(sql)
        assert "COUNT(DISTINCT user_email) as user_count" in str(sql)
        assert "GROUP BY DATE(event_time)" in str(sql)
        assert params["days"] == 30

        # Verify result format
        assert len(result) == 2
        assert result[0]["date"] == yesterday.isoformat()
        assert result[0]["user_count"] == 5
        assert result[1]["date"] == today.isoformat()
        assert result[1]["user_count"] == 8

        # Verify session was closed
        mock_db.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_get_system_stats(self, mock_db):
        """Test getting overall system statistics"""
        # Mock the database response for multiple queries
        mock_values = {
            "total_users": 50,
            "total_queries": 1000,
            "total_documents": 200,
            "total_classes": 20,
            "total_chunks": 5000,
            "queries_last_24h": 100,
            "active_users_last_24h": 30,
            "avg_queries_per_day": 33.3,
        }

        # Set up mock to return different values for different queries
        def mock_execute_scalar_side_effect(query, *args, **kwargs):
            mock_result = MagicMock()

            query_str = str(query)
            if "COUNT(DISTINCT user_email) FROM audit" in query_str and "INTERVAL" not in query_str:
                mock_result.scalar.return_value = mock_values["total_users"]
            elif "COUNT(*) FROM audit" in query_str and "INTERVAL" not in query_str:
                mock_result.scalar.return_value = mock_values["total_queries"]
            elif "COUNT(*) FROM document" in query_str:
                mock_result.scalar.return_value = mock_values["total_documents"]
            elif "COUNT(*) FROM class" in query_str:
                mock_result.scalar.return_value = mock_values["total_classes"]
            elif "COUNT(*) FROM chunk" in query_str:
                mock_result.scalar.return_value = mock_values["total_chunks"]
            elif (
                "COUNT(*) FROM audit WHERE event_time > CURRENT_TIMESTAMP - INTERVAL '1 day'"
                in query_str
            ):
                mock_result.scalar.return_value = mock_values["queries_last_24h"]
            elif (
                "COUNT(DISTINCT user_email) FROM audit WHERE event_time > CURRENT_TIMESTAMP - INTERVAL '1 day'"
                in query_str
            ):
                mock_result.scalar.return_value = mock_values["active_users_last_24h"]
            elif "AVG" in query_str:
                mock_result.scalar.return_value = mock_values["avg_queries_per_day"]
            else:
                mock_result.scalar.return_value = 0

            return mock_result

        mock_db.execute.side_effect = mock_execute_scalar_side_effect

        # Test with mocked SessionLocal
        with patch("src.api.routers.reports.SessionLocal", return_value=mock_db):
            result = await get_system_stats(user_email="test@example.com")

        # Verify all stats were queried (8 different queries)
        assert mock_db.execute.call_count == 8

        # Verify result contains all expected stats
        for key, value in mock_values.items():
            assert result[key] == value

        # Verify session was closed
        mock_db.close.assert_called_once()


if __name__ == "__main__":
    pytest.main(["-xvs", "tests/test_reports.py"])
