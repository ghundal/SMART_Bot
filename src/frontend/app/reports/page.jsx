'use client';

import { useEffect, useRef, useState } from 'react';
import ProtectedRoute from '../../components/auth/ProtectedRoute';
import ReportsService from '../../services/ReportsService';
import styles from '../../components/reports/reports.module.css';
import { exportReport, exportToPDF } from '../../components/reports/exportUtils';

function ReportsContent() {
  const [activeReport, setActiveReport] = useState('overview');
  const [timeRange, setTimeRange] = useState(30);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const reportContentRef = useRef(null);

  // States for different report data
  const [stats, setStats] = useState(null);
  const [queryActivity, setQueryActivity] = useState([]);
  const [topDocuments, setTopDocuments] = useState([]);
  const [topKeywords, setTopKeywords] = useState([]);
  const [topPhrases, setTopPhrases] = useState([]);
  const [userActivity, setUserActivity] = useState([]);
  const [dailyActiveUsers, setDailyActiveUsers] = useState([]);
  const [userCount, setUserCount] = useState(0);
  const [queryCount, setQueryCount] = useState(0);

  useEffect(() => {
    const fetchData = async () => {
      setLoading(true);
      setError(null);
      try {
        // Fetch system stats
        const systemStats = await ReportsService.getSystemStats();
        setStats(systemStats);

        // Fetch user count
        const userCountData = await ReportsService.getUserCount();
        setUserCount(userCountData.user_count);

        // Fetch query count based on time range
        const queryCountData = await ReportsService.getQueryCount(timeRange);
        setQueryCount(queryCountData.query_count);

        // Fetch query activity based on time range
        const queryActivityData = await ReportsService.getQueryActivity(timeRange);
        setQueryActivity(queryActivityData);

        // Fetch top documents
        const topDocsData = await ReportsService.getTopDocuments();
        setTopDocuments(topDocsData);

        // Fetch top keywords
        const topKeywordsData = await ReportsService.getTopKeywords();
        setTopKeywords(topKeywordsData);

        // Fetch top phrases
        const topPhrasesData = await ReportsService.getTopPhrases();
        setTopPhrases(topPhrasesData);

        // Fetch user activity
        const userActivityData = await ReportsService.getUserActivity();
        setUserActivity(userActivityData);

        // Fetch daily active users
        const dailyActiveUsersData = await ReportsService.getDailyActiveUsers(timeRange);
        setDailyActiveUsers(dailyActiveUsersData);
      } catch (err) {
        console.error('Error fetching report data:', err);
        setError('Failed to load report data. Please try again later.');
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [timeRange]);

  const handleTimeRangeChange = (days) => {
    setTimeRange(days);
  };

  // Helper to format numbers with commas
  const formatNumber = (num) => {
    return num?.toString().replace(/\B(?=(\d{3})+(?!\d))/g, ',') || '0';
  };

  if (loading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
        <p>Loading report data...</p>
      </div>
    );
  }

  if (error) {
    return (
      <div className={styles.errorContainer}>
        <p className={styles.errorMessage}>{error}</p>
        <button className={styles.retryButton} onClick={() => window.location.reload()}>
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className={styles.reportsContainer}>
      {/* Hero Section */}
      <section className={styles.hero}>
        <div className={styles.heroContent}>
          <h1>SMART Analytics</h1>
          <p>Comprehensive usage statistics and insights for your document retrieval system</p>
        </div>
      </section>

      {/* Navigation Tabs */}
      <div className={styles.tabContainer}>
        <button
          className={`${styles.tabButton} ${activeReport === 'overview' ? styles.activeTab : ''}`}
          onClick={() => setActiveReport('overview')}
        >
          Overview
        </button>
        <button
          className={`${styles.tabButton} ${activeReport === 'documents' ? styles.activeTab : ''}`}
          onClick={() => setActiveReport('documents')}
        >
          Documents
        </button>
        <button
          className={`${styles.tabButton} ${activeReport === 'queries' ? styles.activeTab : ''}`}
          onClick={() => setActiveReport('queries')}
        >
          Queries
        </button>
        <button
          className={`${styles.tabButton} ${activeReport === 'users' ? styles.activeTab : ''}`}
          onClick={() => setActiveReport('users')}
        >
          Users
        </button>
      </div>

      {/* Time Range Selector */}
      <div className={styles.timeRangeSelector}>
        <span>Time range: </span>
        <button
          className={`${styles.rangeButton} ${timeRange === 7 ? styles.activeRange : ''}`}
          onClick={() => handleTimeRangeChange(7)}
        >
          7 days
        </button>
        <button
          className={`${styles.rangeButton} ${timeRange === 30 ? styles.activeRange : ''}`}
          onClick={() => handleTimeRangeChange(30)}
        >
          30 days
        </button>
        <button
          className={`${styles.rangeButton} ${timeRange === 90 ? styles.activeRange : ''}`}
          onClick={() => handleTimeRangeChange(90)}
        >
          90 days
        </button>
      </div>

      {/* Report Content - Wrapped with a ref for PDF export */}
      <div ref={reportContentRef} className={styles.reportContent} id="reportContent">
        {/* Overview Report */}
        {activeReport === 'overview' && stats && (
          <div className={styles.reportSection}>
            <h2>System Overview</h2>
            <div className={styles.statsGrid}>
              <div className={styles.statCard}>
                <h3>Total Users</h3>
                <p className={styles.statNumber}>{formatNumber(stats.total_users)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Total Queries</h3>
                <p className={styles.statNumber}>{formatNumber(stats.total_queries)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Documents Indexed</h3>
                <p className={styles.statNumber}>{formatNumber(stats.total_documents)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Active Classes</h3>
                <p className={styles.statNumber}>{formatNumber(stats.total_classes)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Queries (Last 24h)</h3>
                <p className={styles.statNumber}>{formatNumber(stats.queries_last_24h)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Active Users (Last 24h)</h3>
                <p className={styles.statNumber}>{formatNumber(stats.active_users_last_24h)}</p>
              </div>
            </div>

            <div className={styles.chartSection}>
              <h3>Query Activity (Last {timeRange} Days)</h3>
              <div className={styles.activityChart}>
                {queryActivity.map((day, index) => (
                  <div key={index} className={styles.barContainer}>
                    <div
                      className={styles.bar}
                      style={{
                        height: `${Math.min(
                          100,
                          (day.query_count / Math.max(...queryActivity.map((d) => d.query_count))) *
                            100,
                        )}%`,
                      }}
                    ></div>
                    <span className={styles.barLabel}>
                      {new Date(day.date).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Documents Report */}
        {activeReport === 'documents' && (
          <div className={styles.reportSection}>
            <h2>Document Analytics</h2>
            <div className={styles.documentStats}>
              <h3>Top Referenced Documents</h3>
              <table className={styles.dataTable}>
                <thead>
                  <tr>
                    <th>Class ID</th>
                    <th>Class Name</th>
                    <th>Authors</th>
                    <th>Reference Count</th>
                  </tr>
                </thead>
                <tbody>
                  {topDocuments.map((doc, index) => (
                    <tr key={index}>
                      <td>{doc.class_id}</td>
                      <td>{doc.class_name}</td>
                      <td>{doc.authors}</td>
                      <td>{doc.reference_count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Queries Report */}
        {activeReport === 'queries' && (
          <div className={styles.reportSection}>
            <h2>Query Analytics</h2>
            <div className={styles.querySummary}>
              <div className={styles.statCard}>
                <h3>Total Queries (Last {timeRange} Days)</h3>
                <p className={styles.statNumber}>{formatNumber(queryCount)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Avg. Queries Per Day</h3>
                <p className={styles.statNumber}>
                  {formatNumber(Math.round(stats.avg_queries_per_day))}
                </p>
              </div>
            </div>

            <div className={styles.keywordSection}>
              <h3>Top Search Keywords</h3>
              <div className={styles.keywordCloud}>
                {topKeywords.map((keyword, index) => (
                  <div
                    key={index}
                    className={styles.keywordTag}
                    style={{
                      fontSize: `${Math.max(
                        100,
                        (keyword.count / Math.max(...topKeywords.map((k) => k.count))) * 150,
                      )}%`,
                    }}
                  >
                    {keyword.keyword}
                  </div>
                ))}
              </div>
            </div>

            <div className={styles.phraseSection}>
              <h3>Top Search Phrases</h3>
              <table className={styles.dataTable}>
                <thead>
                  <tr>
                    <th>Phrase</th>
                    <th>Count</th>
                  </tr>
                </thead>
                <tbody>
                  {topPhrases.map((phrase, index) => (
                    <tr key={index}>
                      <td>{phrase.phrase}</td>
                      <td>{phrase.count}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className={styles.chartSection}>
              <h3>Daily Query Activity</h3>
              <div className={styles.activityChart}>
                {queryActivity.map((day, index) => (
                  <div key={index} className={styles.barContainer}>
                    <div
                      className={styles.bar}
                      style={{
                        height: `${Math.min(
                          100,
                          (day.query_count / Math.max(...queryActivity.map((d) => d.query_count))) *
                            100,
                        )}%`,
                      }}
                    ></div>
                    <span className={styles.barLabel}>
                      {new Date(day.date).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* Users Report */}
        {activeReport === 'users' && (
          <div className={styles.reportSection}>
            <h2>User Analytics</h2>
            <div className={styles.userSummary}>
              <div className={styles.statCard}>
                <h3>Total Users</h3>
                <p className={styles.statNumber}>{formatNumber(userCount)}</p>
              </div>
              <div className={styles.statCard}>
                <h3>Active Users (Last 24h)</h3>
                <p className={styles.statNumber}>{formatNumber(stats.active_users_last_24h)}</p>
              </div>
            </div>

            <div className={styles.userActivitySection}>
              <h3>Most Active Users</h3>
              <table className={styles.dataTable}>
                <thead>
                  <tr>
                    <th>User Email</th>
                    <th>Query Count</th>
                    <th>First Query</th>
                    <th>Last Query</th>
                    <th>Active Days</th>
                  </tr>
                </thead>
                <tbody>
                  {userActivity.map((user, index) => (
                    <tr key={index}>
                      <td>{user.user_email}</td>
                      <td>{user.query_count}</td>
                      <td>{new Date(user.first_query).toLocaleDateString()}</td>
                      <td>{new Date(user.last_query).toLocaleDateString()}</td>
                      <td>{user.active_days}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className={styles.chartSection}>
              <h3>Daily Active Users</h3>
              <div className={styles.activityChart}>
                {dailyActiveUsers.map((day, index) => (
                  <div key={index} className={styles.barContainer}>
                    <div
                      className={styles.bar}
                      style={{
                        height: `${Math.min(
                          100,
                          (day.user_count /
                            Math.max(...dailyActiveUsers.map((d) => d.user_count))) *
                            100,
                        )}%`,
                      }}
                    ></div>
                    <span className={styles.barLabel}>
                      {new Date(day.date).toLocaleDateString(undefined, {
                        month: 'short',
                        day: 'numeric',
                      })}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>

      <div className={styles.exportSection}>
        <button
          className={styles.exportButton}
          onClick={() => {
            // Get appropriate data based on current tab
            let data;
            let reportType;

            switch (activeReport) {
              case 'overview':
                data = stats;
                reportType = 'system-overview';
                break;
              case 'documents':
                data = topDocuments;
                reportType = 'documents';
                break;
              case 'queries':
                data = {
                  queryCount,
                  avgQueriesPerDay: stats?.avg_queries_per_day,
                  queryActivity,
                  topKeywords,
                  topPhrases,
                };
                reportType = 'queries';
                break;
              case 'users':
                data = {
                  userCount,
                  activeUsers24h: stats?.active_users_last_24h,
                  userActivity,
                  dailyActiveUsers,
                };
                reportType = 'users';
                break;
              default:
                data = stats;
                reportType = 'system';
            }

            exportReport(reportType, data, 'csv');
          }}
        >
          Export Data as CSV
        </button>
        <button
          className={styles.exportButton}
          onClick={() => {
            // For PDF export, we'll use the element reference
            if (reportContentRef.current) {
              exportToPDF(
                'reportContent',
                `smart-${activeReport}-report-${new Date().toISOString().slice(0, 10)}.pdf`,
              );
            } else {
              alert('Unable to find report content for PDF export');
            }
          }}
        >
          Generate PDF Report
        </button>
      </div>
    </div>
  );
}

export default function ReportsPage() {
  return (
    <ProtectedRoute>
      <ReportsContent />
    </ProtectedRoute>
  );
}
