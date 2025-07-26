import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import json
from typing import List, Dict, Optional
import os
import time
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configuration
st.set_page_config(
    page_title="Migration Guide MR Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# GitHub/GitLab API configuration
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN", "")
GITLAB_TOKEN = st.secrets.get("GITLAB_TOKEN", "")
REPO_OWNER = st.secrets.get("REPO_OWNER", "")
REPO_NAME = st.secrets.get("REPO_NAME", "")
PLATFORM = st.secrets.get("PLATFORM", "gitlab")  # "github" or "gitlab"
GITLAB_URL = st.secrets.get("GITLAB_URL", "https://gitlab.com")  # Custom GitLab instance URL

def create_robust_session():
    """Create a requests session with retry logic and timeout settings"""
    session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    
    # Add retry adapter
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

class MRDataFetcher:
    """Fetch MR/PR data from GitLab or GitHub"""
    
    def __init__(self, platform: str, token: str, owner: str, repo: str):
        self.platform = platform
        self.token = token
        self.owner = owner
        self.repo = repo
        
        if platform == "github":
            self.base_url = f"https://api.github.com/repos/{owner}/{repo}"
            self.headers = {
                "Authorization": f"token {token}",
                "Accept": "application/vnd.github.v3+json"
            }
        else:  # gitlab
            # Encode the project path properly for custom GitLab instances
            project_path = f"{owner}/{repo}"
            encoded_path = project_path.replace("/", "%2F")
            self.base_url = f"{GITLAB_URL}/api/v4/projects/{encoded_path}"
            self.headers = {"PRIVATE-TOKEN": token}
    
    def fetch_migration_guide_mrs(self, days_back: int = 30) -> List[Dict]:
        """Fetch MRs/PRs with migration-guide label"""
        
        if self.platform == "github":
            return self._fetch_github_mrs(days_back)
        else:
            return self._fetch_gitlab_mrs(days_back)
    
    def _fetch_github_mrs(self, days_back: int) -> List[Dict]:
        """Fetch GitHub PRs with migration-guide label"""
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/issues"
        params = {
            "state": "all",
            "labels": "migration-guide",
            "since": since_date,
            "per_page": 100
        }
        
        try:
            session = create_robust_session()
            response = session.get(
                url, 
                headers=self.headers, 
                params=params,
                timeout=30
            )
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection error to GitHub: {e}")
            return []
        except requests.exceptions.Timeout as e:
            st.error(f"Timeout error: {e}")
            return []
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return []
        
        if response.status_code != 200:
            st.error(f"GitHub API error: {response.status_code} - {response.text}")
            return []
        
        issues = response.json()
        mrs = []
        
        for issue in issues:
            if "pull_request" in issue:  # Only PRs, not issues
                mr_data = {
                    "id": issue["number"],
                    "title": issue["title"],
                    "state": issue["state"],
                    "created_at": issue["created_at"],
                    "updated_at": issue["updated_at"],
                    "closed_at": issue.get("closed_at"),
                    "labels": [label["name"] for label in issue["labels"]],
                    "author": issue["user"]["login"],
                    "url": issue["html_url"]
                }
                mrs.append(mr_data)
        
        return mrs
    
    def _fetch_gitlab_mrs(self, days_back: int) -> List[Dict]:
        """Fetch GitLab MRs with migration-guide label"""
        since_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        
        url = f"{self.base_url}/merge_requests"
        params = {
            "state": "all",
            "labels": "migration-guide",
            "created_after": since_date,
            "per_page": 100
        }
        
        try:
            session = create_robust_session()
            response = session.get(
                url, 
                headers=self.headers, 
                params=params,
                timeout=30  # 30 second timeout
            )
        except requests.exceptions.ConnectionError as e:
            st.error(f"Connection error to GitLab: {e}")
            st.info("This might be due to network restrictions or VPN requirements.")
            return []
        except requests.exceptions.Timeout as e:
            st.error(f"Timeout error: {e}")
            return []
        except Exception as e:
            st.error(f"Unexpected error: {e}")
            return []
        
        if response.status_code != 200:
            st.error(f"GitLab API error: {response.status_code} - {response.text}")
            return []
        
        mrs_data = response.json()
        mrs = []
        
        for mr in mrs_data:
            mr_data = {
                "id": mr["iid"],
                "title": mr["title"],
                "state": mr["state"],
                "created_at": mr["created_at"],
                "updated_at": mr["updated_at"],
                "closed_at": mr.get("closed_at"),
                "labels": mr["labels"],
                "author": mr["author"]["username"],
                "url": mr["web_url"]
            }
            mrs.append(mr_data)
        
        return mrs

def test_gitlab_connection(gitlab_url: str, token: str) -> bool:
    """Test if we can connect to GitLab instance"""
    try:
        session = create_robust_session()
        test_url = f"{gitlab_url}/api/v4/version"
        headers = {"PRIVATE-TOKEN": token}
        
        response = session.get(test_url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True
        else:
            st.error(f"GitLab connection test failed: {response.status_code}")
            return False
    except Exception as e:
        st.error(f"GitLab connection test failed: {e}")
        return False

def create_charts(df: pd.DataFrame):
    """Create various charts from MR data"""
    
    if df.empty:
        st.warning("No migration guide MRs found for the selected time period.")
        return
    
    # 1. MR Status Distribution
    st.subheader("📊 MR Status Distribution")
    status_counts = df['state'].value_counts()
    fig_status = px.pie(
        values=status_counts.values,
        names=status_counts.index,
        title="MR Status Distribution"
    )
    st.plotly_chart(fig_status, use_container_width=True)
    
    # 2. MRs Over Time
    st.subheader("📈 MRs Created Over Time")
    df['created_date'] = pd.to_datetime(df['created_at']).dt.date
    daily_counts = df.groupby('created_date').size().reset_index(name='count')
    
    fig_timeline = px.line(
        daily_counts,
        x='created_date',
        y='count',
        title="MRs Created Daily",
        labels={'created_date': 'Date', 'count': 'Number of MRs'}
    )
    st.plotly_chart(fig_timeline, use_container_width=True)
    
    # 3. Top Contributors
    st.subheader("👥 Top Contributors")
    author_counts = df['author'].value_counts().head(10)
    fig_authors = px.bar(
        x=author_counts.values,
        y=author_counts.index,
        orientation='h',
        title="Top Contributors by MR Count",
        labels={'x': 'Number of MRs', 'y': 'Author'}
    )
    st.plotly_chart(fig_authors, use_container_width=True)
    
    # 4. Label Analysis
    st.subheader("🏷️ Label Analysis")
    all_labels = []
    for labels in df['labels']:
        all_labels.extend(labels)
    
    label_counts = pd.Series(all_labels).value_counts()
    # Remove migration-guide from the count since it's our filter
    label_counts = label_counts.drop('migration-guide', errors='ignore')
    
    if not label_counts.empty:
        fig_labels = px.bar(
            x=label_counts.values,
            y=label_counts.index,
            orientation='h',
            title="Most Common Labels (excluding migration-guide)",
            labels={'x': 'Count', 'y': 'Label'}
        )
        st.plotly_chart(fig_labels, use_container_width=True)
    else:
        st.info("No additional labels found beyond 'migration-guide'")
    
    # 5. MR Duration Analysis (for closed MRs)
    st.subheader("⏱️ MR Duration Analysis")
    closed_mrs = df[df['state'] == 'closed'].copy()
    if not closed_mrs.empty:
        closed_mrs['created_at'] = pd.to_datetime(closed_mrs['created_at'])
        closed_mrs['closed_at'] = pd.to_datetime(closed_mrs['closed_at'])
        closed_mrs['duration_days'] = (closed_mrs['closed_at'] - closed_mrs['created_at']).dt.days
        
        fig_duration = px.histogram(
            closed_mrs,
            x='duration_days',
            title="MR Duration Distribution (Days)",
            labels={'duration_days': 'Days to Close', 'count': 'Number of MRs'}
        )
        st.plotly_chart(fig_duration, use_container_width=True)
        
        avg_duration = closed_mrs['duration_days'].mean()
        st.metric("Average MR Duration", f"{avg_duration:.1f} days")
    else:
        st.info("No closed MRs found for duration analysis")

def main():
    st.title("🚀 Migration Guide MR Dashboard")
    st.markdown("---")
    
    # Sidebar configuration
    st.sidebar.header("⚙️ Configuration")
    
    # Platform selection
    platform = st.sidebar.selectbox(
        "Git Platform",
        ["github", "gitlab"],
        index=1 if PLATFORM == "gitlab" else 0
    )
    
    # Time period selection
    days_back = st.sidebar.slider(
        "Time Period (Days)",
        min_value=7,
        max_value=365,
        value=30,
        help="Number of days to look back for MRs"
    )
    
    # Refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.rerun()
    
    # Check if required secrets are configured
    if not GITHUB_TOKEN and not GITLAB_TOKEN:
        st.error("""
        ⚠️ **Configuration Required**
        
        Please configure the following secrets in your Streamlit deployment:
        
        **For GitLab (Recommended):**
        - `GITLAB_TOKEN`: Your GitLab personal access token
        - `REPO_OWNER`: Project ID or owner/group
        - `REPO_NAME`: Repository name
        - `PLATFORM`: "gitlab"
        - `GITLAB_URL`: Your GitLab instance URL (default: "https://gitlab.com")
        
        **For GitHub:**
        - `GITHUB_TOKEN`: Your GitHub personal access token
        - `REPO_OWNER`: Repository owner/organization
        - `REPO_NAME`: Repository name
        - `PLATFORM`: "github"
        
        **Example for GitLab:**
        ```
        GITLAB_TOKEN = "glpat-xxxxxxxxxxxxxxxxxxxx"
        REPO_OWNER = "your-group"
        REPO_NAME = "your-project"
        PLATFORM = "gitlab"
        GITLAB_URL = "https://gitlab-master.nvidia.com"
        ```
        """)
        return
    
    # Initialize data fetcher
    token = GITHUB_TOKEN if platform == "github" else GITLAB_TOKEN
    fetcher = MRDataFetcher(platform, token, REPO_OWNER, REPO_NAME)
    
    # Test GitLab connection if using GitLab
    if platform == "gitlab":
        st.sidebar.subheader("🔗 GitLab Connection Test")
        if st.sidebar.button("Test GitLab Connection"):
            if test_gitlab_connection(GITLAB_URL, GITLAB_TOKEN):
                st.success("Successfully connected to GitLab instance!")
            else:
                st.error("Could not connect to GitLab instance. Please check your token and URL.")
    
    # Fetch data
    with st.spinner(f"Fetching migration guide MRs from the last {days_back} days..."):
        mrs_data = fetcher.fetch_migration_guide_mrs(days_back)
    
    if not mrs_data:
        st.warning("No migration guide MRs found or unable to fetch data.")
        st.info("Showing sample data for demonstration purposes.")
        
        # Show sample data for demonstration
        sample_data = [
            {
                "id": "SAMPLE-001",
                "title": "Sample Migration Guide MR",
                "state": "open",
                "created_at": "2024-01-15T10:00:00Z",
                "updated_at": "2024-01-15T10:00:00Z",
                "closed_at": None,
                "labels": ["migration-guide", "documentation"],
                "author": "sample-user",
                "url": "#"
            },
            {
                "id": "SAMPLE-002", 
                "title": "Another Sample MR",
                "state": "closed",
                "created_at": "2024-01-10T10:00:00Z",
                "updated_at": "2024-01-12T10:00:00Z",
                "closed_at": "2024-01-12T10:00:00Z",
                "labels": ["migration-guide", "bug-fix"],
                "author": "another-user",
                "url": "#"
            }
        ]
        mrs_data = sample_data
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(mrs_data)
    
    # Display summary metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total MRs", len(df))
    
    with col2:
        open_mrs = len(df[df['state'] == 'open'])
        st.metric("Open MRs", open_mrs)
    
    with col3:
        closed_mrs = len(df[df['state'] == 'closed'])
        st.metric("Closed MRs", closed_mrs)
    
    with col4:
        unique_authors = df['author'].nunique()
        st.metric("Unique Authors", unique_authors)
    
    st.markdown("---")
    
    # Create charts
    create_charts(df)
    
    # Display raw data
    st.subheader("📋 Raw MR Data")
    st.dataframe(
        df[['id', 'title', 'state', 'author', 'created_at', 'labels']],
        use_container_width=True
    )

if __name__ == "__main__":
    main() 