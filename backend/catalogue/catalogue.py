"""
catalogue.py — SQLite-backed flare event database.

Stores nowcasted and forecasted flare events persistently.
Provides query interface for the dashboard.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import List, Optional, Dict
from datetime import datetime


DB_PATH = Path(__file__).parent.parent.parent / 'data' / 'flare_catalogue.db'


class FlareCatalogue:
    """SQLite-backed catalogue of detected and forecast flare events."""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or str(DB_PATH)
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()
    
    def _init_db(self):
        """Create tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                CREATE TABLE IF NOT EXISTS nowcast_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    event_id INTEGER,
                    start_time REAL,
                    peak_time REAL,
                    end_time REAL,
                    duration REAL,
                    peak_soft REAL,
                    peak_hard REAL,
                    detection_type TEXT,
                    flare_class TEXT,
                    confidence REAL,
                    hard_lead_time REAL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS forecast_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    alert_time REAL,
                    probability REAL,
                    predicted_class TEXT,
                    horizon_sec INTEGER,
                    threshold REAL,
                    features_json TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            conn.execute('''
                CREATE TABLE IF NOT EXISTS metrics_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    metric_name TEXT,
                    metric_value REAL,
                    context TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            ''')
    
    def add_nowcast_event(self, event: dict):
        """Add a detected flare event to the catalogue."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO nowcast_events 
                (event_id, start_time, peak_time, end_time, duration,
                 peak_soft, peak_hard, detection_type, flare_class,
                 confidence, hard_lead_time)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.get('event_id', 0),
                event.get('start_time', 0),
                event.get('peak_time', 0),
                event.get('end_time', 0),
                event.get('duration', 0),
                event.get('peak_soft', 0),
                event.get('peak_hard', 0),
                event.get('detection_type', ''),
                event.get('flare_class', ''),
                event.get('confidence', 0),
                event.get('hard_lead_time', 0),
            ))
    
    def add_forecast_alert(self, alert: dict):
        """Add a forecast alert."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('''
                INSERT INTO forecast_alerts
                (alert_time, probability, predicted_class, horizon_sec, threshold, features_json)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                alert.get('alert_time', 0),
                alert.get('probability', 0),
                alert.get('predicted_class', ''),
                alert.get('horizon_sec', 300),
                alert.get('threshold', 0.5),
                json.dumps(alert.get('features', {})),
            ))
    
    def get_recent_events(self, limit: int = 50) -> List[Dict]:
        """Get the most recent nowcast events."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM nowcast_events 
                ORDER BY peak_time DESC LIMIT ?
            ''', (limit,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_recent_alerts(self, limit: int = 50) -> List[Dict]:
        """Get the most recent forecast alerts."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute('''
                SELECT * FROM forecast_alerts
                ORDER BY alert_time DESC LIMIT ?
            ''', (limit,)).fetchall()
            return [dict(row) for row in rows]
    
    def get_event_count_by_class(self) -> Dict[str, int]:
        """Get count of events per GOES class."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute('''
                SELECT flare_class, COUNT(*) as count
                FROM nowcast_events
                GROUP BY flare_class
            ''').fetchall()
            return {row[0]: row[1] for row in rows}
    
    def get_total_stats(self) -> Dict:
        """Get summary statistics."""
        with sqlite3.connect(self.db_path) as conn:
            n_events = conn.execute('SELECT COUNT(*) FROM nowcast_events').fetchone()[0]
            n_alerts = conn.execute('SELECT COUNT(*) FROM forecast_alerts').fetchone()[0]
            
            return {
                'total_events': n_events,
                'total_alerts': n_alerts,
                'events_by_class': self.get_event_count_by_class(),
            }
    
    def clear(self):
        """Clear all data (for testing)."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute('DELETE FROM nowcast_events')
            conn.execute('DELETE FROM forecast_alerts')
            conn.execute('DELETE FROM metrics_log')
