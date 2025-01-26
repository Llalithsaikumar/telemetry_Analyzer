import sqlite3
import os
from datetime import datetime


class RobotRegistry:
    def __init__(self, db_file='robot_registry.db'):
        self.db_file = db_file
        self._create_tables()

    def _get_connection(self):
        conn = sqlite3.connect(self.db_file)
        conn.row_factory = sqlite3.Row  # Enable column access by name
        return conn

    def _create_tables(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS robots (
                    robot_id TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    manufacturer TEXT NOT NULL,
                    registration_date TEXT
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS battery_specs (
                    robot_id TEXT PRIMARY KEY,
                    capacity REAL NOT NULL,
                    type TEXT NOT NULL,
                    nominal_voltage REAL NOT NULL,
                    FOREIGN KEY (robot_id) REFERENCES robots(robot_id)
                )
            ''')
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS lidar_specs (
                    robot_id TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    max_range REAL NOT NULL,
                    points_per_second INTEGER NOT NULL,
                    FOREIGN KEY (robot_id) REFERENCES robots(robot_id)
                )
            ''')
            conn.commit()
        except sqlite3.Error as e:
            print(f"Error creating tables: {e}")
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def register_robot(self, robot_id, name, manufacturer, battery_data=None, lidar_data=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                INSERT INTO robots (robot_id, name, manufacturer, registration_date)
                VALUES (?, ?, ?, ?)
            """, (robot_id, name, manufacturer, str(datetime.now())))

            if battery_data:
                cursor.execute("""
                    INSERT INTO battery_specs (robot_id, capacity, type, nominal_voltage)
                    VALUES (?, ?, ?, ?)
                """, (robot_id, battery_data['capacity'], battery_data['type'], battery_data['nominal_voltage']))

            if lidar_data:
                cursor.execute("""
                    INSERT INTO lidar_specs (robot_id, model, max_range, points_per_second)
                    VALUES (?, ?, ?, ?)
                """, (robot_id, lidar_data['model'], lidar_data['max_range'], lidar_data['points_per_second']))

            conn.commit()
            return robot_id
        except sqlite3.IntegrityError:
            return None
        except sqlite3.Error as e:
            print(f"Error registering robot: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def update_robot(self, robot_id, name=None, manufacturer=None, battery_data=None, lidar_data=None):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            updates = []
            params = []
            if name:
                updates.append("name = ?")
                params.append(name)
            if manufacturer:
                updates.append("manufacturer = ?")
                params.append(manufacturer)

            if not updates:
                return

            params.append(robot_id)
            sql = "UPDATE robots SET " + ", ".join(updates) + " WHERE robot_id = ?"
            cursor.execute(sql, params)

            if battery_data:
                cursor.execute("""
                    INSERT INTO battery_specs (robot_id, capacity, type, nominal_voltage)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(robot_id) DO UPDATE SET
                    capacity = excluded.capacity,
                    type = excluded.type,
                    nominal_voltage = excluded.nominal_voltage
                """, (robot_id, battery_data['capacity'], battery_data['type'], battery_data['nominal_voltage']))

            if lidar_data:
                cursor.execute("""
                    INSERT INTO lidar_specs (robot_id, model, max_range, points_per_second)
                    VALUES (?, ?, ?, ?)
                    ON CONFLICT(robot_id) DO UPDATE SET
                    model = excluded.model,
                    max_range = excluded.max_range,
                    points_per_second = excluded.points_per_second
                """, (robot_id, lidar_data['model'], lidar_data['max_range'], lidar_data['points_per_second']))

            conn.commit()

        except sqlite3.Error as e:
            print(f"Error updating robot: {e}")
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def get_robot_details(self, robot_id):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("""
                SELECT r.*, b.capacity AS battery_capacity, b.type AS battery_type, b.nominal_voltage AS battery_nominal_voltage,
                       l.model AS lidar_model, l.max_range AS lidar_max_range, l.points_per_second AS lidar_points_per_second
                FROM robots r
                LEFT JOIN battery_specs b ON r.robot_id = b.robot_id
                LEFT JOIN lidar_specs l ON r.robot_id = l.robot_id
                WHERE r.robot_id = ?
            """, (robot_id,))
            row = cursor.fetchone()
            if row:
                return {
                    'robot_id': row['robot_id'],
                    'name': row['name'],
                    'manufacturer': row['manufacturer'],
                    'registration_date': row['registration_date'],
                    'battery_data': {
                        'capacity': row['battery_capacity'],
                        'type': row['battery_type'],
                        'nominal_voltage': row['battery_nominal_voltage']
                    } if row['battery_capacity'] is not None else None,
                    'lidar_data': {
                        'model': row['lidar_model'],
                        'max_range': row['lidar_max_range'],
                        'points_per_second': row['lidar_points_per_second']
                    } if row['lidar_model'] is not None else None
                }
            else:
                return None
        except sqlite3.Error as e:
            print(f"Error getting robot details: {e}")
            return None
        finally:
            if cursor:
                cursor.close()
            conn.close()

    def list_robots(self):
        conn = self._get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT robot_id FROM robots")
            return [row['robot_id'] for row in cursor.fetchall()]
        except sqlite3.Error as e:
            print(f"Error listing robots: {e}")
            return []
        finally:
            if cursor:
                cursor.close()
            conn.close()