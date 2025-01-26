import numpy as np
import random
import json
import time
import logging
import requests
import os
import uuid
import threading
import queue
import math
from datetime import datetime
from collections import deque
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from typing import Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

# Global file lock for risk scores
_risk_scores_file_lock = threading.Lock()


class ThreadSafeBuffer:
    def __init__(self, maxsize=1000):
        self.queue = queue.Queue(maxsize=maxsize)
        self.lock = threading.Lock()

    def put(self, item):
        with self.lock:
            if self.queue.full():
                self.queue.get()  # Remove oldest item if full
            self.queue.put(item)

    def get(self):
        with self.lock:
            return self.queue.get()

    def get_all(self):
        with self.lock:
            items = []
            while not self.queue.empty():
                items.append(self.queue.get())
            return items

    def empty(self):
        return self.queue.empty()


class RealTimeDataSimulator:
    def __init__(self, robot_id):
        self.robot_id = robot_id
        # Initialize robot state
        self.battery_level = 95.0  # Start at 95% charge
        self.battery_temp = 25.0   # Start at 25°C
        self.last_update = time.time()
        
        # Initialize LIDAR state
        self.error_rate = 0.02
        self.range_deviation = 0.01
        self.point_cloud_integrity = 98.0
        
        # Initialize movement pattern
        self.movement_pattern = self._generate_movement_pattern()
        self.current_pattern_index = 0
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_theta = 0.0
        self.current_speed = 0.0

    def _generate_movement_pattern(self):
        """Generate a unique movement pattern based on robot_id."""
        # Use robot_id to seed random for consistent but unique patterns
        random.seed(hash(self.robot_id))
        pattern = []
        for _ in range(10):
            duration = random.uniform(2.0, 5.0)
            velocity = random.uniform(0.1, 1.0)
            turn_rate = random.uniform(-0.5, 0.5)
            pattern.append({
                'duration': duration,
                'velocity': velocity,
                'turn_rate': turn_rate
            })
        return pattern

    def generate_telemetry_data(self):
        """Generate realistic telemetry data for the robot."""
        try:
            current_time = time.time()
            dt = current_time - self.last_update
            self.last_update = current_time

            # Update robot state
            self._update_movement(dt)
            self._update_battery(dt)
            self._update_lidar(dt)

            # Generate telemetry data
            telemetry = {
                'robot_id': self.robot_id,
                'timestamp': datetime.now().isoformat(),
                'poses': self._generate_poses(),
                'motion': {
                    'speed': round(self.current_speed, 3),
                    'acceleration': round(random.uniform(-0.2, 0.2), 3),
                    'angular_velocity': round(self.movement_pattern[self.current_pattern_index]['turn_rate'], 3)
                },
                'power': {
                    'battery_level': round(self.battery_level, 2),
                    'battery_temp': round(self.battery_temp, 2),
                    'charging_status': 'discharging',
                    'voltage': round(24.0 + random.uniform(-0.5, 0.5), 2)
                },
                'lidar': {
                    'error_rate': round(self.error_rate, 4),
                    'range_deviation': round(self.range_deviation, 4),
                    'point_cloud_integrity': round(self.point_cloud_integrity, 2),
                    'scan_counter': int(time.time() * 10) % 1000,
                    'max_detection_range': 50.0
                },
                'battery': {
                    'capacity': round(self.battery_level, 2),
                    'is_charging': False,
                    'voltage': round(24.0 + random.uniform(-0.5, 0.5), 2),
                    'temperature': round(self.battery_temp, 2)
                }
            }

            # Generate poses with proper format
            poses = []
            for i in range(3):  # Generate last 3 poses
                t = current_time - (2 - i) * 0.1  # 100ms between poses
                noise_x = random.uniform(-0.01, 0.01)
                noise_y = random.uniform(-0.01, 0.01)
                noise_theta = random.uniform(-0.01, 0.01)
                
                poses.append({
                    'timestamp': datetime.fromtimestamp(t).isoformat(),
                    'px': round(self.current_x + noise_x, 4),
                    'py': round(self.current_y + noise_y, 4),
                    'theta': round(self.current_theta + noise_theta, 4)
                })
            telemetry['poses'] = poses

            return telemetry
        except Exception as e:
            logger.error(f"Error generating telemetry data: {str(e)}")
            return None

    def _update_battery(self, dt):
        """Update battery state based on robot activity."""
        try:
            # Battery discharge rate based on activity
            base_discharge_rate = 0.5  # %/hour
            movement_factor = abs(self.current_speed) * 2.0  # More discharge when moving
            discharge_rate = (base_discharge_rate + movement_factor) * (dt / 3600)
            
            # Update battery level with some randomness
            discharge_variation = random.uniform(0.8, 1.2)  # ±20% variation
            self.battery_level = max(0.0, min(100.0, self.battery_level - discharge_rate * discharge_variation))
            
            # Update battery temperature with environmental factors
            ambient_temp = 22.0 + random.uniform(-2, 2)  # Varying ambient temperature
            activity_heat = abs(self.current_speed) * 5.0 + random.uniform(0, 1)  # Movement generates heat
            temp_change = (ambient_temp - self.battery_temp + activity_heat) * (dt / 60)
            self.battery_temp = max(ambient_temp, min(45.0, self.battery_temp + temp_change))
        except Exception as e:
            logger.error(f"Error updating battery state: {str(e)}")

    def _update_lidar(self, dt):
        """Update LIDAR state based on robot activity."""
        try:
            # Base error rate varies with speed and environmental factors
            base_error = 0.02 + random.uniform(-0.005, 0.005)
            speed_factor = abs(self.current_speed) * 0.05
            self.error_rate = min(0.1, base_error + speed_factor)
            
            # Range deviation affected by speed and vibration
            vibration = random.uniform(0, 0.01) * abs(self.current_speed)
            self.range_deviation = min(0.05, 0.01 + speed_factor + vibration)
            
            # Point cloud integrity decreases with speed and environmental factors
            base_integrity = 98.0
            integrity_loss = abs(self.current_speed) * 5.0
            environmental_factor = random.uniform(-1, 1)  # Random environmental effects
            self.point_cloud_integrity = max(80.0, base_integrity - integrity_loss + environmental_factor)
        except Exception as e:
            logger.error(f"Error updating LIDAR state: {str(e)}")

    def _update_movement(self, dt):
        """Update robot position and movement state."""
        try:
            pattern = self.movement_pattern[self.current_pattern_index]
            
            # Update position with some randomness
            self.current_speed = pattern['velocity'] * random.uniform(0.9, 1.1)  # ±10% speed variation
            self.current_theta += pattern['turn_rate'] * dt * random.uniform(0.95, 1.05)  # ±5% turn variation
            
            # Calculate new position with noise
            dx = self.current_speed * math.cos(self.current_theta) * dt
            dy = self.current_speed * math.sin(self.current_theta) * dt
            position_noise = random.uniform(-0.05, 0.05)  # Small position noise
            
            self.current_x += dx + position_noise
            self.current_y += dy + position_noise
            
            # Update pattern index periodically
            if time.time() - self.last_update > pattern['duration']:
                self.current_pattern_index = (self.current_pattern_index + 1) % len(self.movement_pattern)
                self.last_update = time.time()
        except Exception as e:
            logger.error(f"Error updating movement: {str(e)}")
            return

    def _generate_poses(self):
        """Generate a list of recent poses."""
        try:
            current_time = time.time()
            poses = []
            for i in range(3):  # Generate last 3 poses
                t = current_time - (2 - i) * 0.1  # 100ms between poses
                noise_x = random.uniform(-0.01, 0.01)
                noise_y = random.uniform(-0.01, 0.01)
                noise_theta = random.uniform(-0.01, 0.01)
                
                poses.append({
                    'timestamp': datetime.fromtimestamp(t).isoformat(),
                    'px': round(self.current_x + noise_x, 4),
                    'py': round(self.current_y + noise_y, 4),
                    'theta': round(self.current_theta + noise_theta, 4)
                })
            return poses
        except Exception as e:
            logger.error(f"Error generating poses: {str(e)}")
            return []


class RoboticTelemetryAnalyzer:
    # Class-level lock for risk scores file
    _risk_scores_file_lock = threading.Lock()
    
    def __init__(self, robot_id, robot_name=None, risk_scores_file="risk_scores.json", risk_score_buffer=None):
        # Basic initialization
        self.robot_id = robot_id
        self.robot_name = robot_name
        self.risk_scores = []

        # Configuration parameters
        self.buffer_size = 100
        self.update_interval = 0.5
        self.window_size = 50
        self.risk_scores_file = risk_scores_file
        self.risk_score_buffer = risk_score_buffer
        
        # Thread control
        self.thread = None
        self._stop_event = threading.Event()
        self.running = False
        
        # ML parameters
        self.min_samples_for_training = 3
        self.is_model_fitted = False
        
        # Timestamps
        self.previous_timestamp = None
        self.latest_timestamp = None
        
        # Radar configuration
        self.radar_range = 100.0
        self.range_resolution = 0.05
        self.num_samples = 2000
        self.noise_floor = -95.0
        self.snr_threshold = 12.0
        self.guard_cells = 4
        self.training_cells = 16
        self.pfa = 1e-7
        
        # Initialize data structures and locks
        self.data_buffer = deque(maxlen=self.window_size)
        self.simulated_data_buffer = deque(maxlen=self.buffer_size)
        self.manual_data_buffer = deque(maxlen=self.buffer_size)
        self.buffer_lock = threading.Lock()
        self.lock = threading.Lock()

        # Risk thresholds
        self.battery_thresholds = {
            'critical_low': 20.0,
            'warning_low': 30.0,
            'min_voltage': 22.0,
            'max_voltage': 29.0,
            'max_temperature': 45.0
        }

        self.lidar_thresholds = {
            'max_error_rate': 0.1,
            'max_range_deviation': 0.05,
            'min_point_cloud_integrity': 90.0
        }
        
        self.speed_thresholds = {
            'warning': 1.5,
            'critical': 2.0
        }
        
        self.risk_weights = {
            'speed': 0.25,
            'battery': 0.20,
            'lidar': 0.20,
            'position_jumps': 0.15,
            'radar': 0.20
        }
        
        # Position jump detection
        self.position_history = deque(maxlen=10)
        self.jump_count = 0
        self.jump_threshold = 0.3
        self.consecutive_jumps_threshold = 3
        
        # Initialize ML components
        self.scaler = StandardScaler()
        self.isolation_forest = IsolationForest(
            contamination=0.05,
            n_estimators=200,
            max_samples='auto',
            random_state=42
        )
        
        # Get robot name from registry
        try:
            from robot_registry import RobotRegistry
            registry = RobotRegistry()
            robot_details = registry.get_robot_details(robot_id)
            self.robot_name = robot_details.get('name', 'Unknown Robot')
        except Exception as e:
            logger.error(f"Error getting robot name: {e}")
            self.robot_name = 'Unknown Robot'
        
        # Initialize data source
        self.data_source = RealTimeDataSimulator(robot_id)

        # Results tracking
        self.analysis_results = queue.Queue()
        self.risk_thresholds = {'low': 0.3, 'high': 0.7}
        
        # Load initial data
        self.risk_scores = self.load_risk_scores()

        # Ensure risk scores file exists and is writable
        try:
            directory = os.path.dirname(os.path.abspath(self.risk_scores_file))
            os.makedirs(directory, exist_ok=True)
            if not os.path.exists(self.risk_scores_file):
                with open(self.risk_scores_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
        except Exception as e:
            logger.error(f"Error initializing risk scores file: {str(e)}")
        
        # Inject initial training data
        self._inject_initial_training_data()

    def load_risk_scores(self):
        """Load risk scores from file with proper error handling."""
        try:
            if not os.path.exists(self.risk_scores_file):
                logger.info(f"Risk scores file not found, creating new one at {self.risk_scores_file}")
                os.makedirs(os.path.dirname(os.path.abspath(self.risk_scores_file)), exist_ok=True)
                with open(self.risk_scores_file, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
                return {}

            with self._risk_scores_file_lock:
                try:
                    with open(self.risk_scores_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                        if not isinstance(data, dict):
                            logger.error("Invalid risk scores file format")
                            return {}
                        return data
                except json.JSONDecodeError:
                    logger.warning("Corrupted risk scores file detected, creating new one")
                    return {}
                except Exception as e:
                    logger.error(f"Error reading risk scores file: {str(e)}")
                    return {}
        except Exception as e:
            logger.error(f"Error loading risk scores: {str(e)}")
            return {}

    def save_risk_scores(self, result):
        """Save risk scores to file with proper locking."""
        if not result:
            return

        try:
            # Ensure the directory exists
            directory = os.path.dirname(os.path.abspath(self.risk_scores_file))
            os.makedirs(directory, exist_ok=True)

            # Read existing data with proper locking
            with self._risk_scores_file_lock:
                data = {}
                if os.path.exists(self.risk_scores_file):
                    try:
                        with open(self.risk_scores_file, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                            except json.JSONDecodeError:
                                logger.warning("Corrupted risk scores file detected, starting fresh")
                                data = {}
                    except Exception as e:
                        logger.error(f"Error reading risk scores: {str(e)}")
                        data = {}

                # Initialize list for robot if not exists
                if self.robot_id not in data:
                    data[self.robot_id] = []
                
                # Append new data point
                data[self.robot_id].append(result)
                
                # Keep only last 1000 entries per robot to manage file size
                if len(data[self.robot_id]) > 1000:
                    data[self.robot_id] = data[self.robot_id][-1000:]

                # Write back to file with proper formatting
                try:
                    with open(self.risk_scores_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.write('\n')  # Add newline at end of file
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                except Exception as e:
                    logger.error(f"Error writing risk scores: {str(e)}")
                    
        except Exception as e:
            logger.error(f"Error saving risk scores: {str(e)}")

    def cleanup(self):
        """Cleanup resources before shutdown"""
        try:
            # Save final risk scores
            with self._risk_scores_file_lock:
                try:
                    # Load existing data
                    data = {}
                    if os.path.exists(self.risk_scores_file):
                        with open(self.risk_scores_file, 'r', encoding='utf-8') as f:
                            try:
                                data = json.load(f)
                            except json.JSONDecodeError:
                                logger.warning("Corrupted risk scores file detected, starting fresh")
                                data = {}
                    
                    # Initialize list for robot if not exists
                    if self.robot_id not in data:
                        data[self.robot_id] = []
                    
                    # Get any unsaved scores from the analysis results queue
                    while not self.analysis_results.empty():
                        try:
                            result = self.analysis_results.get_nowait()
                            if result:
                                data[self.robot_id].append(result)
                        except queue.Empty:
                            break
                    
                    # Keep last 1000 entries
                    if len(data[self.robot_id]) > 1000:
                        data[self.robot_id] = data[self.robot_id][-1000:]
                    
                    # Write back to file with proper formatting
                    with open(self.risk_scores_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.write('\n')  # Add newline at end of file
                        f.flush()
                        os.fsync(f.fileno())  # Ensure data is written to disk
                        
                    logger.info(f"Final risk scores saved for robot {self.robot_id}")
                except Exception as e:
                    logger.error(f"Error saving final risk scores: {e}")
                    
            # Clear any temporary files
            pid = os.getpid()
            temp_file = f"risk_scores_{pid}.tmp"
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except Exception as e:
                    logger.error(f"Error cleaning up temp file: {e}")
                    
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
        finally:
            # Clear all buffers and queues
            self.risk_scores = []
            self.data_buffer.clear()
            self.simulated_data_buffer.clear()
            self.manual_data_buffer.clear()
            while not self.analysis_results.empty():
                try:
                    self.analysis_results.get_nowait()
                except queue.Empty:
                    break
            logger.info(f"Cleanup completed for robot {self.robot_id}")

    def _inject_initial_training_data(self):
        """Initialize the model with some training data"""
        try:
            initial_data_list = []
            for _ in range(self.min_samples_for_training):
                initial_data = self.data_source.generate_telemetry_data()
                if initial_data:
                    self._analyze_data(initial_data)
                    initial_data_list.append(initial_data)
        except Exception as e:
            logger.error(f"Error injecting initial training data: {e}")
            
    def start(self):
        """Start the analyzer thread"""
        if not self.running:
            self._stop_event.clear()
            self.running = True
            self.thread = threading.Thread(target=self._run)
            self.thread.daemon = True
            self.thread.start()
            
    def stop(self):
        """Stop the analyzer thread and cleanup resources"""
        try:
            if self.running:
                logger.info(f"Stopping analyzer for robot {self.robot_id}")
                self._stop_event.set()
                self.running = False
                
                # Wait for thread to finish with a reasonable timeout
                if self.thread and self.thread.is_alive():
                    self.thread.join(timeout=5)
                    if self.thread.is_alive():
                        logger.warning("Thread did not stop within timeout")
                
                # Save final data and cleanup
                self.cleanup()
                
                logger.info(f"Analyzer stopped for robot {self.robot_id}")
        except Exception as e:
            logger.error(f"Error stopping analyzer for robot {self.robot_id}: {e}")
        finally:
            self.running = False

    def _run(self):
        """Main analysis loop"""
        while not self._stop_event.is_set():
            try:
                # Get latest telemetry data
                telemetry_data = self.data_source.generate_telemetry_data()
                if telemetry_data:
                    self._analyze_data(telemetry_data)
                    
                # Sleep for update interval
                time.sleep(self.update_interval)
            except Exception as e:
                logger.error(f"Error in analysis loop: {e}")
                if not self._stop_event.is_set():
                    time.sleep(1)  # Wait before retrying

    def _analyze_data(self, telemetry_data):
        """Analyze telemetry data and calculate risk scores."""
        try:
            if not isinstance(telemetry_data, dict):
                logger.error("Invalid telemetry data: not a dictionary")
                return None

            required_fields = ['robot_id', 'timestamp', 'poses', 'motion', 'power', 'lidar']
            missing_fields = [field for field in required_fields if field not in telemetry_data]
            if missing_fields:
                logger.error(f"Missing required fields in telemetry data: {missing_fields}")
                return None

            # Process radar data and calculate risk
            radar_data = self._process_radar_data(telemetry_data.get('poses', []))
            radar_risk = self._calculate_radar_risk(radar_data)

            # Calculate individual risk scores
            risk_scores = {
                'speed': self._calculate_speed_risk(telemetry_data),
                'battery': self._calculate_battery_risk(telemetry_data),
                'lidar': self._calculate_lidar_risk(telemetry_data),
                'position': self._calculate_position_risk(telemetry_data),
                'radar': radar_risk
            }

            # Calculate composite risk score
            riskscore = self._calculate_composite_risk_score(risk_scores)

            # Determine risk level
            if riskscore >= 75:
                risk_level = "High Risk"
            elif riskscore >= 45:
                risk_level = "Medium Risk"
            else:
                risk_level = "Low Risk"

            # Extract speed data
            motion_data = telemetry_data.get('motion', {})
            speed_value = motion_data.get('speed', 0)
            
            # Extract battery data
            power_data = telemetry_data.get('power', {})
            battery_status = {
                'capacity': power_data.get('battery_level', 0),
                'is_charging': power_data.get('charging_status') == 'charging',
                'voltage': power_data.get('voltage', 0),
                'temperature': power_data.get('battery_temp', 0),
                'risk_level': "High Risk" if risk_scores['battery'] >= 75 else 
                            "Medium Risk" if risk_scores['battery'] >= 45 else "Low Risk",
                'alerts': []
            }

            # Extract LIDAR data
            lidar_data = telemetry_data.get('lidar', {})
            lidar_status = {
                'error_rate': lidar_data.get('error_rate', 0),
                'range_deviation': lidar_data.get('range_deviation', 0),
                'point_cloud_integrity': lidar_data.get('point_cloud_integrity', 0),
                'max_detection_range': 50.0,  # Default value
                'risk_level': "High Risk" if risk_scores['lidar'] >= 75 else 
                            "Medium Risk" if risk_scores['lidar'] >= 45 else "Low Risk",
                'alerts': []
            }

            # Add alerts based on thresholds
            if battery_status['capacity'] < self.battery_thresholds['critical_low']:
                battery_status['alerts'].append(f"CRITICAL: Battery level critically low at {battery_status['capacity']}%")
            elif battery_status['capacity'] < self.battery_thresholds['warning_low']:
                battery_status['alerts'].append(f"WARNING: Battery level low at {battery_status['capacity']}%")

            if lidar_status['point_cloud_integrity'] < self.lidar_thresholds['min_point_cloud_integrity']:
                lidar_status['alerts'].append(f"CRITICAL: Low point cloud integrity {lidar_status['point_cloud_integrity']}%")

            # Get latest pose data
            latest_pose = None
            poses = telemetry_data.get('poses', [])
            if poses and len(poses) > 0:
                latest_pose = poses[-1]  # Get the most recent pose

            # Prepare analysis result
            result = {
                'robot_id': telemetry_data['robot_id'],
                'robot_name': self.robot_name,
                'timestamp': datetime.now().isoformat(),
                'risk_score': round(riskscore, 2),
                'riskscore': round(riskscore, 2),
                'risk_scores': {
                    'battery': risk_scores['battery'],
                    'lidar': risk_scores['lidar'],
                    'speed': risk_scores['speed'],
                    'position': risk_scores['position'],
                    'radar': risk_scores['radar']
                },
                'risk_level': risk_level,
                'battery_status': battery_status,
                'lidar_status': lidar_status,
                'radar_status': {
                    'detections': radar_data.get('detections', []),
                    'metadata': radar_data.get('metadata', {}),
                    'risk_level': "High Risk" if radar_risk >= 75 else 
                                "Medium Risk" if radar_risk >= 45 else "Low Risk",
                    'alerts': []
                },
                'speed': {
                    'value': round(speed_value, 3),
                    'unit': 'm/s',
                    'confidence': 1.0,
                    'distance': round(abs(speed_value), 3),
                    'time_diff': 1.001
                },
                'telemetry': {
                    'power': {'battery_level': power_data.get('battery_level', 0)},
                    'lidar': {'error_rate': lidar_data.get('error_rate', 0)},
                    'motion': {'speed': speed_value},
                    'position': {
                        'x': latest_pose['px'] if latest_pose else 0,
                        'y': latest_pose['py'] if latest_pose else 0,
                        'theta': latest_pose['theta'] if latest_pose else 0
                    },
                    'radar': {
                        'detections': radar_data.get('detections', []),
                        'avg_snr': radar_data.get('metadata', {}).get('avg_snr', 0),
                        'detection_density': radar_data.get('metadata', {}).get('detection_density', 0)
                    }
                },
                'position_warning': None,
                'poses': poses
            }

            # Add radar alerts based on risk
            if radar_risk >= 75:
                result['radar_status']['alerts'].append("CRITICAL: High risk of collision detected")
            elif radar_risk >= 45:
                result['radar_status']['alerts'].append("WARNING: Potential obstacles detected")

            # Save results
            self.save_risk_scores(result)
            
            # Update analysis results queue
            try:
                # Clear old results
                while not self.analysis_results.empty():
                    self.analysis_results.get_nowait()
                self.analysis_results.put_nowait(result)
            except queue.Full:
                logger.warning("Analysis results queue is full")

            return result

        except Exception as e:
            logger.error(f"Error in _analyze_data: {str(e)}")
            return None

    def _process_radar_data(self, poses):
        """Process radar data from poses using enhanced CA-CFAR detection.
        
        Implements an improved radar processing pipeline with:
        - Dynamic range resolution
        - Multi-target detection
        - SNR-based filtering
        - Doppler processing
        """
        try:
            if not poses or len(poses) < 3:
                return {'detections': [], 'status': 'error', 'message': 'Insufficient pose data'}

            # Initialize radar parameters
            num_range_bins = 256  # Increased resolution
            max_range = 10.0  # meters
            range_resolution = max_range / num_range_bins
            
            # Create range-doppler map
            range_doppler = np.zeros((num_range_bins, 2))  # [amplitude, velocity]
            
            # Process sequential poses for velocity
            for i in range(1, len(poses)):
                curr_pose = poses[i]
                prev_pose = poses[i-1]
                
                if not all(k in curr_pose for k in ['px', 'py', 'theta']):
                    continue
                
                # Calculate range and velocity
                curr_range = math.sqrt(curr_pose['px']**2 + curr_pose['py']**2)
                prev_range = math.sqrt(prev_pose['px']**2 + prev_pose['py']**2)
                
                if curr_range > max_range:
                    continue
                    
                # Calculate radial velocity
                dt = 0.1  # Assumed time step between poses
                velocity = (curr_range - prev_range) / dt
                
                # Add to range-doppler map with amplitude modeling
                range_idx = int(curr_range / range_resolution)
                if 0 <= range_idx < num_range_bins:
                    # Model radar cross section and path loss
                    amplitude = 30 * math.log10(1 / (curr_range + 0.1))  # Improved amplitude model
                    # Add environmental noise
                    noise = random.gauss(0, 2)
                    range_doppler[range_idx] = [amplitude + noise, velocity]

            # Add realistic clutter
            clutter = np.random.rayleigh(scale=2, size=num_range_bins)
            range_doppler[:, 0] = np.maximum(range_doppler[:, 0], clutter)

            # CA-CFAR Detection with adaptive thresholding
            guard_cells = 4
            training_cells = 8
            pfa = 1e-4  # Probability of false alarm
            
            detections = []
            for i in range(guard_cells + training_cells, num_range_bins - guard_cells - training_cells):
                # Get training cells excluding guard regions
                training_data = np.concatenate([
                    range_doppler[i - training_cells - guard_cells:i - guard_cells, 0],
                    range_doppler[i + guard_cells + 1:i + guard_cells + training_cells + 1, 0]
                ])
                
                # Calculate adaptive threshold
                noise_level = np.mean(training_data)
                noise_variance = np.var(training_data)
                # Use Chi-square statistics for threshold
                threshold = noise_level + np.sqrt(noise_variance) * (-2 * np.log(pfa))
                
                # Check for detection
                if range_doppler[i, 0] > threshold:
                    range_val = i * range_resolution
                    velocity = range_doppler[i, 1]
                    snr = range_doppler[i, 0] - noise_level
                    
                    # Calculate confidence based on SNR and velocity
                    confidence = min(100, max(0, (snr / 10) * 100))
                    
                    detections.append({
                        'range': float(range_val),
                        'velocity': float(velocity),
                        'amplitude': float(range_doppler[i, 0]),
                        'snr': float(snr),
                        'confidence': float(confidence)
                    })

            # Sort detections by SNR and keep top 10 most significant
            detections = sorted(detections, key=lambda x: x['snr'], reverse=True)[:10]
            
            # Calculate detection quality metrics
            avg_snr = np.mean([d['snr'] for d in detections]) if detections else 0
            detection_density = len(detections) / num_range_bins
            
            return {
                'detections': detections,
                'status': 'ok',
                'metadata': {
                    'avg_snr': float(avg_snr),
                    'detection_density': float(detection_density),
                    'range_resolution': float(range_resolution),
                    'max_range': float(max_range),
                    'timestamp': datetime.now().isoformat()
                }
            }

        except Exception as e:
            logger.error(f"Error in radar processing: {str(e)}")
            return {'detections': [], 'status': 'error', 'message': str(e)}

    def _calculate_radar_risk(self, radar_data):
        """Calculate risk based on enhanced radar detections."""
        try:
            if not radar_data or radar_data['status'] != 'ok' or not radar_data['detections']:
                return 50.0  # Default moderate risk when no data

            # Initialize risk factors
            proximity_risk = 0.0
            velocity_risk = 0.0
            detection_quality = 0.0
            
            # Process all detections with confidence weighting
            for detection in radar_data['detections']:
                # Weight based on detection confidence
                confidence_weight = detection['confidence'] / 100.0
                
                # Proximity risk calculation
                range_val = detection['range']
                if range_val < 1.0:  # Critical proximity
                    proximity_risk = max(proximity_risk, 100.0 * confidence_weight)
                elif range_val < 2.0:  # Warning proximity
                    proximity_risk = max(proximity_risk, (75.0 + (2.0 - range_val) * 25.0) * confidence_weight)
                else:
                    proximity_risk = max(proximity_risk, (50.0 * (5.0 - min(range_val, 5.0)) / 3.0) * confidence_weight)
                
                # Velocity risk calculation
                velocity = abs(detection['velocity'])
                if velocity > 2.0:  # Fast approaching/receding
                    velocity_risk = max(velocity_risk, min(100.0, velocity * 25.0) * confidence_weight)
                
            # Detection quality assessment
            metadata = radar_data.get('metadata', {})
            avg_snr = metadata.get('avg_snr', 0)
            detection_density = metadata.get('detection_density', 0)
            
            if avg_snr > 15:  # High quality detections
                detection_quality = 1.0
            elif avg_snr > 10:  # Medium quality
                detection_quality = 0.8
            else:  # Low quality
                detection_quality = 0.6
                
            # Combine risk factors with quality weighting
            final_risk = detection_quality * max(proximity_risk, velocity_risk)
            
            # Ensure risk is in valid range
            return max(0.0, min(100.0, final_risk))

        except Exception as e:
            logger.error(f"Error calculating radar risk: {str(e)}")
            return 50.0  # Default moderate risk on error

    def _calculate_speed_risk(self, telemetry_data):
        """Calculate speed risk score."""
        try:
            motion = telemetry_data.get('motion', {})
            if not isinstance(motion, dict):
                return 50.0

            # Get speed with validation
            speed = motion.get('speed')
            if speed is None:
                return 50.0
            
            current_speed = abs(float(speed))
            
            # Calculate risk based on thresholds
            if current_speed >= self.speed_thresholds['critical']:
                return 100.0
            elif current_speed >= self.speed_thresholds['warning']:
                # Linear interpolation between warning and critical
                risk_range = 100.0 - 60.0
                speed_range = self.speed_thresholds['critical'] - self.speed_thresholds['warning']
                return 60.0 + (current_speed - self.speed_thresholds['warning']) * (risk_range / speed_range)
            else:
                # Linear risk up to warning speed
                return (current_speed / self.speed_thresholds['warning']) * 60.0

        except Exception as e:
            logger.error(f"Error in speed risk calculation: {str(e)}")
            return 50.0

    def _calculate_battery_risk(self, telemetry_data):
        """Calculate battery risk score."""
        try:
            power = telemetry_data.get('power', {})
            if not isinstance(power, dict):
                return 50.0

            # Get battery metrics with validation
            battery_level = power.get('battery_level')
            battery_temp = power.get('battery_temp')

            if battery_level is None or battery_temp is None:
                return 50.0

            battery_level = float(battery_level)
            battery_temp = float(battery_temp)
            
            risks = []
            
            # Capacity risk
            if battery_level <= self.battery_thresholds['critical_low']:
                risks.append(100.0)
            elif battery_level <= self.battery_thresholds['warning_low']:
                risk_range = 100.0 - 60.0
                level_range = self.battery_thresholds['warning_low'] - self.battery_thresholds['critical_low']
                risks.append(60.0 + (self.battery_thresholds['warning_low'] - battery_level) * (risk_range / level_range))
            else:
                risks.append(20.0)

            # Temperature risk
            if battery_temp >= self.battery_thresholds['max_temperature']:
                risks.append(100.0)
            elif battery_temp >= (self.battery_thresholds['max_temperature'] - 10):
                risk_range = 100.0 - 60.0
                temp_range = 10.0
                risks.append(60.0 + (battery_temp - (self.battery_thresholds['max_temperature'] - 10)) * (risk_range / temp_range))
            else:
                risks.append(20.0)

            return max(risks) if risks else 50.0

        except Exception as e:
            logger.error(f"Error in battery risk calculation: {str(e)}")
            return 50.0

    def _calculate_lidar_risk(self, telemetry_data):
        """Calculate risk score based on LIDAR status."""
        try:
            lidar = telemetry_data.get('lidar', {})
            if not isinstance(lidar, dict):
                return 50.0

            # Get LIDAR metrics with validation
            error_rate = lidar.get('error_rate')
            range_dev = lidar.get('range_deviation')
            integrity = lidar.get('point_cloud_integrity')

            if error_rate is None or range_dev is None or integrity is None:
                return 50.0

            error_rate = float(error_rate)
            range_dev = float(range_dev)
            integrity = float(integrity)
            
            risks = []
            
            # Error rate risk
            if error_rate > self.lidar_thresholds['max_error_rate']:
                risks.append(100.0)
            else:
                risks.append((error_rate / self.lidar_thresholds['max_error_rate']) * 80.0)

            # Range deviation risk
            if range_dev > self.lidar_thresholds['max_range_deviation']:
                risks.append(100.0)
            else:
                risks.append((range_dev / self.lidar_thresholds['max_range_deviation']) * 80.0)

            # Point cloud integrity risk
            if integrity < self.lidar_thresholds['min_point_cloud_integrity']:
                risks.append(100.0)
            else:
                risks.append((1 - (integrity / 100.0)) * 80.0)

            return max(risks) if risks else 50.0

        except Exception as e:
            logger.error(f"Error in LIDAR risk calculation: {str(e)}")
            return 50.0

    def _calculate_position_risk(self, telemetry_data):
        """Calculate position risk based on position jumps."""
        try:
            poses = telemetry_data.get('poses', [])
            if not isinstance(poses, list) or len(poses) < 2:
                return 50.0

            # Check each consecutive pair of poses
            for i in range(len(poses) - 1):
                curr_pose = poses[i]
                next_pose = poses[i + 1]

                # Validate pose structure
                if not all(key in curr_pose and key in next_pose for key in ['px', 'py', 'theta']):
                    continue

                # Calculate position change
                dx = abs(next_pose['px'] - curr_pose['px'])
                dy = abs(next_pose['py'] - curr_pose['py'])
                distance = math.sqrt(dx * dx + dy * dy)

                # Check for sudden jumps
                if distance > self.jump_threshold:
                    return 100.0

            return 0.0

        except Exception as e:
            logger.error(f"Error in position risk calculation: {str(e)}")
            return 50.0

    def _calculate_composite_risk_score(self, risk_scores):
        """Calculate composite risk score from individual risk scores."""
        try:
            if not isinstance(risk_scores, dict):
                logger.error("Invalid risk scores format")
                return 50.0

            weighted_sum = 0.0
            total_weight = 0.0

            # Calculate weighted sum of risk scores
            for factor, weight in self.risk_weights.items():
                if factor in risk_scores:
                    try:
                        score = float(risk_scores[factor])
                        weighted_sum += score * weight
                        total_weight += weight
                    except (ValueError, TypeError) as e:
                        logger.error(f"Error processing risk score for {factor}: {str(e)}")
                        continue

            # Calculate final composite score
            if total_weight > 0:
                composite_score = weighted_sum / total_weight
            else:
                logger.warning("No valid risk scores found for composite calculation")
                composite_score = 50.0

            # Ensure score is within valid range
            return min(100.0, max(0.0, composite_score))

        except Exception as e:
            logger.error(f"Error calculating composite risk score: {str(e)}")
            return 50.0

    def get_latest_results(self):
        """Get the latest analysis results."""
        try:
            if not self.analysis_results.empty():
                result = self.analysis_results.get_nowait()
                if result and isinstance(result, dict):
                    # Add required fields if missing
                    result.setdefault('timestamp', datetime.now().isoformat())
                    result.setdefault('robot_id', self.robot_id)
                    result.setdefault('risk_scores', {})
                    result.setdefault('riskscore', 0.0)
                    
                    # Add risk level to result
                    risk_score = result['riskscore']
                    if risk_score >= 75:
                        risk_level = 'Critical'
                    elif risk_score >= 50:
                        risk_level = 'High'
                    elif risk_score >= 25:
                        risk_level = 'Medium'
                    else:
                        risk_level = 'Low'
                    
                    result['risk_level'] = risk_level
                    result['risk_score'] = round(risk_score, 2)
                    
                    # Add telemetry data if missing
                    if 'telemetry' not in result:
                        result['telemetry'] = {
                            'power': {'battery_level': 0},
                            'lidar': {'error_rate': 0},
                            'motion': {'speed': 0},
                            'position': {'x': 0, 'y': 0, 'theta': 0}
                        }
                    
                    return result
                else:
                    logger.error("Invalid analysis result format")
            return None
        except Exception as e:
            logger.error(f"Error getting latest results: {str(e)}")
            return None

    def _send_to_api(self, prediction):
        try:
            response = requests.post(
                f"{self.api_url}/api/risk-scores/push",
                json=prediction,
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Error sending data to API: {e}")

    def analyze_telemetry(self, telemetry_data: Dict) -> Dict:
        """Analyze telemetry data and return risk assessment"""
        if not telemetry_data:
            return None

        try:
            robot_id = telemetry_data.get('robot_id')
            if not robot_id:
                logger.error("Missing robot_id in telemetry data")
                return None

            # Get robot details including name
            robot_name = "Unknown"
            try:
                from robot_registry import RobotRegistry
                registry = RobotRegistry()
                robot_details = registry.get_robot_details(robot_id)
                if robot_details and robot_details.get('name'):
                    robot_name = robot_details['name']
            except Exception as e:
                logger.error(f"Error getting robot name: {e}")

            timestamp = telemetry_data.get('timestamp', datetime.now().isoformat())
            battery_data = telemetry_data.get('battery', {})
            lidar_data = telemetry_data.get('lidar', {})

            # Analyze battery status
            battery_status = self._analyze_battery(battery_data)
            
            # Analyze LIDAR status
            lidar_status = self._analyze_lidar(lidar_data)
            
            # Determine overall risk level
            risk_levels = {
                'HIGH': 3,
                'MEDIUM': 2,
                'LOW': 1
            }
            
            battery_risk = risk_levels.get(battery_status['risk_level'], 0)
            lidar_risk = risk_levels.get(lidar_status['risk_level'], 0)
            
            overall_risk = max(battery_risk, lidar_risk)
            risk_level = {v: k for k, v in risk_levels.items()}.get(overall_risk, 'UNKNOWN')

            # Compile results with robot name
            result = {
                'robot_id': robot_id,
                'robot_name': robot_name,
                'timestamp': timestamp,
                'risk_level': risk_level,
                'battery_status': battery_status,
                'lidar_status': lidar_status,
                'alerts': battery_status.get('alerts', []) + lidar_status.get('alerts', [])
            }

            # Save risk scores with robot name
            if self.risk_score_buffer:
                self.risk_score_buffer.add_score(robot_id, result)
                logger.debug(f"Saved risk score for robot {robot_id} ({robot_name})")

            return result

        except Exception as e:
            logger.error(f"Error analyzing telemetry data: {e}")
            return None

    def analyze_manual_data(self, telemetry_data: Dict) -> Dict:
        """Analyze manually input data"""
        if not telemetry_data:
            return None

        try:
            robot_id = telemetry_data.get('robot_id', 'default')
            
            # Get robot details including name
            robot_name = "Unknown"
            try:
                from robot_registry import RobotRegistry
                registry = RobotRegistry()
                robot_details = registry.get_robot_details(robot_id)
                if robot_details and robot_details.get('name'):
                    robot_name = robot_details['name']
            except Exception as e:
                logger.error(f"Error getting robot name: {e}")

            timestamp = telemetry_data.get('timestamp', datetime.now().isoformat())
            battery_data = telemetry_data.get('battery', {})
            lidar_data = telemetry_data.get('lidar', {})

            # Analyze components
            battery_status = self._analyze_battery(battery_data)
            lidar_status = self._analyze_lidar(lidar_data)
            
            # Determine overall risk level
            risk_levels = {
                'HIGH': 3,
                'MEDIUM': 2,
                'LOW': 1
            }
            
            battery_risk = risk_levels.get(battery_status['risk_level'], 0)
            lidar_risk = risk_levels.get(lidar_status['risk_level'], 0)
            
            overall_risk = max(battery_risk, lidar_risk)
            risk_level = {v: k for k, v in risk_levels.items()}.get(overall_risk, 'UNKNOWN')

            # Compile results
            result = {
                'robot_id': robot_id,
                'robot_name': robot_name,
                'timestamp': timestamp,
                'risk_level': risk_level,
                'battery_status': battery_status,
                'lidar_status': lidar_status,
                'alerts': battery_status.get('alerts', []) + lidar_status.get('alerts', [])
            }

            # Save risk scores
            if self.risk_score_buffer:
                self.risk_score_buffer.add_score(robot_id, result)
                logger.debug(f"Saved manual risk score for robot {robot_id} ({robot_name})")

            return result

        except Exception as e:
            logger.error(f"Error analyzing manual data: {e}")
            return None

    def _analyze_battery(self, battery_data):
        """Analyze battery data and return status with alerts."""
        try:
            if not battery_data:
                return {
                    'capacity': 0,
                    'is_charging': False,
                    'voltage': 0,
                    'temperature': 0,
                    'risk_level': 'HIGH',
                    'alerts': ['No battery data available']
                }

            # Extract battery metrics
            capacity = battery_data.get('capacity', 0)
            voltage = battery_data.get('voltage', 0)
            temperature = battery_data.get('temperature', 0)
            is_charging = battery_data.get('is_charging', False)

            alerts = []
            risk_level = 'LOW'

            # Check battery level
            if capacity < self.battery_thresholds['critical_low']:
                alerts.append(f"CRITICAL: Battery level critically low at {capacity}%")
                risk_level = 'HIGH'
            elif capacity < self.battery_thresholds['warning_low']:
                alerts.append(f"WARNING: Battery level low at {capacity}%")
                risk_level = 'MEDIUM'

            # Check voltage
            if voltage < self.battery_thresholds['min_voltage']:
                alerts.append(f"CRITICAL: Battery voltage low at {voltage}V")
                risk_level = 'HIGH'
            elif voltage > self.battery_thresholds['max_voltage']:
                alerts.append(f"CRITICAL: Battery voltage high at {voltage}V")
                risk_level = 'HIGH'

            # Check temperature
            if temperature > self.battery_thresholds['max_temperature']:
                alerts.append(f"CRITICAL: Battery temperature high at {temperature}°C")
                risk_level = 'HIGH'

            return {
                'capacity': capacity,
                'is_charging': is_charging,
                'voltage': voltage,
                'temperature': temperature,
                'risk_level': risk_level,
                'alerts': alerts
            }

        except Exception as e:
            logger.error(f"Error analyzing battery data: {str(e)}")
            return {
                'capacity': 0,
                'is_charging': False,
                'voltage': 0,
                'temperature': 0,
                'risk_level': 'HIGH',
                'alerts': ['Error analyzing battery data']
            }

    def _analyze_lidar(self, lidar_data):
        """Analyze LIDAR data and return status with alerts."""
        try:
            if not lidar_data:
                return {
                    'error_rate': 0,
                    'range_deviation': 0,
                    'point_cloud_integrity': 0,
                    'max_detection_range': 50.0,
                    'risk_level': 'HIGH',
                    'alerts': ['No LIDAR data available']
                }

            # Extract LIDAR metrics
            error_rate = lidar_data.get('error_rate', 0)
            range_deviation = lidar_data.get('range_deviation', 0)
            point_cloud_integrity = lidar_data.get('point_cloud_integrity', 0)
            max_detection_range = lidar_data.get('max_detection_range', 50.0)

            alerts = []
            risk_level = 'LOW'

            # Check error rate
            if error_rate > self.lidar_thresholds['max_error_rate']:
                alerts.append(f"CRITICAL: High LIDAR error rate at {error_rate}")
                risk_level = 'HIGH'

            # Check range deviation
            if range_deviation > self.lidar_thresholds['max_range_deviation']:
                alerts.append(f"WARNING: High range deviation at {range_deviation}")
                risk_level = max(risk_level, 'MEDIUM')

            # Check point cloud integrity
            if point_cloud_integrity < self.lidar_thresholds['min_point_cloud_integrity']:
                alerts.append(f"CRITICAL: Low point cloud integrity at {point_cloud_integrity}%")
                risk_level = 'HIGH'

            return {
                'error_rate': error_rate,
                'range_deviation': range_deviation,
                'point_cloud_integrity': point_cloud_integrity,
                'max_detection_range': max_detection_range,
                'risk_level': risk_level,
                'alerts': alerts
            }

        except Exception as e:
            logger.error(f"Error analyzing LIDAR data: {str(e)}")
            return {
                'error_rate': 0,
                'range_deviation': 0,
                'point_cloud_integrity': 0,
                'max_detection_range': 50.0,
                'risk_level': 'HIGH',
                'alerts': ['Error analyzing LIDAR data']
            }

def get_latest_risk_scores(robot_id, analyzers):
    """Get latest risk scores for a specific robot."""
    try:
        for analyzer in analyzers:
            if analyzer.robot_id == robot_id:
                latest_result = analyzer.get_latest_results()
                if latest_result:
                    risk_score = latest_result.get('riskscore', 0)
                    risk_level = latest_result.get('risk_level', 'Unknown')
                    risk_scores = latest_result.get('risk_scores', {})
                    telemetry = latest_result.get('telemetry', {})
                    radar_status = latest_result.get('radar_status', {})
                    
                    return {
                        'risk_score': round(risk_score, 2),
                        'risk_level': risk_level,
                        'timestamp': latest_result.get('timestamp'),
                        'battery_status': {
                            'level': telemetry.get('power', {}).get('battery_level', 0),
                            'risk': risk_scores.get('battery', 0)
                        },
                        'lidar_status': {
                            'error_rate': telemetry.get('lidar', {}).get('error_rate', 0),
                            'risk': risk_scores.get('lidar', 0)
                        },
                        'radar_status': {
                            'detections': radar_status.get('detections', []),
                            'avg_snr': telemetry.get('radar', {}).get('avg_snr', 0),
                            'detection_density': telemetry.get('radar', {}).get('detection_density', 0),
                            'risk': risk_scores.get('radar', 0),
                            'alerts': radar_status.get('alerts', [])
                        },
                        'speed': {
                            'value': telemetry.get('motion', {}).get('speed', 0),
                            'unit': 'm/s',
                            'risk': risk_scores.get('speed', 0)
                        },
                        'position': {
                            'risk': risk_scores.get('position', 0)
                        }
                    }
        
        # If no analyzer found for robot_id, return default values
        return {
            'risk_score': 0.0,
            'risk_level': 'Unknown',
            'timestamp': datetime.now().isoformat(),
            'battery_status': {'level': 0, 'risk': 0},
            'lidar_status': {'error_rate': 0, 'risk': 0},
            'radar_status': {
                'detections': [],
                'avg_snr': 0,
                'detection_density': 0,
                'risk': 0,
                'alerts': []
            },
            'speed': {'value': 0.0, 'unit': 'm/s', 'risk': 0},
            'position': {'risk': 0}
        }
    except Exception as e:
        logger.error(f"Error getting latest risk scores: {str(e)}")
    return None

