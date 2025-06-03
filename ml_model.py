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
from datetime import datetime, timedelta
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
            'warning': 2.0,    # m/s
            'critical': 5.0    # m/s
        }
        
        self.position_thresholds = {
            'warning': 10.0,   # meters from origin
            'critical': 20.0   # meters from origin
        }
        
        self.jump_threshold = 5.0  # meters (for sudden position changes)
        
        # Position jump detection
        self.position_history = deque(maxlen=10)
        self.jump_count = 0
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
        
        # Initialize optimization components
        self.feature_selector = None
        self.selected_features = None
        self.optimized_weights = None
        self.study = None
        self.min_optimization_samples = 50
        self.optimization_history = deque(maxlen=1000)
        
        # Initialize weights with default values
        self.weights = {
            'speed': 0.2,
            'position': 0.2,
            'battery': 0.2,
            'lidar': 0.2,
            'radar': 0.2,
            'performance_score': 0.0,
            'last_updated': time.time()
        }
        
        # Initialize risk trends and confidence tracking
        self.risk_trends = {
            factor: deque(maxlen=10) 
            for factor in ['battery', 'lidar', 'speed', 'position', 'radar']
        }
        self.last_update_time = {}
        self.confidence_history = {
            factor: deque(maxlen=5) 
            for factor in ['battery', 'lidar', 'speed', 'position', 'radar']
        }
        
        # Inject initial training data
        self._inject_initial_training_data()

    def _optimize_weights_with_bayesian(self, historical_data):
        """Optimize weights using Bayesian optimization."""
        try:
            import optuna
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            
            if len(historical_data) < self.min_optimization_samples:
                logger.warning("Insufficient data for Bayesian optimization")
                return self.weights
            
            # Extract features and target
            X = []
            y = []
            
            for entry in historical_data:
                features = [
                    entry.get('speed', {}).get('value', 0),
                    entry.get('position', {}).get('distance', 0),
                    entry.get('battery_status', {}).get('capacity', 0),
                    entry.get('lidar_status', {}).get('error_rate', 0),
                    entry.get('radar_status', {}).get('detection_density', 0)
                ]
                X.append(features)
                y.append(entry.get('risk_score', 50.0))
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            def objective(trial):
                # Suggest weights ensuring they sum to 1
                w1 = trial.suggest_float("speed", 0.1, 0.4)
                w2 = trial.suggest_float("position", 0.2, 0.5)
                w3 = trial.suggest_float("battery", 0.1, 0.3)
                w4 = trial.suggest_float("lidar", 0.1, 0.3)
                w5 = 1.0 - (w1 + w2 + w3 + w4)
                
                if w5 < 0.1 or w5 > 0.3:  # Validate radar weight range
                    return float('inf')
                
                # Calculate weighted predictions
                y_pred = (w1 * X_train[:, 0] + 
                         w2 * X_train[:, 1] + 
                         w3 * X_train[:, 2] + 
                         w4 * X_train[:, 3] + 
                         w5 * X_train[:, 4])
                
                # Calculate MSE with regularization
                mse = mean_squared_error(y_train, y_pred)
                regularization = abs(w1 - self.weights.get('speed', 0.3)) + \
                               abs(w2 - self.weights.get('position', 0.3)) + \
                               abs(w3 - self.weights.get('battery', 0.2)) + \
                               abs(w4 - self.weights.get('lidar', 0.1)) + \
                               abs(w5 - self.weights.get('radar', 0.1))
                
                return mse + 0.1 * regularization
            
            # Create and optimize study
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=100)
            
            # Get best weights
            best_params = study.best_params
            new_weights = {
                'speed': best_params['speed'],
                'position': best_params['position'],
                'battery': best_params['battery'],
                'lidar': best_params['lidar'],
                'radar': 1.0 - sum(best_params.values()),
                'performance_score': 1.0 / study.best_value,
                'last_updated': time.time()
            }
            
            # Validate new weights
            if self._validate_weights(new_weights):
                self.weights = new_weights
                self._save_weights()
                logger.info(f"Updated weights via Bayesian optimization: {self.weights}")
            
            return new_weights
            
        except Exception as e:
            logger.error(f"Error in Bayesian optimization: {e}")
            return self.weights
    
    def _select_features(self, historical_data):
        """Select most important features using RFE."""
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import GradientBoostingRegressor
            
            if len(historical_data) < self.min_optimization_samples:
                return None
            
            # Prepare features
            features = []
            targets = []
            
            for entry in historical_data:
                feature_vector = [
                    entry.get('speed', {}).get('value', 0),
                    entry.get('position', {}).get('distance', 0),
                    entry.get('battery_status', {}).get('capacity', 0),
                    entry.get('lidar_status', {}).get('error_rate', 0),
                    entry.get('radar_status', {}).get('detection_density', 0),
                    entry.get('position', {}).get('angle_diff', 0),
                    entry.get('battery_status', {}).get('temperature', 0),
                    entry.get('lidar_status', {}).get('point_cloud_integrity', 0)
                ]
                features.append(feature_vector)
                targets.append(entry.get('risk_score', 50.0))
            
            X = np.array(features)
            y = np.array(targets)
            
            # Initialize estimator
            estimator = GradientBoostingRegressor(n_estimators=100, random_state=42)
            
            # Initialize RFE with 5 features
            selector = RFE(estimator=estimator, n_features_to_select=5, step=1)
            selector = selector.fit(X, y)
            
            # Get selected feature indices
            selected_features = selector.support_
            
            # Store selected features
            self.feature_selector = selector
            self.selected_features = selected_features
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in feature selection: {e}")
            return None
    
    def _select_stable_features(self, historical_data, n_iterations=10):
        """Select stable features using repeated RFE with stability selection."""
        try:
            from sklearn.feature_selection import RFE
            from sklearn.ensemble import GradientBoostingRegressor
            from sklearn.preprocessing import StandardScaler
            import numpy as np
            
            if len(historical_data) < self.min_optimization_samples:
                return None
                
            # Define feature names
            feature_names = [
                'speed', 'position', 'battery', 'lidar', 'radar',
                'angle_deviation', 'temperature', 'point_cloud_integrity'
            ]
            
            # Initialize stability scores
            stability_scores = {name: 0 for name in feature_names}
            
            for _ in range(n_iterations):
                # Prepare features with random subsampling
                features = []
                targets = []
                
                # Random subsampling of historical data
                indices = np.random.choice(
                    len(historical_data), 
                    size=min(len(historical_data), 100), 
                    replace=False
                )
                
                for idx in indices:
                    entry = historical_data[idx]
                    feature_vector = [
                        entry.get('speed', {}).get('value', 0),
                        entry.get('position', {}).get('distance', 0),
                        entry.get('battery_status', {}).get('capacity', 0),
                        entry.get('lidar_status', {}).get('error_rate', 0),
                        entry.get('radar_status', {}).get('detection_density', 0),
                        entry.get('position', {}).get('angle_diff', 0),
                        entry.get('battery_status', {}).get('temperature', 0),
                        entry.get('lidar_status', {}).get('point_cloud_integrity', 0)
                    ]
                    features.append(feature_vector)
                    targets.append(entry.get('risk_score', 50.0))
                
                X = np.array(features)
                y = np.array(targets)
                
                # Standardize features
                scaler = StandardScaler()
                X_scaled = scaler.fit_transform(X)
                
                # Initialize estimator with reduced complexity
                estimator = GradientBoostingRegressor(
                    n_estimators=50,
                    max_depth=3,
                    learning_rate=0.1,
                    random_state=42
                )
                
                # Initialize RFE to select top 5 features
                selector = RFE(
                    estimator=estimator,
                    n_features_to_select=5,
                    step=1
                )
                
                # Fit selector
                selector = selector.fit(X_scaled, y)
                
                # Update stability scores
                for name, selected in zip(feature_names, selector.support_):
                    if selected:
                        stability_scores[name] += 1
            
            # Normalize stability scores
            for name in stability_scores:
                stability_scores[name] /= n_iterations
            
            # Select features with stability score > 0.5
            selected_features = {
                name: score for name, score in stability_scores.items() 
                if score > 0.5
            }
            
            # Sort by stability score
            selected_features = dict(
                sorted(selected_features.items(), 
                key=lambda x: x[1], 
                reverse=True)
            )
            
            logger.info(f"Selected features with stability scores: {selected_features}")
            
            return selected_features
            
        except Exception as e:
            logger.error(f"Error in stable feature selection: {e}")
            return None
    
    def _optimize_weights_simple(self, historical_data, selected_features):
        """Optimize weights using simplified Bayesian optimization on selected features."""
        try:
            import optuna
            from sklearn.model_selection import train_test_split
            from sklearn.metrics import mean_squared_error
            import numpy as np
            
            if not selected_features or len(historical_data) < self.min_optimization_samples:
                logger.warning("Insufficient data or no selected features for optimization")
                return self.weights
            
            # Prepare feature matrix using only selected features
            X = []
            y = []
            feature_list = list(selected_features.keys())
            
            for entry in historical_data:
                features = []
                for feature in feature_list:
                    if feature == 'speed':
                        value = entry.get('speed', {}).get('value', 0)
                    elif feature == 'position':
                        value = entry.get('position', {}).get('distance', 0)
                    elif feature == 'battery':
                        value = entry.get('battery_status', {}).get('capacity', 0)
                    elif feature == 'lidar':
                        value = entry.get('lidar_status', {}).get('error_rate', 0)
                    elif feature == 'radar':
                        value = entry.get('radar_status', {}).get('detection_density', 0)
                    elif feature == 'angle_deviation':
                        value = entry.get('position', {}).get('angle_diff', 0)
                    elif feature == 'temperature':
                        value = entry.get('battery_status', {}).get('temperature', 0)
                    elif feature == 'point_cloud_integrity':
                        value = entry.get('lidar_status', {}).get('point_cloud_integrity', 0)
                    else:
                        value = 0
                    features.append(value)
                
                X.append(features)
                y.append(entry.get('risk_score', 50.0))
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            def objective(trial):
                # Suggest weights that sum to 1
                weights = []
                remaining = 1.0
                
                for i in range(len(feature_list) - 1):
                    if i == len(feature_list) - 2:
                        # For the second-to-last weight, ensure we leave enough for the last one
                        max_weight = remaining - 0.1  # Leave at least 0.1 for the last weight
                        weight = trial.suggest_float(f"w{i}", 0.1, max_weight)
                    else:
                        weight = trial.suggest_float(f"w{i}", 0.1, remaining - 0.1)
                    weights.append(weight)
                    remaining -= weight
                
                # Last weight is determined by what's left
                weights.append(remaining)
                
                # Calculate weighted predictions
                y_pred = np.dot(X_train, weights)
                
                # Calculate MSE with regularization
                mse = mean_squared_error(y_train, y_pred)
                
                # Add regularization for weight smoothness
                weight_diff_penalty = np.sum(np.diff(weights) ** 2)
                
                # Add regularization for deviation from current weights
                current_weights = [self.weights.get(feature, 0.1) for feature in feature_list]
                weight_change_penalty = np.sum((np.array(weights) - np.array(current_weights)) ** 2)
                
                return mse + 0.1 * weight_diff_penalty + 0.05 * weight_change_penalty
            
            # Create and optimize study with reduced number of trials
            study = optuna.create_study(direction="minimize")
            study.optimize(objective, n_trials=50)
            
            # Get best weights
            best_weights = {}
            remaining = 1.0
            for i in range(len(feature_list) - 1):
                weight = study.best_params[f"w{i}"]
                best_weights[feature_list[i]] = weight
                remaining -= weight
            best_weights[feature_list[-1]] = remaining
            
            # Calculate performance score
            performance_score = 1.0 / (study.best_value + 1e-6)
            
            # Create final weights dictionary
            new_weights = {
                **best_weights,
                'performance_score': performance_score,
                'last_updated': time.time()
            }
            
            # Validate and update weights if significantly better
            if (performance_score > self.weights.get('performance_score', 0) * 1.1 and
                all(0.1 <= w <= 0.5 for w in best_weights.values())):
                self.weights = new_weights
                logger.info(f"Updated weights via simplified Bayesian optimization: {self.weights}")
            
            return new_weights
            
        except Exception as e:
            logger.error(f"Error in simplified Bayesian optimization: {e}")
            return self.weights

    def _calculate_composite_risk_score(self, risk_scores):
        """Calculate composite risk score with enhanced feature selection and Bayesian optimization."""
        try:
            if not isinstance(risk_scores, dict):
                logger.error("Invalid risk scores format")
                return 25.0

            # Get historical data
            historical_data = list(self.optimization_history)
            
            # Periodically update feature selection and weights
            current_time = time.time()
            if (not hasattr(self, 'selected_features') or 
                current_time - self.weights.get('last_updated', 0) > 3600):  # Update every hour
                
                # Select stable features
                self.selected_features = self._select_stable_features(historical_data)
                if self.selected_features:
                    # Optimize weights for selected features
                    self.weights = self._optimize_weights_simple(historical_data, self.selected_features)
            
            # Calculate risk using optimized weights and selected features
            if hasattr(self, 'selected_features') and self.selected_features:
                total_risk = 0.0
                total_weight = 0.0
                
                # Use only selected features with their stability scores as additional weights
                for factor, stability_score in self.selected_features.items():
                    if factor in risk_scores and factor in self.weights:
                        try:
                            score = float(risk_scores[factor])
                            weight = float(self.weights[factor])
                            # Use stability score to adjust weight
                            adjusted_weight = weight * stability_score
                            confidence = self._calculate_confidence(factor, score)
                            
                            total_risk += score * adjusted_weight * confidence
                            total_weight += adjusted_weight * confidence
                        except (ValueError, TypeError):
                            continue
                
                if total_weight > 0:
                    final_score = total_risk / total_weight
                else:
                    # Fallback to simple averaging if no valid weights
                    final_score = sum(risk_scores.values()) / len(risk_scores)
            else:
                # Fallback to simple averaging if no selected features
                valid_scores = []
                for score in risk_scores.values():
                    try:
                        valid_scores.append(float(score))
                    except (ValueError, TypeError):
                        continue
                final_score = sum(valid_scores) / len(valid_scores) if valid_scores else 50.0
            
            # Store result in optimization history
            self.optimization_history.append({
                'speed': risk_scores.get('speed', 0),
                'position': risk_scores.get('position', 0),
                'battery_status': {'capacity': risk_scores.get('battery', 0)},
                'lidar_status': {'error_rate': risk_scores.get('lidar', 0)},
                'radar_status': {'detection_density': risk_scores.get('radar', 0)},
                'risk_score': final_score
            })
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Error in composite risk calculation: {str(e)}")
            return 25.0

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
        """Calculate speed risk score with improved accuracy."""
        try:
            motion = telemetry_data.get('motion', {})
            if not isinstance(motion, dict):
                return 15.0  # Lower base risk for invalid motion data

            # Get speed with validation
            speed = motion.get('speed')
            if speed is None:
                return 15.0
            
            current_speed = abs(float(speed))
            
            # Calculate risk based on thresholds with sigmoid scaling
            if current_speed >= self.speed_thresholds['critical']:
                # Gradual increase beyond critical threshold
                excess = (current_speed - self.speed_thresholds['critical']) / self.speed_thresholds['critical']
                return min(65.0, 50.0 + (15.0 * (1 - math.exp(-excess))))
            elif current_speed >= self.speed_thresholds['warning']:
                # Smooth transition between warning and critical
                progress = (current_speed - self.speed_thresholds['warning']) / (self.speed_thresholds['critical'] - self.speed_thresholds['warning'])
                return 25.0 + (25.0 * progress)
            else:
                # Quadratic risk up to warning speed with lower base
                normalized_speed = current_speed / self.speed_thresholds['warning']
                return 10.0 + (15.0 * normalized_speed * normalized_speed)

        except Exception as e:
            logger.error(f"Error in speed risk calculation: {str(e)}")
            return 15.0  # Default moderate-low risk on error

    def _calculate_battery_risk(self, telemetry_data):
        """Calculate battery risk score with enhanced precision and reliability."""
        try:
            power = telemetry_data.get('power', {})
            if not isinstance(power, dict):
                return 35.0
            battery_level = float(power.get('battery_level', 0))
            battery_temp = float(power.get('battery_temp', 25))
            voltage = float(power.get('voltage', 24))
            charging_status = power.get('charging_status', 'discharging')

            # Enhanced battery level risk with exponential scaling
            level_risk = 0.0
            if battery_level <= 10:  # Critical
                level_risk = 90.0 + (10.0 - battery_level)
            elif battery_level <= 20:  # Severe
                level_risk = 70.0 + (20.0 - battery_level) * 2.0
            elif battery_level <= 30:  # Warning
                level_risk = 50.0 + (30.0 - battery_level)
            else:  # Normal
                level_risk = max(0.0, 40.0 * math.exp(-0.05 * (battery_level - 30)))

            # Enhanced temperature risk with dynamic thresholds
            temp_risk = 0.0
            optimal_temp_range = (20.0, 30.0)
            if battery_temp >= 45.0:  # Critical high
                temp_risk = 100.0
            elif battery_temp >= 40.0:  # Warning high
                temp_risk = 70.0 + (battery_temp - 40.0) * 6.0
            elif battery_temp <= 0.0:  # Critical low
                temp_risk = 100.0
            elif battery_temp <= 5.0:  # Warning low
                temp_risk = 70.0 + (5.0 - battery_temp) * 6.0
            else:  # Normal range
                temp_deviation = min(
                    abs(battery_temp - optimal_temp_range[0]),
                    abs(battery_temp - optimal_temp_range[1])
                )
                temp_risk = temp_deviation * 3.0

            # Enhanced voltage risk with dynamic nominal range
            voltage_risk = 0.0
            nominal_voltage = 24.0
            if voltage <= 20.0:  # Critical low
                voltage_risk = 100.0
            elif voltage <= 22.0:  # Warning low
                voltage_risk = 70.0 + (22.0 - voltage) * 15.0
            elif voltage >= 28.0:  # Critical high
                voltage_risk = 100.0
            elif voltage >= 26.0:  # Warning high
                voltage_risk = 70.0 + (voltage - 26.0) * 15.0
            else:  # Normal range
                voltage_deviation = abs(voltage - nominal_voltage)
                voltage_risk = voltage_deviation * 12.0

            # Dynamic weighting based on charging status and risk levels
            weights = {
                'level': 0.4,
                'temp': 0.35,
                'voltage': 0.25
            }

            if charging_status == 'charging':
                # Adjust weights when charging
                weights['temp'] = 0.45  # Temperature more critical during charging
                weights['voltage'] = 0.30
                weights['level'] = 0.25

            # Calculate weighted risk with exponential penalties for critical values
            total_risk = (
                level_risk * weights['level'] +
                temp_risk * weights['temp'] +
                voltage_risk * weights['voltage']
            )

            # Apply exponential penalty for multiple high risks
            high_risks = sum(1 for risk in [level_risk, temp_risk, voltage_risk] if risk > 70)
            if high_risks >= 2:
                total_risk *= 1.2  # 20% penalty for multiple high risks

            return min(100.0, max(0.0, total_risk))

        except Exception as e:
            logger.error(f"Error in battery risk calculation: {str(e)}")
            return 35.0  # Default moderate-low risk on error

    def _calculate_lidar_risk(self, telemetry_data):
        """Calculate LIDAR risk with enhanced precision and multi-factor analysis."""
        try:
            lidar = telemetry_data.get('lidar', {})
            if not isinstance(lidar, dict):
                return 35.0

            error_rate = float(lidar.get('error_rate', 0))
            range_dev = float(lidar.get('range_deviation', 0))
            integrity = float(lidar.get('point_cloud_integrity', 100))
            max_range = float(lidar.get('max_detection_range', 50.0))

            # Enhanced error rate risk with exponential scaling
            error_risk = 0.0
            if error_rate >= 0.1:  # Critical
                error_risk = 100.0
            elif error_rate >= 0.05:  # High
                error_risk = 70.0 + (error_rate - 0.05) * 600
            elif error_rate >= 0.02:  # Moderate
                error_risk = 40.0 + (error_rate - 0.02) * 1000
            else:  # Normal
                error_risk = error_rate * 2000

            # Enhanced range deviation risk with dynamic thresholds
            range_risk = 0.0
            if range_dev >= 0.05:  # Critical
                range_risk = 100.0
            elif range_dev >= 0.03:  # High
                range_risk = 70.0 + (range_dev - 0.03) * 1500
            elif range_dev >= 0.01:  # Moderate
                range_risk = 40.0 + (range_dev - 0.01) * 1500
            else:  # Normal
                range_risk = range_dev * 4000

            # Enhanced integrity risk with sigmoid scaling
            integrity_risk = 0.0
            if integrity <= 80:  # Critical
                integrity_risk = 100.0
            elif integrity <= 90:  # Poor
                integrity_risk = 70.0 + (90 - integrity) * 3
            elif integrity <= 95:  # Moderate
                integrity_risk = 40.0 + (95 - integrity) * 6
            else:  # Good
                integrity_risk = max(0.0, (100 - integrity) * 8)

            # Dynamic weights based on operational context
            weights = {
                'error': 0.35,
                'range': 0.35,
                'integrity': 0.30
            }

            # Adjust weights based on risk levels
            if error_risk > 70 or range_risk > 70:
                weights['error'] = 0.40
                weights['range'] = 0.40
                weights['integrity'] = 0.20

            # Calculate composite risk score with enhanced precision
            total_risk = (
                error_risk * weights['error'] +
                range_risk * weights['range'] +
                integrity_risk * weights['integrity']
            )

            # Apply penalties for combined high risks
            high_risks = sum(1 for risk in [error_risk, range_risk, integrity_risk] if risk > 70)
            if high_risks >= 2:
                total_risk *= 1.15  # 15% penalty for multiple high risks

            return min(100.0, max(0.0, total_risk))

        except Exception as e:
            logger.error(f"Error in LIDAR risk calculation: {str(e)}")
            return 35.0  # Default moderate-low risk on error

    def _calculate_position_risk(self, telemetry_data):
        """Calculate position risk with standardized ranges and enhanced precision."""
        try:
            position = telemetry_data.get('position', {})
            if not isinstance(position, dict):
                return 25.0

            # Calculate current position metrics
            px = float(position.get('px', 0))
            py = float(position.get('py', 0))
            distance = math.sqrt(px * px + py * py)

            # Enhanced distance risk calculation
            if distance >= self.position_thresholds['critical']:
                excess = (distance - self.position_thresholds['critical']) / self.position_thresholds['critical']
                distance_risk = 75.0 + (25.0 * (1 - math.exp(-2.0 * excess)))
            elif distance >= self.position_thresholds['warning']:
                progress = (distance - self.position_thresholds['warning']) / (self.position_thresholds['critical'] - self.position_thresholds['warning'])
                sigmoid = 1 / (1 + math.exp(-12 * (progress - 0.5)))
                distance_risk = 45.0 + (30.0 * sigmoid)
            else:
                normalized_distance = distance / self.position_thresholds['warning']
                distance_risk = max(5.0, 40.0 * (normalized_distance * normalized_distance))

            # Calculate orientation risk
            theta = float(position.get('theta', 0))
            target_angle = math.atan2(py, px)
            angle_diff = abs(theta - target_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff

            orientation_risk = 0.0
            if distance > 1.0:
                normalized_angle = angle_diff / math.pi
                orientation_risk = 30.0 * normalized_angle * normalized_angle
                if angle_diff > math.pi/2:
                    orientation_risk += 20.0 * ((angle_diff - math.pi/2) / (math.pi/2))
                    if distance > self.position_thresholds['warning']:
                        orientation_risk *= 1.3

            # Calculate velocity risk
            motion_data = telemetry_data.get('motion', {})
            velocity = abs(float(motion_data.get('speed', 0)))
            velocity_risk = 0.0
            if velocity > 5.0:
                velocity_risk = 75.0 + min(25.0, (velocity - 5.0) * 5.0)
            elif velocity > 2.0:
                velocity_risk = 45.0 + (velocity - 2.0) * 10.0
            else:
                velocity_risk = velocity * 20.0

            # Combine risks with weighted factors
            weights = {
                'distance': 0.5,    # Distance from origin is primary factor
                'orientation': 0.3, # Orientation relative to origin
                'velocity': 0.2     # Movement speed
            }

            final_risk = (
                distance_risk * weights['distance'] +
                orientation_risk * weights['orientation'] +
                velocity_risk * weights['velocity']
            )

            # Apply penalty for multiple high risks
            high_risks = sum(1 for risk in [distance_risk, orientation_risk, velocity_risk] if risk >= 75)
            if high_risks >= 2:
                penalty_factor = (high_risks - 1) * 0.05
                final_risk *= (1 + penalty_factor)

            return min(100.0, max(0.0, final_risk))

        except Exception as e:
            logger.error(f"Error in position risk calculation: {str(e)}")
            return 25.0

    def _calculate_composite_risk_score(self, risk_scores):
        """Calculate composite risk score with enhanced feature selection and Bayesian optimization."""
        try:
            if not isinstance(risk_scores, dict):
                logger.error("Invalid risk scores format")
                return 25.0

            # Get historical data
            historical_data = list(self.optimization_history)
            
            # Periodically update feature selection and weights
            current_time = time.time()
            if (not hasattr(self, 'selected_features') or 
                current_time - self.weights.get('last_updated', 0) > 3600):  # Update every hour
                
                # Select stable features
                self.selected_features = self._select_stable_features(historical_data)
                if self.selected_features:
                    # Optimize weights for selected features
                    self.weights = self._optimize_weights_simple(historical_data, self.selected_features)
            
            # Calculate risk using optimized weights and selected features
            if hasattr(self, 'selected_features') and self.selected_features:
                total_risk = 0.0
                total_weight = 0.0
                
                # Use only selected features with their stability scores as additional weights
                for factor, stability_score in self.selected_features.items():
                    if factor in risk_scores and factor in self.weights:
                        try:
                            score = float(risk_scores[factor])
                            weight = float(self.weights[factor])
                            # Use stability score to adjust weight
                            adjusted_weight = weight * stability_score
                            confidence = self._calculate_confidence(factor, score)
                            
                            total_risk += score * adjusted_weight * confidence
                            total_weight += adjusted_weight * confidence
                        except (ValueError, TypeError):
                            continue
                
                if total_weight > 0:
                    final_score = total_risk / total_weight
                else:
                    # Fallback to simple averaging if no valid weights
                    final_score = sum(risk_scores.values()) / len(risk_scores)
            else:
                # Fallback to simple averaging if no selected features
                valid_scores = []
                for score in risk_scores.values():
                    try:
                        valid_scores.append(float(score))
                    except (ValueError, TypeError):
                        continue
                final_score = sum(valid_scores) / len(valid_scores) if valid_scores else 50.0
            
            # Store result in optimization history
            self.optimization_history.append({
                'speed': risk_scores.get('speed', 0),
                'position': risk_scores.get('position', 0),
                'battery_status': {'capacity': risk_scores.get('battery', 0)},
                'lidar_status': {'error_rate': risk_scores.get('lidar', 0)},
                'radar_status': {'detection_density': risk_scores.get('radar', 0)},
                'risk_score': final_score
            })
            
            return min(100.0, max(0.0, final_score))
            
        except Exception as e:
            logger.error(f"Error in composite risk calculation: {str(e)}")
            return 25.0

    def _calculate_confidence(self, factor, score, is_anomaly=False):
        """Calculate confidence level for a risk factor with enhanced precision and anomaly awareness."""
        try:
            # Initialize data structures if not exists
            if not hasattr(self, 'risk_trends'):
                self.risk_trends = {factor: deque(maxlen=10) for factor in ['battery', 'lidar', 'speed', 'position', 'radar']}
            if not hasattr(self, 'last_update_time'):
                self.last_update_time = {}
            if not hasattr(self, 'confidence_history'):
                self.confidence_history = {factor: deque(maxlen=5) for factor in ['battery', 'lidar', 'speed', 'position', 'radar']}

            # Dynamic base confidence with anomaly awareness
            base_confidence = 0.85  # Base confidence
            if is_anomaly:
                base_confidence *= 0.8  # Reduce base confidence for anomalies
            
            # Factor-specific and context-aware confidence adjustments
            if factor == 'battery':
                if score < 30:  # Critical battery level
                    base_confidence *= 0.9
                elif score < 50:  # Low battery
                    base_confidence *= 0.95
            elif factor == 'lidar':
                if score > 70:  # LIDAR issues
                    base_confidence *= 0.9
                elif is_anomaly:  # Anomalous LIDAR readings
                    base_confidence *= 0.85
            
            confidence = base_confidence

            # Enhanced historical variance analysis with anomaly awareness
            if factor in self.risk_trends and len(self.risk_trends[factor]) > 3:
                values = list(self.risk_trends[factor])
                if values:
                    # Calculate recent variance with weighted samples
                    recent_values = values[-5:] if len(values) >= 5 else values
                    weights = np.linspace(0.5, 1.0, len(recent_values))  # More weight to recent values
                    weighted_mean = np.average(recent_values, weights=weights)
                    weighted_var = np.average((recent_values - weighted_mean) ** 2, weights=weights)
                    
                    # Calculate trend stability with anomaly awareness
                    if len(values) >= 3:
                        diffs = np.diff(values[-3:])
                        trend_stability = 1.0 - min(1.0, np.std(diffs) / (5.0 if is_anomaly else 10.0))
                    else:
                        trend_stability = 0.8
                    
                    # Combine variance and stability impacts with context
                    variance_factor = max(0.6, 1.0 - (weighted_var / (500 if is_anomaly else 1000)))
                    confidence *= variance_factor * trend_stability

            # Enhanced temporal confidence decay with anomaly awareness
            current_time = time.time()
            if factor in self.last_update_time:
                time_since_update = current_time - self.last_update_time[factor]
                if time_since_update > 0:
                    # Multi-stage temporal decay with context
                    if time_since_update <= 5.0 and is_anomaly:  # Shorter grace period for anomalies
                        decay_factor = 1.0
                    elif time_since_update <= 10.0 and not is_anomaly:  # Normal grace period
                        decay_factor = 1.0
                    elif time_since_update <= 30.0:  # Moderate age
                        decay_base = 15.0 if is_anomaly else 20.0
                        decay_factor = math.exp(-(time_since_update - 10.0) / decay_base)
                    else:  # Older data
                        decay_base = 8.0 if is_anomaly else 10.0
                        decay_factor = math.exp(-(time_since_update - 30.0) / decay_base) * 0.8
                    
                    confidence *= max(0.4, decay_factor)
            else:
                # If no timestamp exists, apply context-aware reduction
                confidence *= 0.8 if is_anomaly else 0.85

            # Enhanced confidence adjustment based on score characteristics
            if score >= 90:  # Very high risk
                confidence *= 0.8 if is_anomaly else 0.85
            elif score <= 10:  # Very low risk
                confidence *= 0.8 if is_anomaly else 0.85
            elif 40 <= score <= 60:  # Middle range
                if not is_anomaly:  # Only boost confidence for non-anomalous moderate values
                    confidence *= 1.1
                
            # Adjust confidence based on data consistency with anomaly awareness
            if factor in self.confidence_history and len(self.confidence_history[factor]) > 0:
                prev_confidences = list(self.confidence_history[factor])
                confidence_stability = 1.0 - min(0.3, np.std(prev_confidences) / (1.5 if is_anomaly else 2.0))
                confidence *= confidence_stability

            # Store confidence for historical tracking
            if factor in self.confidence_history:
                self.confidence_history[factor].append(confidence)

            # Ensure confidence stays within valid range with context-aware bounds
            min_confidence = 0.15 if is_anomaly else 0.2
            max_confidence = 0.9 if is_anomaly else 0.95
            confidence = max(min_confidence, min(max_confidence, confidence))

            return confidence

        except Exception as e:
            logger.error(f"Error calculating confidence: {str(e)}")
            # Return dynamic fallback based on factor type and anomaly status
            fallback_confidences = {
                'battery': 0.55 if is_anomaly else 0.6,
                'lidar': 0.45 if is_anomaly else 0.5,
                'speed': 0.5 if is_anomaly else 0.55,
                'position': 0.45 if is_anomaly else 0.5,
                'radar': 0.4 if is_anomaly else 0.45
            }
            return fallback_confidences.get(factor, 0.45 if is_anomaly else 0.5)

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

    def analyze_manual_position(self, px, py, theta):
        """Analyze manual position data and calculate risk."""
        try:
            # Calculate distance from origin
            distance = math.sqrt(px * px + py * py)
            
            # Calculate angle to origin
            target_angle = math.atan2(py, px)
            angle_diff = abs(theta - target_angle)
            if angle_diff > math.pi:
                angle_diff = 2 * math.pi - angle_diff
            
            alerts = []
            
            # Check distance thresholds
            if distance >= self.position_thresholds['critical']:
                alerts.append(f"CRITICAL: Distance from origin ({round(distance, 2)}m) exceeds critical threshold")
            elif distance >= self.position_thresholds['warning']:
                alerts.append(f"WARNING: Distance from origin ({round(distance, 2)}m) exceeds warning threshold")
            
            # Check orientation
            if angle_diff > math.pi/2:
                alerts.append(f"WARNING: Large orientation deviation ({round(math.degrees(angle_diff), 2)} degrees)")
            
            # Calculate position-based risk
            if distance >= self.position_thresholds['critical']:
                risk_score = 75.0 + min(25.0, (distance - self.position_thresholds['critical']) * 5.0)
            elif distance >= self.position_thresholds['warning']:
                progress = (distance - self.position_thresholds['warning']) / (self.position_thresholds['critical'] - self.position_thresholds['warning'])
                risk_score = 45.0 + (30.0 * progress)
            else:
                normalized_distance = distance / self.position_thresholds['warning']
                risk_score = max(5.0, 40.0 * (normalized_distance * normalized_distance))
            
            # Add orientation penalty
            if angle_diff > math.pi/2:
                risk_score += 20.0 * ((angle_diff - math.pi/2) / (math.pi/2))
            
            return {
                'position': {
                    'px': px,
                    'py': py,
                    'theta': theta,
                    'distance': distance,
                    'angle_diff': angle_diff
                },
                'risk_score': min(100.0, risk_score),
                'alerts': alerts
            }
            
        except Exception as e:
            logger.error(f"Error in manual position analysis: {str(e)}")
            return {
                'position': {
                    'px': px,
                    'py': py,
                    'theta': theta,
                    'distance': 0.0,
                    'angle_diff': 0.0
                },
                'risk_score': 50.0,
                'alerts': [f"Error analyzing position: {str(e)}"]
            }

    def analyze_manual_data(self, data):
        """Analyze position-only manual input data"""
        try:
            # Handle single entry case
            single_entry = False
            if isinstance(data, dict):
                data = [data]
                single_entry = True
            
            # Basic input validation
            if not data or not isinstance(data, list):
                logger.warning(f"Invalid input data type: {type(data)}")
                return None

            # Process each data point in sequence
            results = []
            for entry in data:
                try:
                    if not isinstance(entry, dict):
                        logger.warning(f"Skipping invalid entry type: {type(entry)}")
                        continue
                    
                    # Get position data
                    position = None
                    timestamp = entry.get('timestamp', datetime.now().isoformat())
                    
                    # Try to get position from poses array first
                    poses = entry.get('poses', [])
                    if poses and isinstance(poses, list) and len(poses) > 0:
                        latest_pose = poses[-1]  # Get the most recent pose
                        if all(key in latest_pose for key in ['px', 'py', 'theta']):
                            position = {
                                'px': float(latest_pose['px']),
                                'py': float(latest_pose['py']),
                                'theta': float(latest_pose['theta'])
                            }

                    # If no poses array or invalid, try direct position fields
                    if not position and all(key in entry for key in ['px', 'py', 'theta']):
                        position = {
                            'px': float(entry['px']),
                            'py': float(entry['py']),
                            'theta': float(entry['theta'])
                        }
                    
                    if not position:
                        logger.warning("No valid position data found in entry")
                        results.append({
                            'robot_id': entry.get('robot_id', 'default'),
                            'robot_name': entry.get('robot_name', 'Manual Input'),
                            'timestamp': timestamp,
                            'error': 'No valid position data'
                        })
                        continue
                        
                    # Get risk assessment using position-only calculation
                    risk_result = self.analyze_manual_position(
                        position['px'],
                        position['py'],
                        position['theta']
                    )
                    
                    if not risk_result:
                        results.append({
                            'robot_id': entry.get('robot_id', 'default'),
                            'robot_name': entry.get('robot_name', 'Manual Input'),
                            'timestamp': timestamp,
                            'error': 'Failed to calculate risk'
                        })
                        continue

                    # Initialize final risk score with position-based risk
                    final_risk_score = risk_result['risk_score']

                    # Calculate speed and acceleration if multiple poses available
                    speed = 0.0
                    acceleration = 0.0
                    speeds = []
                    deviations = []
                    joint_angles = []

                    if poses and len(poses) >= 2:
                        try:
                            for i in range(1, len(poses)):
                                p1 = poses[i-1]
                                p2 = poses[i]
                                dx = p2['px'] - p1['px']
                                dy = p2['py'] - p1['py']
                                dt = 0.1  # 100ms default
                                
                                if 'timestamp' in p1 and 'timestamp' in p2:
                                    try:
                                        t1 = datetime.fromisoformat(p1['timestamp'])
                                        t2 = datetime.fromisoformat(p2['timestamp'])
                                        dt = (t2 - t1).total_seconds()
                                    except (ValueError, TypeError):
                                        pass
                                
                                if dt > 0:
                                    segment_speed = math.sqrt(dx*dx + dy*dy) / dt
                                    speeds.append(segment_speed)
                                    
                                    # Calculate deviation from expected path
                                    expected_x = p1['px'] + (dx/dt) * dt
                                    expected_y = p1['py'] + (dy/dt) * dt
                                    actual_deviation = math.sqrt((p2['px'] - expected_x)**2 + (p2['py'] - expected_y)**2)
                                    deviations.append(actual_deviation)
                                    
                                    # Calculate joint angles (using theta as approximation)
                                    joint_angles.append(abs(p2.get('theta', 0) - p1.get('theta', 0)))

                            # Calculate feature statistics
                            avg_speed = np.mean(speeds) if speeds else 0
                            max_deviation = np.max(deviations) if deviations else 0
                            std_joint_angles = np.std(joint_angles) if joint_angles else 0
                            
                            # Additional features for anomaly detection
                            speed_variance = np.var(speeds) if speeds else 0
                            path_smoothness = np.mean(deviations) if deviations else 0
                            
                            # Calculate dynamic risk factors
                            speed_risk = min(100.0, avg_speed * 20.0)  # 5 m/s -> 100% risk
                            deviation_risk = min(100.0, max_deviation * 50.0)  # 2m deviation -> 100% risk
                            smoothness_risk = min(100.0, std_joint_angles * 100.0)  # π/2 variance -> 100% risk
                            
                            # Combine with position-based risk (weighted average)
                            final_risk_score = (
                                0.5 * final_risk_score +  # Position-based risk
                                0.3 * speed_risk +        # Speed risk
                                0.15 * deviation_risk +   # Path deviation risk
                                0.05 * smoothness_risk    # Movement smoothness risk
                            )
                            
                            speed = avg_speed
                            acceleration = (speeds[-1] - speeds[0]) / (dt * len(speeds)) if len(speeds) >= 2 else 0
                            
                        except Exception as e:
                            logger.warning(f"Error calculating dynamic risk metrics: {e}")
                            # Keep the position-based risk score

                    # Create complete result structure with updated risk score
                    result = {
                        'robot_id': entry.get('robot_id', 'default'),
                        'robot_name': entry.get('robot_name', 'Manual Input'),
                        'timestamp': timestamp,
                        'position': risk_result['position'],
                        'motion': {
                            'speed': round(speed, 3),
                            'acceleration': round(acceleration, 3),
                            'distance': risk_result['position']['distance']
                        },
                        'risk_score': round(final_risk_score, 2),
                        'riskscore': round(final_risk_score, 2),
                        'risk_level': 'High Risk' if final_risk_score > 80 else 
                                    'Medium Risk' if final_risk_score > 50 else 'Low Risk',
                        'alerts': risk_result['alerts'].copy()
                    }

                    # Add speed and acceleration alerts
                    if speed >= self.speed_thresholds['critical']:
                        result['alerts'].append(f"INFO: High speed detected ({round(speed, 2)} m/s)")
                    elif speed >= self.speed_thresholds['warning']:
                        result['alerts'].append(f"INFO: Elevated speed detected ({round(speed, 2)} m/s)")

                    if abs(acceleration) > 2.0:
                        result['alerts'].append(f"INFO: High acceleration detected ({round(abs(acceleration), 2)} m/s²)")

                    # Only preserve non-sensor fields from input
                    for key, value in entry.items():
                        if (key not in result and 
                            key not in ['px', 'py', 'theta', 'poses', 'battery', 'lidar']):
                            result[key] = value

                    results.append(result)

                except Exception as e:
                    logger.warning(f"Error processing entry: {e}")
                    results.append({
                        'robot_id': entry.get('robot_id', 'default'),
                        'robot_name': entry.get('robot_name', 'Manual Input'),
                        'timestamp': timestamp,
                        'error': str(e)
                    })
                    continue

            if not results:
                logger.warning("No valid data could be processed")
                return None

            # Return single result if input was single entry
            return results[0] if single_entry else results

        except Exception as e:
            logger.error(f"Error processing manual data: {e}")
            return None

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
        if not robot_id or not analyzers or robot_id not in analyzers:
            logger.warning(f"Invalid robot_id or analyzer not found: {robot_id}")
            return None

        analyzer = analyzers[robot_id]
        latest_results = analyzer.get_latest_results()

        if not latest_results:
            logger.warning(f"No latest results available for robot: {robot_id}")
            return {
                'robot_id': robot_id,
                'robot_name': analyzer.robot_name,
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

        return latest_results

    except Exception as e:
        logger.error(f"Error getting latest risk scores: {str(e)}")
        return None


