from flask import Flask, jsonify, request, make_response, Response, stream_with_context
from flask_cors import CORS
import logging
import json
import os
import threading
import queue
from flask_socketio import SocketIO, emit
from robot_registry import RobotRegistry
from ml_model import ThreadSafeBuffer, RoboticTelemetryAnalyzer, get_latest_risk_scores
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any
from dataclasses import dataclass, field
import uuid
import random

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class RiskScore:
    robot_id: str
    score: float
    timestamp: str
    status: str
    alerts: List[str] = field(default_factory=list)


class RiskScoreBuffer:
    def __init__(self, max_size: int = 1000):
        self.buffer: Dict[str, List[Dict]] = {}
        self.max_size = max_size
        self.lock = threading.Lock()
        self.robot_registry = RobotRegistry()  # Initialize robot registry
        
    def load_from_json(self, json_file: str):
        try:
            if os.path.exists(json_file):
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    with self.lock:
                        # Update buffer with new data while maintaining max size
                        for robot_id, scores in data.items():
                            if robot_id not in self.buffer:
                                self.buffer[robot_id] = []
                            
                            # Get robot details for this robot_id
                            robot_name = "Unknown"
                            try:
                                robot_details = self.robot_registry.get_robot_details(robot_id)
                                if robot_details:
                                    robot_name = robot_details.get('name', 'Unknown')
                            except Exception as e:
                                logger.error(f"Error getting robot name: {e}")
                            
                            # Update each score with robot name
                            for score in scores:
                                score['robot_name'] = robot_name
                            
                            self.buffer[robot_id].extend(scores)
                            if len(self.buffer[robot_id]) > self.max_size:
                                self.buffer[robot_id] = self.buffer[robot_id][-self.max_size:]
                    
                    logger.info(f"Loaded risk scores for {len(data)} robots from {json_file}")
                    # Save back to ensure robot names are persisted
                    self.save_to_json(json_file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.warning(f"Error loading risk scores from JSON: {e}. Starting with an empty buffer.")
        except Exception as e:
            logger.error(f"Unexpected error loading risk scores: {e}")
            
    def save_to_json(self, json_file: str):
        """Save the current buffer state to JSON file"""
        try:
            with self.lock:
                # Create a copy of the buffer for saving
                data_to_save = {}
                for robot_id, scores in self.buffer.items():
                    # Get robot details for this robot_id
                    robot_name = "Unknown"
                    try:
                        robot_details = self.robot_registry.get_robot_details(robot_id)
                        if robot_details:
                            robot_name = robot_details.get('name', 'Unknown')
                    except Exception as e:
                        logger.error(f"Error getting robot name: {e}")
                    
                    # Update each score with robot name
                    updated_scores = []
                    for score in scores:
                        score_copy = score.copy()  # Create a copy to avoid modifying original
                        score_copy['robot_name'] = robot_name
                        updated_scores.append(score_copy)
                    
                    data_to_save[robot_id] = updated_scores
                
                # Write to file with proper formatting
                directory = os.path.dirname(os.path.abspath(json_file))
                os.makedirs(directory, exist_ok=True)
                
                with open(json_file, 'w', encoding='utf-8') as f:
                    json.dump(data_to_save, f, indent=2, ensure_ascii=False)
                logger.debug(f"Saved risk scores to {json_file}")
        except Exception as e:
            logger.error(f"Error saving risk scores to JSON: {e}")
            
    def add_score(self, robot_id: str, score: Dict):
        logger.info(f"Adding score for robot {robot_id}")
        with self.lock:
            if robot_id not in self.buffer:
                self.buffer[robot_id] = []
            
            # Get robot name from registry
            try:
                robot_details = self.robot_registry.get_robot_details(robot_id)
                if robot_details:
                    robot_name = robot_details.get('name', 'Unknown')
                else:
                    robot_name = "Unknown"
            except Exception as e:
                logger.error(f"Error getting robot name: {e}")
                robot_name = "Unknown"
            
            # Create a copy of the score and ensure it has the robot name
            score_copy = score.copy()
            score_copy['robot_name'] = robot_name
            
            # Add to buffer
            self.buffer[robot_id].append(score_copy)
            
            if len(self.buffer[robot_id]) > self.max_size:
                self.buffer[robot_id].pop(0)
            
            # Save to file immediately after adding new score
            self.save_to_json('risk_scores.json')
                
    def get_latest_scores(self) -> Dict[str, Any]:
        with self.lock:
            latest_scores = {}
            for robot_id, scores in self.buffer.items():
                if scores:  # Only include robots with scores
                    latest_score = scores[-1].copy()  # Create a copy of the latest score
                    
                    # Ensure robot name is present
                    if 'robot_name' not in latest_score:
                        try:
                            robot_details = self.robot_registry.get_robot_details(robot_id)
                            if robot_details:
                                latest_score['robot_name'] = robot_details.get('name', 'Unknown')
                            else:
                                latest_score['robot_name'] = "Unknown"
                        except Exception as e:
                            logger.error(f"Error getting robot name: {e}")
                            latest_score['robot_name'] = "Unknown"
                    
                    latest_scores[robot_id] = latest_score
            return latest_scores
            
    def get_robot_history(self, robot_id: str, minutes: int = 60) -> List[Dict]:
        with self.lock:
            if robot_id not in self.buffer:
                return []
            cutoff_time = datetime.now() - timedelta(minutes=minutes)
            return [
                score for score in self.buffer[robot_id]
                if datetime.fromisoformat(score['timestamp']) > cutoff_time
            ]


def emit_risk_scores(risk_score_buffer, active_clients, socketio):
    while True:
        try:
            # Reload scores from file periodically
            risk_score_buffer.load_from_json('risk_scores.json')
            
            latest_scores = risk_score_buffer.get_latest_scores()
            logger.info(f"Latest scores retrieved: {latest_scores}")
            
            if latest_scores:
                for client_id, client_data in active_clients.items():
                    subscribed_robots = client_data['subscribed_robots']
                    if subscribed_robots:
                        filtered_scores = {
                            robot_id: score
                            for robot_id, score in latest_scores.items()
                            if robot_id in subscribed_robots and score is not None
                        }
                        if filtered_scores:
                            # Add robot names to the emission data
                            emission_data = {
                                'timestamp': datetime.now().isoformat(),
                                'data': filtered_scores
                            }
                            socketio.emit('risk_scores', emission_data, room=client_id)
                            logger.debug(f"Emitted scores to client {client_id}: {emission_data}")
            time.sleep(1)
        except Exception as e:
            logger.error(f"Error in risk score emission: {e}")
            time.sleep(1)


def handle_error(error_message: str, status_code: int = 400) -> make_response:
    logger.error(f"Error: {error_message} (Status: {status_code})")
    return make_response(jsonify({'error': error_message}), status_code)


def create_app(config_file: str = 'config.json'):
    app = Flask(__name__)
    CORS(app)
    telemetry_buffer = ThreadSafeBuffer(maxsize=1000)
    risk_score_buffer = RiskScoreBuffer(max_size=1000)
    risk_score_buffer.load_from_json('risk_scores.json')  # Load existing scores
    telemetry_file = "all_robot_telemetry.json"
    risk_score_thread = None
    active_clients = {}

    try:
        with open(config_file, 'r') as f:
            config = json.load(f)
            logger.info(f"Loaded configuration from {config_file}")
    except FileNotFoundError:
        logger.warning(f"Configuration file '{config_file}' not found. Using default settings.")
        config = {}

    socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

    # WebSocket Event Handlers
    @socketio.on('connect')
    def handle_connect():
        client_id = str(uuid.uuid4())
        active_clients[client_id] = {
            'connected_at': datetime.now().isoformat(),
            'subscribed_robots': set()
        }
        logger.info(f'Client {client_id} connected')
        emit('connection_established', {
            'client_id': client_id,
            'message': 'Connected to risk score stream'
        })

    @socketio.on('disconnect')
    def handle_disconnect():
        client_id = request.sid
        if client_id in active_clients:
            del active_clients[client_id]
            logger.info(f'Client {client_id} disconnected')

    @socketio.on('subscribe_robot')
    def handle_robot_subscription(data):
        client_id = request.sid
        robot_id = data.get('robot_id')

        if not robot_id:
            emit('error', {'message': 'Robot ID is required'})
            return

        if client_id in active_clients:
            active_clients[client_id]['subscribed_robots'].add(robot_id)
            emit('subscription_success', {
                'robot_id': robot_id,
                'message': f'Subscribed to robot {robot_id}'
            })
            logger.info(f'Client {client_id} subscribed to robot {robot_id}')

    def init_robot_registry():
        try:
            return RobotRegistry()
        except Exception as e:
            logger.critical(f"Failed to initialize RobotRegistry: {e}")
            return None

    return app, socketio, init_robot_registry, telemetry_buffer, risk_score_buffer, telemetry_file, risk_score_thread, active_clients


def register_routes(app, robot_registry, telemetry_buffer, risk_score_buffer, telemetry_file, risk_score_thread,
                    emit_risk_scores, active_clients, socketio, analyzers):
    @app.route('/api/robots/list', methods=['GET'])
    def get_all_robots():
        try:
            robots = {robot_id: robot_registry.get_robot_details(robot_id) for robot_id in robot_registry.list_robots()}
            return jsonify(robots), 200
        except Exception as e:
            logger.exception(f"Error in get_all_robots: {e}")
            return handle_error(f"Failed to retrieve robots: {e}", 500)

    @app.route('/api/robots/<robot_id>', methods=['GET'])
    def get_robot_details(robot_id):
        try:
            details = robot_registry.get_robot_details(robot_id)
            if details:
                latest_scores = risk_score_buffer.get_latest_scores()
                if robot_id in latest_scores:
                    details['latest_risk_score'] = latest_scores[robot_id]
                return jsonify(details), 200
            return handle_error('Robot not found', 404)
        except Exception as e:
            logger.exception(f"Error in get_robot_details: {e}")
            return handle_error(f"Failed to retrieve robot details: {e}", 500)

    @app.route('/api/robots/<robot_id>/history', methods=['GET'])
    def get_robot_history(robot_id):
        try:
            minutes = int(request.args.get('minutes', 60))
            history = risk_score_buffer.get_robot_history(robot_id, minutes)
            return jsonify(history), 200
        except ValueError:
            return handle_error("Invalid 'minutes' parameter", 400)
        except Exception as e:
            logger.exception(f"Error in get_robot_history: {e}")
            return handle_error(f"Failed to retrieve robot history: {e}", 500)

    @app.route('/api/robots/new', methods=['POST'])
    def register_robot():
        if not request.is_json:
            return handle_error("Request must contain JSON data", 415)
        try:
            data = request.get_json()
            required_fields = ['id', 'name', 'manufacturer']
            missing_fields = [field for field in required_fields if field not in data]
            if missing_fields:
                return handle_error(f"Missing required fields: {', '.join(missing_fields)}", 400)

            robot_id = data['id']
            if robot_registry.get_robot_details(robot_id):
                return handle_error(f"Robot with ID '{robot_id}' already exists", 409)  # 409 Conflict

            battery_data = data.get('battery_specs')
            lidar_data = data.get('lidar_specs')

            robot_registry.register_robot(
                robot_id=robot_id,
                name=data['name'],
                manufacturer=data['manufacturer'],
                battery_data=battery_data,
                lidar_data=lidar_data
            )
            return jsonify({'robot_id': robot_id, 'message': 'Robot registered successfully'}), 201
        except Exception as e:
            logger.exception(f"Error in register_robot: {e}")
            return handle_error(f"Failed to register robot: {e}", 500)

    @app.route('/api/telemetry/push', methods=['POST'])
    def push_telemetry():
        if not request.is_json:
            return handle_error("Request must contain JSON data", 415)
        try:
            data = request.get_json()
            telemetry_buffer.put(data)
            threading.Thread(target=save_telemetry, args=(telemetry_buffer, telemetry_file), daemon=True).start()
            return jsonify({'status': 'success', 'message': 'Telemetry data received'}), 200
        except queue.Full:
            return handle_error("Telemetry buffer is full", 503)
        except json.JSONDecodeError:
            return handle_error("Invalid JSON data", 400)
        except Exception as e:
            logger.exception(f"Error in push_telemetry: {e}")
            return handle_error(f"Failed to push telemetry data: {e}", 500)

    @app.route('/api/risk-scores/start', methods=['POST'])
    def start_risk_score_stream():
        nonlocal risk_score_thread
        try:
            if risk_score_thread is None or not risk_score_thread.is_alive():
                risk_score_thread = threading.Thread(target=emit_risk_scores,
                                                     args=(risk_score_buffer, active_clients, socketio),
                                                     daemon=True)
                risk_score_thread.start()
                logger.info("Risk score stream started")
                return jsonify({'message': 'Risk score stream started'}), 200
            return jsonify({'message': 'Risk score stream already running'}), 200
        except Exception as e:
            logger.exception(f"Error in start_risk_score_stream: {e}")
            return handle_error(f"Failed to start risk score stream: {e}", 500)

    @app.route('/api/risk-scores/push', methods=['POST'])
    def push_risk_score():
        if not request.is_json:
            return handle_error("Request must contain JSON data", 415)
        try:
            data = request.get_json()
            robot_id = data['robot_id']
            risk_score_buffer.add_score(robot_id, data)
            return jsonify({'status': 'success', 'message': 'Risk score received'}), 200
        except KeyError as e:
            return handle_error(f"Missing required key in JSON: {e}", 400)
        except Exception as e:
            logger.exception(f"Error in push_risk_score: {e}")
            return handle_error(f"Failed to push risk score: {e}", 500)

    @app.route('/api/risk-scores', methods=['GET'])
    def get_current_risk_scores():
        try:
            latest_scores = risk_score_buffer.get_latest_scores()
            return jsonify(latest_scores), 200
        except Exception as e:
            logger.exception(f"Error retrieving risk scores: {e}")
            return handle_error(f"Failed to retrieve risk scores: {e}", 500)

    @app.route('/api/risks/stream', methods=['GET'])
    def stream_robot_risks():
        robot_id = request.args.get('robot_id')

        def generate():
            while True:
                try:
                    risk_data = risk_score_buffer.get_latest_scores()
                    if robot_id:
                        if robot_id in risk_data and risk_data[robot_id]:
                            risk_data = {robot_id: risk_data[robot_id]}
                    if risk_data:
                        yield f"data: {json.dumps(risk_data)}\n\n"
                    time.sleep(1)
                except Exception as e:
                    logger.error(f"Error in risk stream: {e}")
                    break

        return Response(generate(), mimetype="text/event-stream")

    @app.route('/api/risks/stream/<robot_id>', methods=['GET'])
    def stream_risks(robot_id):
        def generate():
            while True:  # Keep streaming data
                try:
                    risk_data = get_latest_risk_scores(robot_id, analyzers)  # Pass analyzers list
                    if risk_data:
                        yield f"data: {json.dumps(risk_data)}\n\n"  # Format as Server-Sent Events
                    else:
                        logger.error(f"No risk data found for robot_id '{robot_id}'")
                        yield f"data: {{'error': 'No risk data found'}}\n\n"
                        break
                except KeyError:
                    logger.error(f"Robot ID '{robot_id}' not found")
                    yield f"data: {{'error': 'Robot ID not found'}}\n\n"
                    break  # Stop if the robot isn't found
                except Exception as e:  # Catch general exceptions for debugging
                    logger.error(f"Error in stream_risks for robot_id '{robot_id}': {e}", exc_info=True)
                    yield f"data: {{'error': 'Server Error'}}\n\n"
                    break

        return Response(stream_with_context(generate()), mimetype='text/event-stream')

    @app.route('/api/risk-scores/status', methods=['GET'])
    def get_risk_score_stream_status():
        if risk_score_thread and risk_score_thread.is_alive():
            return jsonify({'status': 'running'}), 200
        else:
            return jsonify({'status': 'stopped'}), 200

    @app.route('/health', methods=['GET'])
    def health_check():
        return jsonify({
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'active_clients': len(active_clients),
            'registered_robots': len(robot_registry.list_robots()) if robot_registry else 0
        }), 200

    def normalize_pose_data(raw_data):
        """
        Normalize different pose data formats into a standard format.
        Handles various input formats and structures.
        """

        def extract_numeric_value(value):
            """Extract numeric values from various formats"""
            try:
                if isinstance(value, (int, float)):
                    return float(value)
                elif isinstance(value, str):
                    # Remove any non-numeric characters except decimal point and minus
                    return float(''.join(c for c in value if c.isdigit() or c in '.-'))
                return None
            except (ValueError, TypeError):
                return None

        def is_valid_pose(pose):
            """Check if a dictionary can be interpreted as a valid pose"""
            if isinstance(pose, dict):
                # Try to find keys that might represent x, y, theta coordinates
                possible_x = next((v for k, v in pose.items() if any(x in k.lower() for x in ['px', 'x', 'posx'])),
                                  None)
                possible_y = next((v for k, v in pose.items() if any(y in k.lower() for y in ['py', 'y', 'posy'])),
                                  None)
                possible_theta = next(
                    (v for k, v in pose.items() if any(t in k.lower() for t in ['theta', 'angle', 'orientation'])),
                    None)

                return all(extract_numeric_value(v) is not None for v in [possible_x, possible_y, possible_theta])
            return False

        def convert_to_standard_pose(pose_data):
            """Convert various pose formats to standard format"""
            if isinstance(pose_data, dict):
                # Find appropriate keys for coordinates
                px = next((extract_numeric_value(v) for k, v in pose_data.items() if
                           any(x in k.lower() for x in ['px', 'x', 'posx'])), None)
                py = next((extract_numeric_value(v) for k, v in pose_data.items() if
                           any(y in k.lower() for y in ['py', 'y', 'posy'])), None)
                theta = next((extract_numeric_value(v) for k, v in pose_data.items() if
                              any(t in k.lower() for t in ['theta', 'angle', 'orientation'])), None)

                if all(v is not None for v in [px, py, theta]):
                    return {'px': px, 'py': py, 'theta': theta}

            elif isinstance(pose_data, (list, tuple)) and len(pose_data) >= 3:
                # Assume first three numbers are x, y, theta
                values = [extract_numeric_value(v) for v in pose_data[:3]]
                if all(v is not None for v in values):
                    return {'px': values[0], 'py': values[1], 'theta': values[2]}

            return None

        normalized_batches = []

        try:
            # Handle string input (JSON)
            if isinstance(raw_data, str):
                try:
                    raw_data = json.loads(raw_data)
                except json.JSONDecodeError:
                    # Try to parse as space/comma-separated values
                    lines = raw_data.strip().split('\n')
                    raw_data = []
                    for line in lines:
                        values = line.strip().split(',') if ',' in line else line.strip().split()
                        if len(values) >= 3:
                            raw_data.append(values)

            # Handle different data structures
            if isinstance(raw_data, dict):
                # Extract any array-like values from the dictionary
                pose_arrays = [v for v in raw_data.values() if isinstance(v, (list, tuple))]
                if pose_arrays:
                    raw_data = pose_arrays
                else:
                    raw_data = [raw_data]

            if not isinstance(raw_data, (list, tuple)):
                return []

            # Handle flat array vs array of arrays
            if not raw_data:
                return []

            # If it's a flat array of numbers, try to group them into poses
            if all(isinstance(x, (int, float, str)) for x in raw_data):
                poses = []
                for i in range(0, len(raw_data), 3):
                    if i + 2 < len(raw_data):
                        pose = [raw_data[i], raw_data[i + 1], raw_data[i + 2]]
                        normalized_pose = convert_to_standard_pose(pose)
                        if normalized_pose:
                            poses.append(normalized_pose)
                if poses:
                    normalized_batches.append(poses)
            else:
                # Handle nested structures
                current_batch = []
                for item in raw_data:
                    if isinstance(item, (list, tuple)):
                        # Array of arrays
                        batch_poses = []
                        for pose_data in item:
                            normalized_pose = convert_to_standard_pose(pose_data)
                            if normalized_pose:
                                batch_poses.append(normalized_pose)
                        if batch_poses:
                            normalized_batches.append(batch_poses)
                    elif is_valid_pose(item):
                        # Array of pose objects
                        normalized_pose = convert_to_standard_pose(item)
                        if normalized_pose:
                            current_batch.append(normalized_pose)

                if current_batch:
                    normalized_batches.append(current_batch)

        except Exception as e:
            logger.error(f"Error normalizing pose data: {e}")
            return []

        return normalized_batches

    @app.route('/api/manual-input', methods=['POST'])
    def manual_input():
        """
        Enhanced manual input handler that can process almost any format of pose data.
        """
        try:
            # Handle multipart/form-data
            if request.files:
                file = next(iter(request.files.values()))
                content = file.read().decode('utf-8')
                raw_data = content
            # Handle JSON data
            elif request.is_json:
                raw_data = request.get_json()
            # Handle raw data
            else:
                raw_data = request.get_data().decode('utf-8')

            logger.info(f"Received raw data type: {type(raw_data)}")

            # Normalize the data into standard format
            pose_batches = normalize_pose_data(raw_data)

            if not pose_batches:
                return handle_error("No valid pose data could be extracted from input", 400)

            results = []

            for pose_batch in pose_batches:
                # Process poses in groups of 3
                for i in range(0, len(pose_batch), 3):
                    pose_group = pose_batch[i:i + 3]
                    if len(pose_group) < 3:
                        continue

                    # Create telemetry data with simulated sensor data
                    telemetry_data = {
                        'robot_id': "default",
                        'timestamp': datetime.now().isoformat(),
                        'poses': pose_group,
                        'battery': {
                            'capacity': random.uniform(80, 100),
                            'is_charging': random.choice([True, False]),
                            'voltage': random.uniform(12.0, 14.0),
                            'temperature': random.uniform(20, 40)
                        },
                        'lidar': {
                            'error_rate': random.uniform(0, 2),
                            'range_deviation': random.uniform(0, 0.05),
                            'point_cloud_integrity': random.uniform(95, 100),
                            'max_detection_range': random.uniform(25, 35)
                        }
                    }

                    # Process with analyzer
                    analyzer = next((a for a in analyzers if a.robot_id == "default"), None)
                    if not analyzer:
                        analyzer = RoboticTelemetryAnalyzer(
                            robot_id="default",
                            risk_score_buffer=risk_score_buffer
                        )
                        analyzers.append(analyzer)
                        analyzer.start()

                    result = analyzer.analyze_manual_data(telemetry_data)
                    if result:
                        results.append(result)

            if results:
                response_data = {
                    'status': 'success',
                    'message': f'Successfully processed {len(results)} data points',
                    'risk_scores': results
                }
                return jsonify(response_data), 200

            return handle_error("No valid data could be processed", 400)

        except Exception as e:
            logger.exception(f"Error in manual_input: {e}")
            return handle_error(f"Failed to process manual input: {str(e)}", 500)

    @app.route('/api/robots/<robot_id>/battery-status', methods=['GET'])
    def get_robot_battery_status(robot_id):
        """Retrieve detailed battery status for a specific robot."""
        try:
            latest_scores = risk_score_buffer.get_latest_scores()

            if robot_id in latest_scores:
                score_data = latest_scores[robot_id]
                battery_status = score_data.get('battery_status', {})

                return jsonify({
                    'status': battery_status.get('risk_level', 'Unknown'),
                    'capacity': battery_status.get('capacity', 0),
                    'is_charging': battery_status.get('is_charging', False),
                    'voltage': battery_status.get('voltage', 0),
                    'temperature': battery_status.get('temperature', 0),
                    'alerts': battery_status.get('alerts', [])
                }), 200

            return handle_error(f'No battery data found for robot {robot_id}', 404)

        except Exception as e:
            return handle_error(f"Failed to retrieve battery status: {e}", 500)

    @app.route('/api/robots/<robot_id>/lidar-status', methods=['GET'])
    def get_robot_lidar_status(robot_id):
        """Retrieve detailed LIDAR status for a specific robot."""
        try:
            latest_scores = risk_score_buffer.get_latest_scores()

            if robot_id in latest_scores:
                score_data = latest_scores[robot_id]
                lidar_status = score_data.get('lidar_status', {})

                return jsonify({
                    'status': lidar_status.get('risk_level', 'Unknown'),
                    'error_rate': lidar_status.get('error_rate', 0),
                    'range_deviation': lidar_status.get('range_deviation', 0),
                    'point_cloud_integrity': lidar_status.get('point_cloud_integrity', 100),
                    'max_detection_range': lidar_status.get('max_detection_range', 0),
                    'alerts': lidar_status.get('alerts', [])
                }), 200

            return handle_error(f'No LIDAR data found for robot {robot_id}', 404)

        except Exception as e:
            return handle_error(f"Failed to retrieve LIDAR status: {e}", 500)

    @app.route('/api/robots/<robot_id>/health', methods=['GET'])
    def get_robot_comprehensive_health(robot_id):
        """Retrieve comprehensive health status including battery, LIDAR, and overall risk."""
        try:
            latest_scores = risk_score_buffer.get_latest_scores()

            if robot_id in latest_scores:
                score_data = latest_scores[robot_id]
                battery_status = score_data.get('battery_status', {})
                lidar_status = score_data.get('lidar_status', {})

                comprehensive_health = {
                    'robot_id': robot_id,
                    'name': robot_registry.get_robot_details(robot_id).get('name', 'Unknown'),  # Ensure name is fetched
                    'overall_risk_level': score_data.get('risk_level', 'Unknown'),
                    'battery': {
                        'status': battery_status.get('risk_level', 'Unknown'),
                        'capacity': battery_status.get('capacity', 0),
                        'is_charging': battery_status.get('is_charging', False),
                        'voltage': battery_status.get('voltage', 0),
                        'temperature': battery_status.get('temperature', 0),
                        'alerts': battery_status.get('alerts', [])
                    },
                    'lidar': {
                        'status': lidar_status.get('risk_level', 'Unknown'),
                        'error_rate': lidar_status.get('error_rate', 0),
                        'range_deviation': lidar_status.get('range_deviation', 0),
                        'point_cloud_integrity': lidar_status.get('point_cloud_integrity', 100),
                        'max_detection_range': lidar_status.get('max_detection_range', 0),
                        'alerts': lidar_status.get('alerts', [])
                    },
                    'timestamp': score_data.get('timestamp', datetime.now().isoformat())
                }

                return jsonify(comprehensive_health), 200

            return handle_error(f'No health data found for robot {robot_id}', 404)

        except Exception as e:
            return handle_error(f"Failed to retrieve comprehensive health status: {e}", 500)

    @app.route('/api/alerts', methods=['GET'])
    def get_all_robot_alerts():
        """Retrieve alerts for all registered robots."""
        try:
            latest_scores = risk_score_buffer.get_latest_scores()
            all_alerts = {}

            for robot_id, score_data in latest_scores.items():
                robot_details = robot_registry.get_robot_details(robot_id)
                robot_alerts = {
                    'robot_name': robot_details.get('name', 'Unknown'),
                    'battery_alerts': score_data.get('battery_status', {}).get('alerts', []),
                    'lidar_alerts': score_data.get('lidar_status', {}).get('alerts', [])
                }

                # Only include robots with active alerts
                if robot_alerts['battery_alerts'] or robot_alerts['lidar_alerts']:
                    all_alerts[robot_id] = robot_alerts

            return jsonify(all_alerts), 200

        except Exception as e:
            return handle_error(f"Failed to retrieve system-wide alerts: {e}", 500)

    def save_telemetry(telemetry_buffer: ThreadSafeBuffer, telemetry_file: str):
        try:
            try:
                with open(telemetry_file, 'r') as f:
                    existing_telemetry = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_telemetry = []

            new_telemetry_data = telemetry_buffer.get_all()
            combined_telemetry = existing_telemetry + new_telemetry_data
            combined_telemetry = combined_telemetry[-1000:]

            with open(telemetry_file, 'w') as f:
                json.dump(combined_telemetry, f, indent=4)

            logger.info(f"Telemetry data saved to {telemetry_file}")
        except Exception as e:
            logger.error(f"Error saving telemetry data: {e}")


if __name__ == '__main__':
    app, socketio, init_robot_registry, telemetry_buffer, risk_score_buffer, telemetry_file, risk_score_thread, active_clients = create_app()
    robot_registry = init_robot_registry()
    if robot_registry:
        analyzers = []  # Initialize the analyzers list
        register_routes(app, robot_registry, telemetry_buffer, risk_score_buffer, telemetry_file, risk_score_thread,
                        emit_risk_scores, active_clients, socketio, analyzers)
        socketio.run(app, debug=True, host='0.0.0.0', port=8000)