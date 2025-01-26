import requests
import json
import logging
import time
from datetime import datetime
import websocket
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# API Configuration
BASE_URL = "http://localhost:8000/api"
WS_URL = "ws://localhost:8765"

def test_robot_registration():
    """Test robot registration API"""
    logger.info("Testing robot registration...")
    
    # Test data
    robot_data = {
        "id": "test_robot_1",
        "name": "Test Robot 1",
        "manufacturer": "Test Manufacturer",
        "battery_specs": {
            "capacity": 5000,
            "type": "Li-ion",
            "nominal_voltage": 24.0
        },
        "lidar_specs": {
            "model": "Test LIDAR",
            "max_range": 30.0,
            "points_per_second": 40000
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/robots/new", json=robot_data)
        logger.info(f"Registration Response: {response.status_code} - {response.json()}")
        return response.status_code in [201, 409]  # 409 is ok if robot already exists
    except Exception as e:
        logger.error(f"Error in robot registration: {e}")
        return False

def test_telemetry_push():
    """Test telemetry data push API"""
    logger.info("Testing telemetry push...")
    
    telemetry_data = {
        "robot_id": "test_robot_1",
        "timestamp": datetime.now().isoformat(),
        "battery": {
            "capacity": 85,
            "is_charging": False,
            "voltage": 23.5,
            "temperature": 25
        },
        "lidar": {
            "error_rate": 0.5,
            "range_deviation": 0.02,
            "point_cloud_integrity": 98.5,
            "max_detection_range": 29.5
        },
        "poses": [
            {
                "px": 1.0,
                "py": 2.0,
                "theta": 0.5
            }
        ],
        "motion": {
            "speed": 0.5,
            "acceleration": 0.1,
            "angular_velocity": 0.2
        }
    }
    
    try:
        response = requests.post(f"{BASE_URL}/telemetry/push", json=telemetry_data)
        logger.info(f"Telemetry Push Response: {response.status_code} - {response.json()}")
        return response.status_code == 200
    except Exception as e:
        logger.error(f"Error in telemetry push: {e}")
        return False

def test_websocket_connection():
    """Test WebSocket connection and data reception"""
    logger.info("Testing WebSocket connection...")
    
    received_data = []
    connection_established = threading.Event()
    
    def on_message(ws, message):
        try:
            data = json.loads(message)
            logger.info(f"Received WebSocket message: {data}")
            if 'robot_name' in data:
                logger.info(f"Robot name received: {data['robot_name']}")
            received_data.append(data)
        except Exception as e:
            logger.error(f"Error processing WebSocket message: {e}")
    
    def on_error(ws, error):
        logger.error(f"WebSocket error: {error}")
    
    def on_close(ws, close_status_code, close_msg):
        logger.info("WebSocket connection closed")
    
    def on_open(ws):
        logger.info("WebSocket connection opened")
        # Subscribe to test robot
        ws.send(json.dumps({"subscribe": "test_robot_1"}))
        connection_established.set()
    
    try:
        ws = websocket.WebSocketApp(
            WS_URL,
            on_message=on_message,
            on_error=on_error,
            on_close=on_close,
            on_open=on_open
        )
        
        ws_thread = threading.Thread(target=ws.run_forever)
        ws_thread.daemon = True
        ws_thread.start()
        
        # Wait for connection
        if not connection_established.wait(timeout=5):
            logger.error("WebSocket connection timeout")
            return False
            
        time.sleep(2)  # Wait for subscription confirmation
        
        # Send some test telemetry to trigger messages
        test_telemetry_push()
        
        time.sleep(2)  # Wait for messages
        
        return len(received_data) > 0 and any('robot_name' in msg for msg in received_data)
    except Exception as e:
        logger.error(f"Error in WebSocket test: {e}")
        return False
    finally:
        if 'ws' in locals():
            ws.close()

def test_risk_scores():
    """Test risk score APIs"""
    logger.info("Testing risk score APIs...")
    
    try:
        # Start risk score stream
        response = requests.post(f"{BASE_URL}/risk-scores/start")
        logger.info(f"Risk Score Stream Start Response: {response.status_code} - {response.json()}")
        if response.status_code != 200:
            return False
        
        # Get current risk scores
        response = requests.get(f"{BASE_URL}/risk-scores")
        logger.info(f"Risk Scores Response: {response.status_code} - {response.json()}")
        if response.status_code != 200:
            return False
        
        # Get robot history
        response = requests.get(f"{BASE_URL}/robots/test_robot_1/history")
        logger.info(f"Robot History Response: {response.status_code} - {response.json()}")
        if response.status_code != 200:
            return False
            
        # Check if robot names are present in the responses
        scores_data = response.json()
        if isinstance(scores_data, list) and scores_data:
            if 'robot_name' not in scores_data[0]:
                logger.error("Robot name missing in history data")
                return False
                
        return True
    except Exception as e:
        logger.error(f"Error in risk scores test: {e}")
        return False

def run_tests():
    """Run all API tests"""
    try:
        # Test robot registration
        if not test_robot_registration():
            logger.error("Robot registration test failed")
            return False
        
        # Test telemetry push
        if not test_telemetry_push():
            logger.error("Telemetry push test failed")
            return False
        
        # Test WebSocket connection
        if not test_websocket_connection():
            logger.error("WebSocket connection test failed")
            return False
        
        # Test risk scores
        if not test_risk_scores():
            logger.error("Risk score API test failed")
            return False
        
        logger.info("All tests passed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Error running tests: {e}")
        return False

if __name__ == "__main__":
    run_tests() 