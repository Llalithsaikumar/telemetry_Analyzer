import time
import threading
import sys
import signal
import json
from app import create_app, register_routes, emit_risk_scores
from ml_model import RoboticTelemetryAnalyzer
import logging
import random
from datetime import datetime
from websocket_api import TelemetryWebSocketServer
from robot_registry import RobotRegistry
from exit_handler import ExitHandler

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

def start_api_server(app, socketio):
    """Start the Flask API server"""
    try:
        print("Starting Flask API server...")
        socketio.run(app, host='0.0.0.0', port=8000, debug=False, allow_unsafe_werkzeug=True)
    except Exception as e:
        logger.error("Failed to start API server: %s", str(e))

def main():
    exit_handler = ExitHandler()
    
    try:
        # Initialize Flask app and dependencies
        app, socketio, init_robot_registry, telemetry_buffer, risk_score_buffer, telemetry_file, risk_score_thread, active_clients = create_app()
        robot_registry = init_robot_registry()
        
        if not robot_registry:
            print("No robot registry initialized. Exiting...")
            return
            
        # Initialize analyzers list and register routes
        analyzers = []
        register_routes(app, robot_registry, telemetry_buffer, risk_score_buffer, telemetry_file, risk_score_thread,
                      emit_risk_scores, active_clients, socketio, analyzers)
        
        # Start Flask API server in a thread
        api_thread = threading.Thread(target=start_api_server, args=(app, socketio))
        api_thread.daemon = True
        api_thread.start()
        
        # Give API server time to start
        time.sleep(1)
        
        # Initialize and start WebSocket server
        websocket_server = TelemetryWebSocketServer(host='localhost')
        websocket_server.start_server()
        
        # Get registered robots
        robot_ids = robot_registry.list_robots()
        
        if robot_ids:
            print(f"Found {len(robot_ids)} registered robots")
            
            # Initialize analyzers for each robot
            for robot_id in robot_ids:
                try:
                    # Verify robot exists in registry
                    robot_details = robot_registry.get_robot_details(robot_id)
                    if not robot_details:
                        logger.warning(f"Skipping analyzer for unregistered robot {robot_id}")
                        continue
                        
                    analyzer = RoboticTelemetryAnalyzer(
                        robot_id=robot_id,
                        risk_score_buffer=risk_score_buffer
                    )
                    analyzer.start()
                    analyzers.append(analyzer)
                    logger.info("Started analysis for robot %s (%s)", robot_id, robot_details.get('name', 'Unknown'))
                except Exception as e:
                    logger.error("Error initializing analyzer for robot %s: %s", robot_id, str(e))
            
            # Start telemetry analysis with WebSocket broadcasting
            def telemetry_analysis_with_websocket():
                broadcast_errors = 0
                max_broadcast_errors = 50
                last_broadcast_time = time.time()
                
                try:
                    while not exit_handler.exit_now:
                        current_time = time.time()
                        
                        # Reset error count if some time has passed without errors
                        if current_time - last_broadcast_time > 10:
                            broadcast_errors = 0
                        
                        for analyzer in analyzers:
                            try:
                                # Get latest results for this robot
                                latest_result = analyzer.get_latest_results()
                                if latest_result:
                                    # Broadcast telemetry data
                                    success = websocket_server.broadcast_telemetry(analyzer.robot_id, latest_result)
                                    
                                    if success:
                                        last_broadcast_time = current_time
                                        broadcast_errors = 0
                                        
                                        # Log status (reduced frequency)
                                        if random.random() < 0.1:  # Log approximately 10% of updates
                                            logger.info("Robot %s - Risk: %s, Score: %.2f",
                                                      analyzer.robot_id,
                                                      latest_result.get('risk_level', 'Unknown'),
                                                      latest_result.get('risk_score', 0.0))
                                    else:
                                        broadcast_errors += 1
                                        if broadcast_errors > max_broadcast_errors:
                                            logger.error("Too many broadcast errors, checking WebSocket server status")
                                            if not websocket_server.get_status()['running']:
                                                logger.info("Attempting to restart WebSocket server")
                                                websocket_server.start_server()
                                            broadcast_errors = 0
                            
                            except Exception as e:
                                logger.error("Error processing robot %s: %s", analyzer.robot_id, str(e))
                                continue
                        
                        # Small delay before next cycle
                        time.sleep(0.1)  # Reduced delay to 100ms for more responsive updates
                
                except Exception as e:
                    logger.error("Error in telemetry analysis loop: %s", str(e))
                finally:
                    for analyzer in analyzers:
                        analyzer.stop()
                    logger.info("Telemetry analysis stopped.")
            
            # Start telemetry analysis in a thread
            telemetry_thread = threading.Thread(target=telemetry_analysis_with_websocket)
            telemetry_thread.daemon = True
            telemetry_thread.start()
            
            print("Both API and WebSocket servers are running.")
            print("API server: http://localhost:8000")
            print("WebSocket server: ws://localhost:8765")
            
            # Keep the main thread running
            while not exit_handler.exit_now:
                time.sleep(0.5)
                
        else:
            print("No robots registered in the database. Exiting...")
            
    except KeyboardInterrupt:
        print("\nReceived exit signal. Shutting down gracefully...")
        if 'websocket_server' in locals():
            websocket_server.stop()
        for analyzer in analyzers:
            analyzer.stop()
    except Exception as e:
        print(f"Unexpected error: {e}")
        logger.exception("Unexpected error in main")
    finally:
        if 'websocket_server' in locals():
            websocket_server.stop()
        print("Cleanup complete. Exiting...")
        sys.exit(0)

if __name__ == "__main__":
    main()