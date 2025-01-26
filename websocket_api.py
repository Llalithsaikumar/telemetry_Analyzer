import json
import logging
import os
from datetime import datetime
from collections import defaultdict
from simple_websocket_server import WebSocketServer, WebSocket
from dotenv import load_dotenv
import time
import threading
import queue
import uuid

# Load environment variables
load_dotenv()

# WebSocket Configuration
WS_HOST = os.getenv('WS_HOST', 'localhost')
WS_PORT = int(os.getenv('WS_PORT', 8765))

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s')
logger = logging.getLogger(__name__)

class TelemetryWebSocket(WebSocket):
    clients = defaultdict(set)
    _lock = threading.Lock()

    def handle(self):
        """Handle incoming messages"""
        try:
            data = json.loads(self.data)
            if 'subscribe' in data:
                robot_id = data['subscribe']
                # Remove from any existing subscriptions first
                self._cleanup_existing_subscriptions()
                
                # Add to new subscription with proper locking
                with self._lock:
                    self.clients[robot_id].add(self)
                    client_count = len(self.clients[robot_id])
                
                # Get robot name for confirmation message
                robot_name = "Unknown"
                try:
                    from robot_registry import RobotRegistry
                    registry = RobotRegistry()
                    robot_details = registry.get_robot_details(robot_id)
                    if robot_details and robot_details.get('name'):
                        robot_name = robot_details['name']
                except Exception as e:
                    logger.error(f"Error getting robot name: {e}")
                
                logger.info(f"New client connected for robot {robot_id} ({robot_name}). Total clients: {client_count}")
                
                # Send immediate confirmation with robot details
                self.send_message(json.dumps({
                    'status': 'subscribed',
                    'robot_id': robot_id,
                    'robot_name': robot_name,
                    'client_id': id(self),
                    'timestamp': datetime.now().isoformat(),
                    'message': f'Successfully connected to WebSocket server for robot {robot_name}',
                    'connection_info': {
                        'total_clients': client_count,
                        'subscription_time': datetime.now().isoformat()
                    }
                }))
        except json.JSONDecodeError:
            logger.error("Invalid JSON message received")
            self.send_message(json.dumps({
                'error': 'Invalid JSON format',
                'timestamp': datetime.now().isoformat()
            }))
        except Exception as e:
            logger.error("Error handling message: %s", str(e))
            self.send_message(json.dumps({
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }))

    def _cleanup_existing_subscriptions(self):
        """Remove client from all existing subscriptions with proper locking"""
        with self._lock:
            for robot_id in list(self.clients.keys()):
                if self in self.clients[robot_id]:
                    self.clients[robot_id].remove(self)
                    if not self.clients[robot_id]:
                        del self.clients[robot_id]

    def connected(self):
        """Handle new connection"""
        logger.info("Client connected from %s", str(self.address))
        try:
            self.send_message(json.dumps({
                'status': 'connected',
                'timestamp': datetime.now().isoformat(),
                'message': 'Connected to WebSocket server'
            }))
        except Exception as e:
            logger.error("Error sending connection acknowledgment: %s", str(e))

    def handle_close(self):
        """Handle client disconnection"""
        try:
            self._cleanup_existing_subscriptions()
            logger.info("Client disconnected from %s", str(self.address))
        except Exception as e:
            logger.error("Error handling client disconnection: %s", str(e))

class TelemetryWebSocketServer:
    def __init__(self, host=None, port=None):
        self.host = host or WS_HOST
        self.port = port or WS_PORT
        self.server = None
        self.running = False
        self.connection_error = None
        self._retry_count = 0
        self._max_retries = 5
        self._retry_delay = 1
        self._server_thread = None
        self._message_queue = queue.Queue(maxsize=2000)
        self._broadcast_thread = None
        self._lock = threading.Lock()
        self._broadcast_errors = 0
        self._max_broadcast_errors = 100
        self._last_broadcast_time = time.time()
        self._shutdown_event = threading.Event()
        logger.info("Initializing WebSocket server on %s:%d", self.host, self.port)

    def start_server(self):
        """Start the WebSocket server"""
        try:
            if not self.running:
                self._shutdown_event.clear()
                self.server = WebSocketServer(self.host, self.port, TelemetryWebSocket)
                self._server_thread = threading.Thread(target=self._run_server)
                self._server_thread.daemon = True
                self._server_thread.start()
                self.running = True
                self._broadcast_thread = threading.Thread(target=self._broadcast_worker)
                self._broadcast_thread.daemon = True
                self._broadcast_thread.start()
                logger.info("WebSocket server started successfully")
                return True
        except Exception as e:
            self.connection_error = str(e)
            logger.error("Failed to start WebSocket server: %s", str(e))
            return False

    def _run_server(self):
        """Run server with proper shutdown handling"""
        try:
            while not self._shutdown_event.is_set():
                try:
                    self.server.handle_request()
                except Exception as e:
                    if not self._shutdown_event.is_set():
                        logger.error(f"Error handling request: {e}")
                    break
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"Server thread error: {e}")

    def stop(self):
        """Stop the WebSocket server gracefully"""
        try:
            # Signal threads to stop
            self._shutdown_event.set()
            self.running = False

            # Close all client connections
            if hasattr(TelemetryWebSocket, 'clients'):
                for robot_id in list(TelemetryWebSocket.clients.keys()):
                    for client in list(TelemetryWebSocket.clients[robot_id]):
                        try:
                            client.close_connection()
                        except Exception as e:
                            logger.error(f"Error closing client connection: {e}")
                TelemetryWebSocket.clients.clear()

            # Close server and cleanup
            if self.server:
                try:
                    # Close all listeners
                    if hasattr(self.server, 'listeners'):
                        for sock in list(self.server.listeners):
                            try:
                                sock.close()
                            except Exception:
                                pass
                        self.server.listeners.clear()
                except Exception as e:
                    logger.error(f"Error closing server listeners: {e}")

            # Wait for threads to finish
            if self._server_thread and self._server_thread.is_alive():
                self._server_thread.join(timeout=1)
            if self._broadcast_thread and self._broadcast_thread.is_alive():
                self._broadcast_thread.join(timeout=1)

            # Clear message queue
            while not self._message_queue.empty():
                try:
                    self._message_queue.get_nowait()
                except queue.Empty:
                    break

            self.server = None
            logger.info("WebSocket server stopped")
        except Exception as e:
            logger.error(f"Error stopping WebSocket server: {e}")
        finally:
            self.running = False
            self.server = None

    def broadcast_telemetry(self, robot_id, telemetry_data):
        """Queue telemetry data for broadcasting with enhanced error handling and monitoring"""
        if not self.running:
            logger.warning("Server not running, cannot broadcast telemetry")
            return False

        try:
            # Get robot name from registry
            robot_name = "Unknown"
            try:
                from robot_registry import RobotRegistry
                registry = RobotRegistry()
                robot_details = registry.get_robot_details(robot_id)
                if robot_details and robot_details.get('name'):
                    robot_name = robot_details['name']
            except Exception as e:
                logger.error(f"Error getting robot name: {e}")

            # Reset error count if last broadcast was successful
            if time.time() - self._last_broadcast_time > 5:
                self._broadcast_errors = 0

            # Add metadata to help track message delivery
            message = {
                'robot_id': robot_id,
                'robot_name': robot_name,
                'timestamp': datetime.now().isoformat(),
                'data': telemetry_data,
                'message_id': str(uuid.uuid4()),
                'sequence': int(time.time() * 1000)
            }
            
            # Try to add to queue with monitoring
            try:
                if self._message_queue.qsize() >= self._message_queue.maxsize * 0.9:
                    logger.warning(f"Message queue is at {self._message_queue.qsize()}/{self._message_queue.maxsize}")
                    # Remove oldest messages if queue is near capacity
                    while self._message_queue.qsize() > self._message_queue.maxsize * 0.7:
                        try:
                            old_msg = self._message_queue.get_nowait()
                            logger.debug(f"Dropped old message for robot {old_msg.get('robot_id')} ({old_msg.get('robot_name', 'Unknown')}) from {old_msg.get('timestamp')}")
                        except queue.Empty:
                            break

                self._message_queue.put_nowait(message)
                self._last_broadcast_time = time.time()
                logger.debug(f"Queued message {message['message_id']} for robot {robot_id} ({robot_name})")
                return True

            except queue.Full:
                self._broadcast_errors += 1
                logger.error(f"Message queue is full. Size: {self._message_queue.qsize()}")
                if self._broadcast_errors > self._max_broadcast_errors:
                    logger.warning("Too many broadcast errors, attempting server restart")
                    self._try_restart_server()
                return False
                    
        except Exception as e:
            self._broadcast_errors += 1
            logger.error(f"Error queueing telemetry data: {str(e)}")
            if self._broadcast_errors > self._max_broadcast_errors:
                self._try_restart_server()
            return False

    def _broadcast_worker(self):
        """Process broadcast queue with enhanced error handling and delivery tracking"""
        consecutive_errors = 0
        message_tracking = {}
        
        while self.running:
            try:
                message = self._message_queue.get(timeout=0.1)
                if not self.running:  # Check if server was stopped while waiting
                    break
                    
                robot_id = message['robot_id']
                message_id = message.get('message_id')
                
                if message_id:
                    message_tracking[message_id] = {
                        'timestamp': datetime.now().isoformat(),
                        'attempts': 0,
                        'delivered': False
                    }
                
                with self._lock:
                    if robot_id in TelemetryWebSocket.clients:
                        json_message = json.dumps(message)
                        disconnected = set()
                        active_clients = 0
                        delivery_success = False
                        
                        for client in TelemetryWebSocket.clients[robot_id]:
                            try:
                                client.send_message(json_message)
                                active_clients += 1
                                consecutive_errors = 0
                                delivery_success = True
                            except Exception as e:
                                logger.error(f"Error sending to client: {str(e)}")
                                disconnected.add(client)
                                consecutive_errors += 1
                        
                        # Clean up disconnected clients
                        for client in disconnected:
                            if client in TelemetryWebSocket.clients[robot_id]:
                                TelemetryWebSocket.clients[robot_id].remove(client)
                                logger.info(f"Removed disconnected client for robot {robot_id}")

                        # Update message tracking
                        if message_id:
                            message_tracking[message_id]['attempts'] += 1
                            if delivery_success:
                                message_tracking[message_id]['delivered'] = True
                                logger.debug(f"Message {message_id} delivered successfully to {active_clients} clients")
                            elif message_tracking[message_id]['attempts'] < 3:  # Retry up to 3 times
                                if self.running:  # Only requeue if server is still running
                                    self._message_queue.put(message)  # Requeue for retry
                                    logger.warning(f"Message {message_id} delivery failed, requeueing (attempt {message_tracking[message_id]['attempts']})")
                            else:
                                logger.error(f"Message {message_id} delivery failed after 3 attempts")

                        # Log successful broadcast
                        if active_clients > 0:
                            logger.debug(f"Successfully broadcast to {active_clients} clients for robot {robot_id}")
                
                # Clean up old message tracking entries
                current_time = datetime.now()
                old_messages = [msg_id for msg_id, data in message_tracking.items() 
                              if (current_time - datetime.fromisoformat(data['timestamp'])).total_seconds() > 300]
                for msg_id in old_messages:
                    del message_tracking[msg_id]
                
            except queue.Empty:
                continue
            except Exception as e:
                if self.running:  # Only log errors if server is still running
                    logger.error(f"Error in broadcast worker: {str(e)}")
                    consecutive_errors += 1
                    if consecutive_errors > 10:
                        logger.warning("High number of consecutive errors, checking server status")
                        if not self.running:
                            break
                        consecutive_errors = 0

    def _try_restart_server(self):
        """Attempt to restart the server after an error"""
        if self._retry_count < self._max_retries:
            self._retry_count += 1
            logger.info("Attempting server restart (%d/%d)", self._retry_count, self._max_retries)
            try:
                self.stop()
                time.sleep(self._retry_delay)
                if self.start_server():
                    self._retry_count = 0
                    logger.info("Server restarted successfully")
                    return True
            except Exception as e:
                logger.error("Failed to restart server: %s", str(e))
        else:
            logger.error("Max retry attempts reached. Server will not restart.")
        return False

    def get_status(self):
        """Get current server status"""
        return {
            'running': self.running,
            'connection_error': self.connection_error,
            'retry_count': self._retry_count,
            'queue_size': self._message_queue.qsize(),
            'broadcast_errors': self._broadcast_errors
        }

if __name__ == "__main__":
    # For testing the WebSocket server directly
    server = TelemetryWebSocketServer()
    try:
        server.start_server()
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        server.stop()
