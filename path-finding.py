import cv2
import numpy as np
from ultralytics import YOLO
import heapq
import time
from datetime import datetime
import json

# ============================================
# CONFIGURATION
# ============================================

# Warehouse Grid Configuration (in meters)
WAREHOUSE_WIDTH = 20
WAREHOUSE_HEIGHT = 15

# Warehouse Layout Definitions (x, y coordinates in meters)
POINT_A = (2, 2)      # Storage/Pickup Point
POINT_B = (18, 13)    # Packing Area
SHELF_1 = (10, 3)     # Shelf 1
SHELF_2 = (15, 8)     # Shelf 2
INSPECTION = (5, 13)  # Inspection Area

# Robot Starting Position
ROBOT_START = (1, 1)

# Robot Specifications
ROBOT_SPEED = 1.5  # meters per second
ROBOT_TURN_TIME = 0.5  # seconds per 90-degree turn
PICKUP_TIME = 2.0  # seconds to pick up item
PLACE_TIME = 2.0   # seconds to place item

# Obstacles (walls, pillars, etc.)
OBSTACLES = [
    (6, 5), (7, 5), (8, 5), (9, 5),
    (6, 6), (9, 6),
    (6, 7), (9, 7),
    (6, 8), (7, 8), (8, 8), (9, 8),
    (13, 10), (14, 10), (15, 10),
    (13, 11), (15, 11),
]

# Object-to-Destination Mapping
DESTINATION_MAP = {
    'bottle': SHELF_2,
    'cup': SHELF_1,
    'bowl': SHELF_1,
    'book': SHELF_2,
    'cell phone': INSPECTION,
    'laptop': INSPECTION,
    'mouse': SHELF_1,
    'keyboard': SHELF_2,
    'backpack': POINT_B,
    'handbag': POINT_B,
    'unknown': INSPECTION
}

# ============================================
# A* PATH FINDING ALGORITHM
# ============================================

class AStarPathFinder:
    def __init__(self, width, height, obstacles):
        self.width = width
        self.height = height
        self.obstacles = set(obstacles)
        self.total_paths_computed = 0
        self.total_nodes_explored = 0
    
    def heuristic(self, pos1, pos2):
        """Euclidean distance heuristic for accurate real-world distance"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_neighbors(self, pos):
        """Get valid neighboring cells (8-directional movement)"""
        x, y = pos
        neighbors = [
            (x + 1, y, 1.0),         # Right
            (x - 1, y, 1.0),         # Left
            (x, y + 1, 1.0),         # Down
            (x, y - 1, 1.0),         # Up
            (x + 1, y + 1, 1.414),   # Down-Right (diagonal)
            (x + 1, y - 1, 1.414),   # Up-Right
            (x - 1, y + 1, 1.414),   # Down-Left
            (x - 1, y - 1, 1.414),   # Up-Left
        ]
        
        valid_neighbors = []
        for nx, ny, cost in neighbors:
            if (0 <= nx < self.width and 
                0 <= ny < self.height and 
                (nx, ny) not in self.obstacles):
                valid_neighbors.append(((nx, ny), cost))
        
        return valid_neighbors
    
    def calculate_path_distance(self, path):
        """Calculate actual distance traveled in meters"""
        if len(path) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance += distance
        
        return total_distance
    
    def calculate_turns(self, path):
        """Calculate number of direction changes"""
        if len(path) < 3:
            return 0
        
        turns = 0
        for i in range(1, len(path) - 1):
            prev_dx = path[i][0] - path[i-1][0]
            prev_dy = path[i][1] - path[i-1][1]
            next_dx = path[i+1][0] - path[i][0]
            next_dy = path[i+1][1] - path[i][1]
            
            if (prev_dx, prev_dy) != (next_dx, next_dy):
                turns += 1
        
        return turns
    
    def find_path(self, start, goal):
        """A* pathfinding algorithm with detailed statistics"""
        start_time = time.time()
        
        if start in self.obstacles or goal in self.obstacles:
            return None, None
        
        counter = 0
        nodes_explored = 0
        frontier = [(0, counter, start, [start], 0)]
        explored = set()
        g_scores = {start: 0}
        
        while frontier:
            f_score, _, current, path, g_score = heapq.heappop(frontier)
            nodes_explored += 1
            
            if current == goal:
                computation_time = time.time() - start_time
                distance = self.calculate_path_distance(path)
                turns = self.calculate_turns(path)
                
                self.total_paths_computed += 1
                self.total_nodes_explored += nodes_explored
                
                stats = {
                    'path': path,
                    'path_length_steps': len(path),
                    'distance_meters': round(distance, 2),
                    'turns': turns,
                    'nodes_explored': nodes_explored,
                    'computation_time_ms': round(computation_time * 1000, 2),
                    'success': True
                }
                return path, stats
            
            if current in explored:
                continue
            
            explored.add(current)
            
            for neighbor, move_cost in self.get_neighbors(current):
                tentative_g = g_score + move_cost
                
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    g_scores[neighbor] = tentative_g
                    h_score = self.heuristic(neighbor, goal)
                    f_score = tentative_g + h_score
                    
                    counter += 1
                    new_path = path + [neighbor]
                    heapq.heappush(frontier, (f_score, counter, neighbor, new_path, tentative_g))
        
        computation_time = time.time() - start_time
        stats = {
            'path': None,
            'path_length_steps': 0,
            'distance_meters': 0,
            'turns': 0,
            'nodes_explored': nodes_explored,
            'computation_time_ms': round(computation_time * 1000, 2),
            'success': False,
            'error': 'No path found'
        }
        return None, stats

# ============================================
# OBJECT DETECTION MODULE
# ============================================

class ObjectDetector:
    def __init__(self, model_name='yolov8n.pt'):
        """Initialize YOLO model"""
        print("📦 Loading YOLO model...")
        self.model = YOLO(model_name)
        self.detection_count = 0
        self.detection_history = []
        print("✅ YOLO model loaded successfully\n")
    
    def detect_object(self, frame):
        """Detect objects and return detailed statistics"""
        start_time = time.time()
        
        results = self.model(frame, verbose=False)
        detection_time = time.time() - start_time
        
        self.detection_count += 1
        
        if len(results) == 0 or len(results[0].boxes) == 0:
            stats = {
                'success': False,
                'object_name': None,
                'confidence': 0.0,
                'detection_time_ms': round(detection_time * 1000, 2),
                'frame_resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                'total_detections': self.detection_count
            }
            return None, stats
        
        boxes = results[0].boxes
        
        # Find highest confidence detection
        max_conf = 0
        best_detection = None
        all_detections = []
        
        for box in boxes:
            confidence = float(box.conf[0].cpu().numpy())
            class_id = int(box.cls[0].cpu().numpy())
            class_name = self.model.names[class_id]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            area = (x2 - x1) * (y2 - y1)
            
            detection_info = {
                'class': class_name,
                'confidence': round(confidence, 4),
                'bbox_area_pixels': int(area)
            }
            all_detections.append(detection_info)
            
            if confidence > max_conf:
                max_conf = confidence
                best_detection = detection_info
        
        if best_detection and max_conf > 0.3:
            stats = {
                'success': True,
                'object_name': best_detection['class'],
                'confidence': best_detection['confidence'],
                'bbox_area_pixels': best_detection['bbox_area_pixels'],
                'detection_time_ms': round(detection_time * 1000, 2),
                'frame_resolution': f"{frame.shape[1]}x{frame.shape[0]}",
                'total_objects_in_frame': len(all_detections),
                'all_detections': all_detections,
                'total_detections': self.detection_count
            }
            
            self.detection_history.append({
                'timestamp': datetime.now().isoformat(),
                'object': best_detection['class'],
                'confidence': best_detection['confidence']
            })
            
            return best_detection['class'], stats
        
        stats = {
            'success': False,
            'object_name': None,
            'confidence': max_conf,
            'detection_time_ms': round(detection_time * 1000, 2),
            'reason': 'Low confidence detections only',
            'total_detections': self.detection_count
        }
        return None, stats

# ============================================
# MISSION ANALYTICS
# ============================================

class MissionAnalytics:
    def __init__(self):
        self.missions = []
        self.total_distance = 0.0
        self.total_time = 0.0
        
    def calculate_mission_time(self, path_stats_1, path_stats_2):
        """Calculate total mission time including movement and operations"""
        # Movement time
        distance_1 = path_stats_1['distance_meters']
        distance_2 = path_stats_2['distance_meters']
        travel_time_1 = distance_1 / ROBOT_SPEED
        travel_time_2 = distance_2 / ROBOT_SPEED
        
        # Turn time
        turns_1 = path_stats_1['turns']
        turns_2 = path_stats_2['turns']
        turn_time_1 = turns_1 * ROBOT_TURN_TIME
        turn_time_2 = turns_2 * ROBOT_TURN_TIME
        
        # Total time
        total_time = (travel_time_1 + travel_time_2 + 
                     turn_time_1 + turn_time_2 + 
                     PICKUP_TIME + PLACE_TIME)
        
        return {
            'travel_time_sec': round(travel_time_1 + travel_time_2, 2),
            'turn_time_sec': round(turn_time_1 + turn_time_2, 2),
            'pickup_time_sec': PICKUP_TIME,
            'place_time_sec': PLACE_TIME,
            'total_mission_time_sec': round(total_time, 2),
            'total_mission_time_min': round(total_time / 60, 2)
        }
    
    def calculate_energy(self, distance, time):
        """Estimate energy consumption (simplified model)"""
        # Assume 50W for movement, 20W for idle
        movement_energy = 50 * time  # Watt-seconds
        return round(movement_energy / 3600, 4)  # Convert to Wh
    
    def log_mission(self, mission_data):
        """Log mission and update statistics"""
        self.missions.append(mission_data)
        self.total_distance += mission_data['total_distance_meters']
        self.total_time += mission_data['timing']['total_mission_time_sec']
    
    def get_summary(self):
        """Get overall system statistics"""
        if not self.missions:
            return None
        
        successful_missions = [m for m in self.missions if m['success']]
        
        return {
            'total_missions': len(self.missions),
            'successful_missions': len(successful_missions),
            'failed_missions': len(self.missions) - len(successful_missions),
            'success_rate': round(len(successful_missions) / len(self.missions) * 100, 2),
            'total_distance_traveled_meters': round(self.total_distance, 2),
            'total_distance_traveled_km': round(self.total_distance / 1000, 2),
            'total_operation_time_sec': round(self.total_time, 2),
            'total_operation_time_min': round(self.total_time / 60, 2),
            'average_mission_time_sec': round(self.total_time / len(successful_missions), 2) if successful_missions else 0,
            'total_energy_consumption_wh': round(sum([m.get('energy_wh', 0) for m in successful_missions]), 2)
        }

# ============================================
# MAIN WAREHOUSE MANAGEMENT SYSTEM
# ============================================

class WarehouseBot:
    def __init__(self):
        print("=" * 70)
        print("🤖 WAREHOUSE MANAGEMENT SYSTEM - PURE BACKEND")
        print("=" * 70)
        print()
        
        self.detector = ObjectDetector()
        self.pathfinder = AStarPathFinder(WAREHOUSE_WIDTH, WAREHOUSE_HEIGHT, OBSTACLES)
        self.analytics = MissionAnalytics()
        self.robot_pos = ROBOT_START
        self.mission_count = 0
        
    def get_destination(self, object_name):
        """Determine destination based on object type"""
        for key in DESTINATION_MAP:
            if key in object_name.lower():
                return DESTINATION_MAP[key], key
        return DESTINATION_MAP['unknown'], 'unknown'
    
    def print_section(self, title):
        """Print formatted section header"""
        print("\n" + "─" * 70)
        print(f"  {title}")
        print("─" * 70)
    
    def print_stats(self, data, indent=0):
        """Pretty print statistics"""
        prefix = "  " * indent
        for key, value in data.items():
            if isinstance(value, dict):
                print(f"{prefix}📊 {key}:")
                self.print_stats(value, indent + 1)
            elif isinstance(value, list):
                print(f"{prefix}📋 {key}:")
                for item in value:
                    if isinstance(item, dict):
                        self.print_stats(item, indent + 1)
                    else:
                        print(f"{prefix}  • {item}")
            else:
                print(f"{prefix}▸ {key}: {value}")
    
    def execute_mission(self, object_name, destination, detection_stats):
        """Execute complete pick and place mission with full analytics"""
        self.mission_count += 1
        mission_start_time = time.time()
        
        self.print_section(f"MISSION #{self.mission_count} STARTED")
        print(f"⏰ Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"📦 Object: {object_name}")
        print(f"🎯 Destination: {destination}")
        print()
        
        mission_data = {
            'mission_id': self.mission_count,
            'timestamp': datetime.now().isoformat(),
            'object': object_name,
            'destination': destination,
            'start_position': self.robot_pos,
            'detection_stats': detection_stats
        }
        
        # Phase 1: Navigate to Point A
        print(f"📍 PHASE 1: Navigation to Storage Point A")
        print(f"   Route: {self.robot_pos} → {POINT_A}")
        
        path_1, stats_1 = self.pathfinder.find_path(self.robot_pos, POINT_A)
        
        if not path_1:
            print("❌ MISSION FAILED: Cannot reach Point A")
            mission_data['success'] = False
            mission_data['failure_reason'] = 'No path to Point A'
            self.analytics.log_mission(mission_data)
            return False
        
        print(f"   ✅ Path computed successfully")
        self.print_stats(stats_1, indent=1)
        
        mission_data['phase_1'] = stats_1
        self.robot_pos = POINT_A
        
        print(f"\n   🤏 Picking up {object_name}...")
        time.sleep(0.1)  # Simulate pickup
        print(f"   ✅ Pickup complete ({PICKUP_TIME}s)")
        
        # Phase 2: Navigate to Destination
        print(f"\n📍 PHASE 2: Navigation to Destination")
        print(f"   Route: {POINT_A} → {destination}")
        
        path_2, stats_2 = self.pathfinder.find_path(POINT_A, destination)
        
        if not path_2:
            print("❌ MISSION FAILED: Cannot reach destination")
            mission_data['success'] = False
            mission_data['failure_reason'] = 'No path to destination'
            self.analytics.log_mission(mission_data)
            return False
        
        print(f"   ✅ Path computed successfully")
        self.print_stats(stats_2, indent=1)
        
        mission_data['phase_2'] = stats_2
        self.robot_pos = destination
        
        print(f"\n   📦 Placing {object_name}...")
        time.sleep(0.1)  # Simulate placement
        print(f"   ✅ Placement complete ({PLACE_TIME}s)")
        
        # Calculate mission metrics
        total_distance = stats_1['distance_meters'] + stats_2['distance_meters']
        timing = self.analytics.calculate_mission_time(stats_1, stats_2)
        energy = self.analytics.calculate_energy(total_distance, timing['total_mission_time_sec'])
        
        mission_execution_time = time.time() - mission_start_time
        
        mission_data['success'] = True
        mission_data['total_distance_meters'] = round(total_distance, 2)
        mission_data['timing'] = timing
        mission_data['energy_wh'] = energy
        mission_data['actual_execution_time_sec'] = round(mission_execution_time, 2)
        
        # Print mission summary
        self.print_section(f"MISSION #{self.mission_count} COMPLETE ✅")
        
        summary = {
            'Total Distance': f"{total_distance:.2f} meters",
            'Total Steps': stats_1['path_length_steps'] + stats_2['path_length_steps'],
            'Total Turns': stats_1['turns'] + stats_2['turns'],
            'Mission Time': f"{timing['total_mission_time_sec']:.2f} seconds ({timing['total_mission_time_min']:.2f} min)",
            'Energy Consumed': f"{energy:.4f} Wh",
            'Path Computation Time': f"{stats_1['computation_time_ms'] + stats_2['computation_time_ms']:.2f} ms",
            'Nodes Explored': stats_1['nodes_explored'] + stats_2['nodes_explored']
        }
        
        self.print_stats(summary)
        
        self.analytics.log_mission(mission_data)
        
        # Reset robot position
        self.robot_pos = ROBOT_START
        
        return True
    
    def run(self):
        """Main execution loop"""
        print("\n📹 Initializing camera...")
        
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ ERROR: Cannot access camera!")
            return
        
        print("✅ Camera initialized successfully")
        print("\n" + "=" * 70)
        print("SYSTEM READY")
        print("=" * 70)
        print("\n📋 Instructions:")
        print("   • Show an object to the camera")
        print("   • Press SPACE to detect and execute mission")
        print("   • Press 'S' to view system statistics")
        print("   • Press 'Q' to quit and see final report")
        print()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Minimal display (only for capture, no UI)
            cv2.imshow('Camera', frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # Detect and execute mission
            if key == ord(' '):
                self.print_section("OBJECT DETECTION INITIATED")
                
                obj_name, detection_stats = self.detector.detect_object(frame)
                
                if obj_name:
                    print("✅ DETECTION SUCCESSFUL")
                    self.print_stats(detection_stats)
                    
                    destination, dest_type = self.get_destination(obj_name)
                    
                    # Execute mission
                    self.execute_mission(obj_name, destination, detection_stats)
                    
                else:
                    print("❌ DETECTION FAILED")
                    self.print_stats(detection_stats)
                    print("\n⚠️  Please show a clear object and try again")
            
            # Show statistics
            elif key == ord('s'):
                summary = self.analytics.get_summary()
                if summary:
                    self.print_section("SYSTEM STATISTICS")
                    self.print_stats(summary)
                else:
                    print("\n⚠️  No missions completed yet")
            
            # Quit
            elif key == ord('q'):
                break
        
        # Final report
        cap.release()
        cv2.destroyAllWindows()
        
        self.print_section("FINAL SYSTEM REPORT")
        
        summary = self.analytics.get_summary()
        if summary:
            self.print_stats(summary)
            
            print("\n📊 Mission History:")
            for mission in self.analytics.missions:
                print(f"\n  Mission #{mission['mission_id']}:")
                print(f"    • Object: {mission['object']}")
                print(f"    • Distance: {mission['total_distance_meters']:.2f}m")
                print(f"    • Time: {mission['timing']['total_mission_time_sec']:.2f}s")
                print(f"    • Success: {'✅' if mission['success'] else '❌'}")
        else:
            print("No missions were executed.")
        
        print("\n" + "=" * 70)
        print("SYSTEM SHUTDOWN COMPLETE")
        print("=" * 70)

# ============================================
# RUN THE SYSTEM
# ============================================

if __name__ == "__main__":
    try:
        bot = WarehouseBot()
        bot.run()
    except KeyboardInterrupt:
        print("\n\n⚠️  System interrupted by user")
    except Exception as e:
        print(f"\n❌ CRITICAL ERROR: {e}")
        import traceback
        traceback.print_exc()

