import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from typing import List, Tuple, Dict, Optional

class RealTimeObjectDetector:
    """Real-time YOLO object detection"""
    def __init__(self, model_path: str = 'yolov8n.pt'):
        print("🚀 Loading YOLO model for real-time detection...")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        self.detection_threshold = 0.6
        
        # Warehouse object mapping
        self.warehouse_map = {
            'box': ['cardboard box', 'package', 'box'],
            'bottle': ['bottle', 'wine bottle', 'water bottle'],
            'can': ['can', 'tin can'],
            'book': ['book'],
            'cup': ['cup', 'mug'],
            'laptop': ['laptop'],
            'cell phone': ['cell phone'],
            'sports ball': ['sports ball']
        }
        print("✅ YOLO model loaded!")

    def detect_objects(self, frame: np.ndarray) -> List[Dict]:
        """Real YOLO object detection"""
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = box.conf.item()
                    if confidence > self.detection_threshold:
                        class_id = int(box.cls.item())
                        class_name = self.class_names[class_id]
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        
                        warehouse_type = self._map_to_warehouse_type(class_name)
                        if warehouse_type:
                            center_x = int((x1 + x2) / 2)
                            center_y = int((y1 + y2) / 2)
                            
                            detections.append({
                                'type': warehouse_type,
                                'original_class': class_name,
                                'confidence': confidence,
                                'bbox': (int(x1), int(y1), int(x2-x1), int(y2-y1)),
                                'center': (center_x, center_y),
                                'pixel_area': (x2-x1) * (y2-y1)
                            })
        return detections

    def _map_to_warehouse_type(self, class_name: str) -> Optional[str]:
        """Map YOLO classes to warehouse types"""
        for warehouse_type, yolo_classes in self.warehouse_map.items():
            if class_name in yolo_classes:
                return warehouse_type
        return None

class WarehouseCoordinateSystem:
    """Real coordinate system based on camera frame"""
    def __init__(self, frame_width: int, frame_height: int):
        self.width = frame_width
        self.height = frame_height
        self._setup_warehouse_layout()
        
    def _setup_warehouse_layout(self):
        """Define real warehouse areas in pixel coordinates"""
        self.landmarks = {
            'home_base': (self.width // 2, self.height - 50),
            'storage_zone': (100, self.height - 100),
            'packing_station': (self.width - 150, self.height - 100),
            'inspection_area': (self.width // 2, 100)
        }
        
        self.shelves = {
            'shelf_1': {'pos': (200, 150), 'type': 'box'},
            'shelf_2': {'pos': (400, 150), 'type': 'bottle'},
            'shelf_3': {'pos': (600, 150), 'type': 'can'},
            'shelf_4': {'pos': (800, 150), 'type': 'book'},
            'shelf_5': {'pos': (1000, 150), 'type': 'cup'}
        }
        
        # Real obstacles in pixel space
        self.obstacles = [
            {'type': 'wall', 'pos': (300, 200), 'size': (50, 300)},
            {'type': 'wall', 'pos': (500, 300), 'size': (200, 50)},
            {'type': 'pillar', 'pos': (700, 150), 'size': (50, 200)},
            {'type': 'block', 'pos': (150, 400), 'size': (100, 50)}
        ]

    def is_valid_position(self, x: int, y: int) -> bool:
        """Check if position is not obstructed"""
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
            
        for obstacle in self.obstacles:
            ox, oy = obstacle['pos']
            ow, oh = obstacle['size']
            if ox <= x <= ox + ow and oy <= y <= oy + oh:
                return False
        return True

    def get_destination_for_object(self, obj_type: str) -> Tuple[int, int]:
        """Get shelf coordinates for object type"""
        for shelf, info in self.shelves.items():
            if info['type'] == obj_type:
                return info['pos']
        return self.landmarks['inspection_area']

class RealPathFinder:
    """Pure A* path finding without simulation"""
    def __init__(self, coordinate_system: WarehouseCoordinateSystem):
        self.cs = coordinate_system
        self.grid_size = 25
        
    def calculate_path(self, start: Tuple[int, int], goal: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Calculate real path using A* algorithm"""
        print(f"🧭 Calculating path: {start} → {goal}")
        
        open_set = [(0, start)]
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        iterations = 0
        max_iterations = 2000
        
        while open_set and iterations < max_iterations:
            open_set.sort()
            current_f, current = open_set.pop(0)
            
            if self._heuristic(current, goal) < self.grid_size:
                path = self._reconstruct_path(came_from, current)
                print(f"✅ Path calculated: {len(path)} points, distance: {self._calculate_path_length(path):.1f}px")
                return path
            
            for neighbor in self._get_neighbors(current):
                tentative_g = g_score[current] + self._heuristic(current, neighbor)
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                    
                    if neighbor not in [item[1] for item in open_set]:
                        open_set.append((f_score[neighbor], neighbor))
            
            iterations += 1
        
        print("❌ No path found")
        return []

    def _heuristic(self, a: Tuple[int, int], b: Tuple[int, int]) -> float:
        return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2)

    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        x, y = pos
        neighbors = []
        directions = [(self.grid_size,0), (-self.grid_size,0), 
                     (0,self.grid_size), (0,-self.grid_size),
                     (self.grid_size,self.grid_size), (-self.grid_size,-self.grid_size),
                     (self.grid_size,-self.grid_size), (-self.grid_size,self.grid_size)]
        
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.cs.is_valid_position(nx, ny):
                neighbors.append((nx, ny))
        return neighbors

    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        return path[::-1]

    def _calculate_path_length(self, path: List[Tuple[int, int]]) -> float:
        if len(path) < 2:
            return 0
        total = 0
        for i in range(1, len(path)):
            total += self._heuristic(path[i-1], path[i])
        return total

class PureWarehouseBot:
    """Pure warehouse bot - only detection and calculation, no simulation"""
    def __init__(self):
        print("🤖 Initializing Pure Warehouse Management System...")
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("❌ Cannot access camera")
        
        # Get camera properties
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"📷 Camera: {self.width}x{self.height}")
        
        # Initialize components
        self.detector = RealTimeObjectDetector()
        self.coords = WarehouseCoordinateSystem(self.width, self.height)
        self.path_finder = RealPathFinder(self.coords)
        
        # Operation state
        self.current_position = self.coords.landmarks['home_base']
        self.operation_log = []
        
        print("✅ Pure Warehouse Bot Ready!")
        self._print_system_info()

    def _print_system_info(self):
        print(f"\n🏭 PURE WAREHOUSE SYSTEM:")
        print(f"   Resolution: {self.width} x {self.height}")
        print(f"   Home Base: {self.coords.landmarks['home_base']}")
        print(f"   Shelves: {len(self.coords.shelves)}")
        print(f"   Obstacles: {len(self.coords.obstacles)}")
        print("   Mode: REAL-TIME DETECTION + PATH CALCULATION\n")

    def process_real_frame(self) -> Dict:
        """Process single real camera frame"""
        ret, frame = self.cap.read()
        if not ret:
            return {'error': 'Frame read failed'}
        
        # Real object detection
        start_time = time.time()
        objects = self.detector.detect_objects(frame)
        detection_time = time.time() - start_time
        
        # Create perception report
        perception = self._create_perception_report(objects, detection_time)
        
        # Calculate paths for detected objects
        if objects:
            self._calculate_object_paths(objects, perception)
        
        return perception

    def _create_perception_report(self, objects: List[Dict], detection_time: float) -> Dict:
        """Create detailed perception report"""
        print(f"\n{'='*80}")
        print(f"👁️  REAL-TIME PERCEPTION - {time.strftime('%H:%M:%S')}")
        print(f"   Detection Time: {detection_time*1000:.1f}ms")
        print(f"   Current Position: {self.current_position}")
        
        report = {
            'timestamp': time.time(),
            'detection_time_ms': detection_time * 1000,
            'objects_detected': objects,
            'current_position': self.current_position,
            'frame_resolution': (self.width, self.height)
        }
        
        # Report detected objects
        if objects:
            print(f"   🔍 OBJECTS DETECTED ({len(objects)}):")
            for obj in objects:
                print(f"     📦 {obj['type'].upper():<8} {obj['original_class']:<15} "
                      f"conf: {obj['confidence']:.3f} at {obj['center']} "
                      f"area: {obj['pixel_area']:.0f}px²")
        else:
            print(f"   🔍 No objects detected")
            
        return report

    def _calculate_object_paths(self, objects: List[Dict], perception: Dict):
        """Calculate paths for all detected objects"""
        print(f"\n   🧭 PATH CALCULATIONS:")
        
        paths_calculated = []
        
        for i, obj in enumerate(objects):
            obj_pos = obj['center']
            destination = self.coords.get_destination_for_object(obj['type'])
            
            # Calculate path from current position to object
            pickup_path = self.path_finder.calculate_path(self.current_position, obj_pos)
            
            # Calculate path from object to destination
            delivery_path = self.path_finder.calculate_path(obj_pos, destination)
            
            if pickup_path and delivery_path:
                total_path = pickup_path + delivery_path[1:]  # Avoid duplicate point
                total_distance = (self.path_finder._calculate_path_length(pickup_path) + 
                                self.path_finder._calculate_path_length(delivery_path))
                
                path_info = {
                    'object_type': obj['type'],
                    'object_position': obj_pos,
                    'destination': destination,
                    'pickup_path_points': len(pickup_path),
                    'delivery_path_points': len(delivery_path),
                    'total_path_points': len(total_path),
                    'total_distance_pixels': total_distance,
                    'pickup_path': pickup_path[:3] + ['...'] + pickup_path[-2:] if len(pickup_path) > 5 else pickup_path,
                    'delivery_path': delivery_path[:3] + ['...'] + delivery_path[-2:] if len(delivery_path) > 5 else delivery_path
                }
                
                paths_calculated.append(path_info)
                
                print(f"     🎯 {obj['type'].upper()} Path:")
                print(f"       📍 Object: {obj_pos} → Shelf: {destination}")
                print(f"       🗺️  Pickup: {len(pickup_path)} points, "
                      f"Delivery: {len(delivery_path)} points")
                print(f"       📏 Total: {len(total_path)} points, {total_distance:.1f} pixels")
                
        perception['calculated_paths'] = paths_calculated

    def run_pure_operations(self, duration: int = 600):
        """Run pure detection and calculation operations"""
        print(f"\n🚀 STARTING PURE DETECTION & PATH CALCULATION")
        print(f"   Duration: {duration} seconds")
        print(f"   Mode: REAL-TIME OBJECT DETECTION + PATH FINDING")
        print("=" * 80)
        
        start_time = time.time()
        frame_count = 0
        
        try:
            while time.time() - start_time < duration:
                frame_count += 1
                
                print(f"\n📊 FRAME {frame_count} - "
                      f"Elapsed: {time.time() - start_time:.1f}s")
                
                # Process real frame
                perception = self.process_real_frame()
                self.operation_log.append(perception)
                
                # Calculate and display statistics
                self._display_operation_stats(frame_count, start_time, perception)
                
                # Brief pause to prevent overwhelming output
                time.sleep(0.5)
                
        except KeyboardInterrupt:
            print(f"\n⏹️  Operation interrupted")
        finally:
            self._shutdown_system(frame_count, start_time)

    def _display_operation_stats(self, frame_count: int, start_time: float, perception: Dict):
        """Display operation statistics"""
        total_time = time.time() - start_time
        fps = frame_count / total_time if total_time > 0 else 0
        
        objects_detected = len(perception.get('objects_detected', []))
        paths_calculated = len(perception.get('calculated_paths', []))
        
        print(f"\n📈 REAL-TIME STATS:")
        print(f"   📡 Processing: {fps:.1f} FPS")
        print(f"   🔍 Objects: {objects_detected} detected")
        print(f"   🗺️  Paths: {paths_calculated} calculated")
        print(f"   ⏱️  Detection: {perception.get('detection_time_ms', 0):.1f}ms")
        
        if paths_calculated > 0:
            total_distance = sum(p['total_distance_pixels'] 
                               for p in perception['calculated_paths'])
            avg_distance = total_distance / paths_calculated
            print(f"   📏 Avg Path: {avg_distance:.1f} pixels")

    def _shutdown_system(self, frame_count: int, start_time: float):
        """Shutdown and generate final report"""
        self.cap.release()
        
        total_time = time.time() - start_time
        
        # Generate final statistics
        total_objects = sum(len(log.get('objects_detected', [])) 
                          for log in self.operation_log)
        total_paths = sum(len(log.get('calculated_paths', [])) 
                         for log in self.operation_log)
        
        print(f"\n{'='*80}")
        print("📊 PURE OPERATION COMPLETE - FINAL REPORT:")
        print(f"   ✅ Frames Processed: {frame_count}")
        print(f"   ⏱️  Total Time: {total_time:.2f}s")
        print(f"   📡 Avg FPS: {frame_count/total_time:.2f}")
        print(f"   🔍 Total Objects Detected: {total_objects}")
        print(f"   🗺️  Total Paths Calculated: {total_paths}")
        print(f"   📊 Operation Logs: {len(self.operation_log)} entries")
        
        if total_paths > 0:
            all_paths = []
            for log in self.operation_log:
                all_paths.extend(log.get('calculated_paths', []))
            
            avg_distance = sum(p['total_distance_pixels'] for p in all_paths) / len(all_paths)
            print(f"   📏 Average Path Distance: {avg_distance:.1f} pixels")
        
        print(f"\n🤖 Pure Warehouse System shutdown complete.")

def main():
    """Main execution"""
    try:
        print("🚀 PURE WAREHOUSE MANAGEMENT SYSTEM")
        print("   Real-time Object Detection + Path Calculation")
        print("   No Simulation - No UI - Pure Processing")
        
        bot = PureWarehouseBot()
        bot.run_pure_operations(duration=600)  # 10 minutes
        
    except Exception as e:
        print(f"❌ System Error: {e}")
        print("💡 Check camera connection and YOLO model installation")

if __name__ == "__main__":
    main()
