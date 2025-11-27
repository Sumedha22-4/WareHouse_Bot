import cv2
import numpy as np
import pygame
import heapq
import time
from collections import deque
import threading
import random

# Initialize Pygame for visualization
pygame.init()
WIDTH, HEIGHT = 1200, 700
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Warehouse Management System - REAL-TIME")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
GRAY = (200, 200, 200)
ORANGE = (255, 165, 0)
PURPLE = (128, 0, 128)

# Warehouse layout - 20x12 grid
GRID_SIZE = 40
GRID_WIDTH = 20
GRID_HEIGHT = 12

# Define warehouse areas
WAREHOUSE_MAP = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)

# Define obstacles (1 = obstacle, 0 = free space)
WAREHOUSE_MAP[0, :] = 1  # Top wall
WAREHOUSE_MAP[-1, :] = 1  # Bottom wall
WAREHOUSE_MAP[:, 0] = 1  # Left wall
WAREHOUSE_MAP[:, -1] = 1  # Right wall

# Add some internal obstacles
WAREHOUSE_MAP[2:4, 5:8] = 1
WAREHOUSE_MAP[7:9, 12:15] = 1
WAREHOUSE_MAP[4:6, 3:5] = 1

# Define key locations
STORAGE_AREA = (1, 1)
PACKING_AREA = (1, GRID_WIDTH-2)
SHELF_1 = (GRID_HEIGHT-2, 5)   # For boxes
SHELF_2 = (GRID_HEIGHT-2, 10)  # For balls
SHELF_3 = (GRID_HEIGHT-2, 15)  # For bottles
INSPECTION_AREA = (5, GRID_WIDTH-2)

# Robot state
robot_pos = STORAGE_AREA
robot_path = []
current_target = None
object_detected = None
path_finding_in_progress = False
last_object_detection = None
detection_confidence = 0
camera_frame = None
frame_lock = threading.Lock()

# A* pathfinding algorithm
def heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def a_star_search(start, goal, grid):
    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    
    open_set = []
    heapq.heappush(open_set, (0, start))
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}
    
    while open_set:
        _, current = heapq.heappop(open_set)
        
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path
        
        for dx, dy in neighbors:
            neighbor = (current[0] + dx, current[1] + dy)
            
            if (0 <= neighbor[0] < grid.shape[0] and 
                0 <= neighbor[1] < grid.shape[1] and 
                grid[neighbor[0], neighbor[1]] == 0):
                
                tentative_g_score = g_score[current] + 1
                
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                    heapq.heappush(open_set, (f_score[neighbor], neighbor))
    
    return []

# Function to find path in a separate thread
def find_path_async(start, goal):
    global robot_path, path_finding_in_progress
    path_finding_in_progress = True
    robot_path = a_star_search(start, goal, WAREHOUSE_MAP)
    path_finding_in_progress = False

# REAL-TIME Object Detection with enhanced color detection
def detect_objects_real_time(frame):
    if frame is None:
        return "Unknown", 0
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    height, width = frame.shape[:2]
    
    # Define color ranges with better thresholds
    # Red color range
    lower_red1 = np.array([0, 120, 70])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([170, 120, 70])
    upper_red2 = np.array([180, 255, 255])
    
    # Blue color range
    lower_blue = np.array([100, 150, 70])
    upper_blue = np.array([130, 255, 255])
    
    # Green color range
    lower_green = np.array([40, 100, 70])
    upper_green = np.array([80, 255, 255])
    
    # Yellow color range
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([30, 255, 255])
    
    # Create masks
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)
    
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
    
    # Apply morphological operations to reduce noise
    kernel = np.ones((5, 5), np.uint8)
    mask_red = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    mask_blue = cv2.morphologyEx(mask_blue, cv2.MORPH_OPEN, kernel)
    mask_green = cv2.morphologyEx(mask_green, cv2.MORPH_OPEN, kernel)
    mask_yellow = cv2.morphologyEx(mask_yellow, cv2.MORPH_OPEN, kernel)
    
    # Count pixels for each color
    red_pixels = cv2.countNonZero(mask_red)
    blue_pixels = cv2.countNonZero(mask_blue)
    green_pixels = cv2.countNonZero(mask_green)
    yellow_pixels = cv2.countNonZero(mask_yellow)
    
    # Calculate percentage of frame covered
    total_pixels = height * width
    red_ratio = red_pixels / total_pixels
    blue_ratio = blue_pixels / total_pixels
    green_ratio = green_pixels / total_pixels
    yellow_ratio = yellow_pixels / total_pixels
    
    # Threshold for detection (adjust based on testing)
    threshold = 0.01  # 1% of frame
    
    # Find the dominant color
    ratios = {
        "Box": red_ratio,      # Red
        "Ball": blue_ratio,    # Blue
        "Bottle": green_ratio, # Green
        "Package": yellow_ratio # Yellow
    }
    
    max_object = max(ratios, key=ratios.get)
    max_ratio = ratios[max_object]
    
    if max_ratio > threshold:
        return max_object, max_ratio
    else:
        return "Unknown", 0

# Camera processing thread
def camera_processing():
    global camera_frame, object_detected, detection_confidence, last_object_detection
    
    # Try different camera indices
    for i in range(3):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Camera found at index {i}")
            break
    else:
        print("ERROR: No camera found! Please check your camera connection.")
        return
    
    # Set camera resolution for better performance
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                camera_frame = frame.copy()
            
            # Real-time object detection on every frame
            detected_obj, confidence = detect_objects_real_time(frame)
            
            # Only update if we have a confident detection
            if confidence > 0.02:  # Increased threshold for more stability
                object_detected = detected_obj
                detection_confidence = confidence
                last_object_detection = time.time()
        
        time.sleep(0.033)  

# Start camera thread
camera_thread = threading.Thread(target=camera_processing, daemon=True)
camera_thread.start()

# Wait for camera to initialize
time.sleep(2)

# Main loop
running = True
clock = pygame.time.Clock()
last_path_update = 0
path_update_interval = 1.0  # seconds

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                # Reset system
                robot_pos = STORAGE_AREA
                robot_path = []
                current_target = None
                object_detected = None
    
    # Clear screen
    screen.fill(WHITE)
    
    # Draw warehouse grid
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            rect = pygame.Rect(x * GRID_SIZE + 50, y * GRID_SIZE + 50, GRID_SIZE, GRID_SIZE)
            if WAREHOUSE_MAP[y, x] == 1:
                pygame.draw.rect(screen, GRAY, rect)
            else:
                pygame.draw.rect(screen, WHITE, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Draw key areas with labels
    areas = [
        (STORAGE_AREA, GREEN, "STORAGE"),
        (PACKING_AREA, BLUE, "PACKING"),
        (SHELF_1, YELLOW, "SHELF 1 (Box)"),
        (SHELF_2, ORANGE, "SHELF 2 (Ball)"),
        (SHELF_3, PURPLE, "SHELF 3 (Bottle)"),
        (INSPECTION_AREA, RED, "INSPECTION")
    ]
    
    for (y, x), color, label in areas:
        rect = pygame.Rect(x * GRID_SIZE + 50, y * GRID_SIZE + 50, GRID_SIZE, GRID_SIZE)
        pygame.draw.rect(screen, color, rect)
        pygame.draw.rect(screen, BLACK, rect, 2)
        
        # Draw label
        font_small = pygame.font.SysFont(None, 20)
        text = font_small.render(label, True, BLACK)
        screen.blit(text, (x * GRID_SIZE + 45, y * GRID_SIZE + 30))
    
    # Draw path if available
    if robot_path:
        for i in range(len(robot_path) - 1):
            start_pos = (robot_path[i][1] * GRID_SIZE + 50 + GRID_SIZE // 2, 
                         robot_path[i][0] * GRID_SIZE + 50 + GRID_SIZE // 2)
            end_pos = (robot_path[i+1][1] * GRID_SIZE + 50 + GRID_SIZE // 2, 
                       robot_path[i+1][0] * GRID_SIZE + 50 + GRID_SIZE // 2)
            pygame.draw.line(screen, BLUE, start_pos, end_pos, 4)
    
    # Draw robot
    pygame.draw.circle(screen, RED, 
                      (robot_pos[1] * GRID_SIZE + 50 + GRID_SIZE // 2, 
                       robot_pos[0] * GRID_SIZE + 50 + GRID_SIZE // 2), 
                      GRID_SIZE // 3)
    
    # REAL-TIME Decision Making and Path Planning
    current_time = time.time()
    
    # If we detect an object and don't have a current target, plan a path
    if (object_detected and object_detected != "Unknown" and 
        current_target is None and not robot_path and
        current_time - last_path_update > path_update_interval):
        
        # Determine target based on detected object
        if object_detected == "Box":
            current_target = SHELF_1
        elif object_detected == "Ball":
            current_target = SHELF_2
        elif object_detected == "Bottle":
            current_target = SHELF_3
        elif object_detected == "Package":
            current_target = PACKING_AREA
        else:
            current_target = INSPECTION_AREA
        
        # Start path finding in separate thread
        if not path_finding_in_progress:
            threading.Thread(target=find_path_async, args=(robot_pos, current_target)).start()
            last_path_update = current_time
    
    # Move robot along path in real-time
    if robot_path and len(robot_path) > 1:
        # Move one step per frame for real-time movement
        robot_pos = robot_path[1]
        robot_path.pop(0)
        
        # If we reached the target
        if robot_pos == current_target:
            # Simulate processing time
            pygame.time.delay(500)
            # Return to storage
            current_target = STORAGE_AREA
            threading.Thread(target=find_path_async, args=(robot_pos, STORAGE_AREA)).start()
    
    # Display camera feed
    with frame_lock:
        if camera_frame is not None:
            # Resize frame for display
            display_frame = cv2.resize(camera_frame, (320, 240))
            
            # Convert BGR to RGB for Pygame
            display_frame_rgb = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)
            display_frame_surface = pygame.surfarray.make_surface(display_frame_rgb.swapaxes(0, 1))
            screen.blit(display_frame_surface, (WIDTH - 350, 50))
    
    # Draw information panel
    info_panel = pygame.Rect(WIDTH - 400, 300, 380, 350)
    pygame.draw.rect(screen, (240, 240, 240), info_panel)
    pygame.draw.rect(screen, BLACK, info_panel, 2)
    
    # Display status information
    font = pygame.font.SysFont(None, 28)
    font_small = pygame.font.SysFont(None, 24)
    
    screen.blit(font.render("REAL-TIME WAREHOUSE SYSTEM", True, BLUE), (WIDTH - 390, 310))
    
    # Object detection status
    obj_status = f"Detected: {object_detected}" if object_detected else "Detected: None"
    conf_status = f"Confidence: {detection_confidence:.1%}" if object_detected else "Confidence: 0%"
    screen.blit(font_small.render(obj_status, True, BLACK), (WIDTH - 390, 350))
    screen.blit(font_small.render(conf_status, True, BLACK), (WIDTH - 390, 380))
    
    # Robot status
    if current_target:
        target_names = {
            str(SHELF_1): "SHELF 1 (Box)",
            str(SHELF_2): "SHELF 2 (Ball)", 
            str(SHELF_3): "SHELF 3 (Bottle)",
            str(INSPECTION_AREA): "INSPECTION",
            str(STORAGE_AREA): "STORAGE",
            str(PACKING_AREA): "PACKING"
        }
        target_name = target_names.get(str(current_target), "Unknown")
        screen.blit(font_small.render(f"Target: {target_name}", True, BLACK), (WIDTH - 390, 420))
    
    # Path status
    if path_finding_in_progress:
        screen.blit(font_small.render("Status: Finding Path...", True, ORANGE), (WIDTH - 390, 450))
    elif robot_path:
        screen.blit(font_small.render(f"Status: Moving ({len(robot_path)} steps)", True, GREEN), (WIDTH - 390, 450))
    else:
        screen.blit(font_small.render("Status: Waiting for Object", True, BLUE), (WIDTH - 390, 450))
    
    # Instructions
    screen.blit(font_small.render("INSTRUCTIONS:", True, BLACK), (WIDTH - 390, 500))
    screen.blit(font_small.render("Show colored objects to camera:", True, BLACK), (WIDTH - 390, 530))
    screen.blit(font_small.render("RED = Box, BLUE = Ball", True, BLACK), (WIDTH - 390, 555))
    screen.blit(font_small.render("GREEN = Bottle, YELLOW = Package", True, BLACK), (WIDTH - 390, 580))
    screen.blit(font_small.render("Press R to Reset", True, BLACK), (WIDTH - 390, 605))
    
    # Draw camera label
    screen.blit(font_small.render("LIVE CAMERA FEED", True, BLACK), (WIDTH - 350, 20))
    
    pygame.display.flip()
    clock.tick(10)  # Control update rate

# Cleanup
cv2.destroyAllWindows()
pygame.quit()
