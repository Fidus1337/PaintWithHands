import cv2
import os
import HandDetector_module as hdm
import time
import numpy as np
from collections import deque
from dataclasses import dataclass
from typing import Optional, Tuple, List

@dataclass
class Config:
    """Application configuration"""
    WINDOW_WIDTH: int = 1940
    WINDOW_HEIGHT: int = 1080
    OVERLAY_HEIGHT: int = 125
    OVERLAY_WIDTH: int = 1280
    CAMERA_PREVIEW_WIDTH: int = 320
    CAMERA_PREVIEW_HEIGHT: int = 180
    CANVAS_WIDTH: int = 1280
    CANVAS_HEIGHT: int = 720
    BUFFER_SIZE: int = 5
    SMOOTHING_FACTOR: float = 0.7
    BRUSH_THICKNESS: int = 5
    TOOLS_FOLDER: str = "tools_images"
    
    # Drawing area (excluding the tool interface)
    DRAWING_AREA_TOP: int = 125  # Start of drawing area (below tools)
    DRAWING_AREA_BOTTOM: int = 720 * 2  # End of drawing area
    DRAWING_AREA_LEFT: int = 0
    DRAWING_AREA_RIGHT: int = 1280

class CoordinateSmoothing:
    """Class for coordinate smoothing"""
    
    def __init__(self, buffer_size: int = 5, smoothing_factor: float = 0.7):
        self.buffer_x = deque(maxlen=buffer_size)
        self.buffer_y = deque(maxlen=buffer_size)
        self.smooth_x = 0
        self.smooth_y = 0
        self.smoothing_factor = smoothing_factor
        self.initialized = False
    
    def add_coordinates(self, x: int, y: int) -> Tuple[int, int]:
        """Adds coordinates to buffer and returns smoothed coordinates"""
        self.buffer_x.append(x)
        self.buffer_y.append(y)
        
        # Averaging
        avg_x = sum(self.buffer_x) / len(self.buffer_x)
        avg_y = sum(self.buffer_y) / len(self.buffer_y)
        
        # Smoothing
        if not self.initialized:
            self.smooth_x, self.smooth_y = avg_x, avg_y
            self.initialized = True
        else:
            self.smooth_x += (avg_x - self.smooth_x) * self.smoothing_factor
            self.smooth_y += (avg_y - self.smooth_y) * self.smoothing_factor
        
        return int(self.smooth_x), int(self.smooth_y)
    
    def scale_coordinates_to_canvas(self, x: int, y: int, 
                                   camera_width: int, camera_height: int,
                                   canvas_width: int, canvas_height: int,
                                   drawing_area_top: int = 0) -> Tuple[int, int]:
        """Scales coordinates from camera frame to canvas"""
        scaled_x = int((x / camera_width) * canvas_width)
        available_height = canvas_height - drawing_area_top
        scaled_y = int((y / camera_height) * available_height) + drawing_area_top
        scaled_x = max(0, min(scaled_x, canvas_width - 1))
        scaled_y = max(drawing_area_top, min(scaled_y, canvas_height - 1))
        return scaled_x, scaled_y
    
    def clear(self):
        """Clears the coordinate buffer"""
        self.buffer_x.clear()
        self.buffer_y.clear()
        self.initialized = False

class ColorPalette:
    """Class for color palette and tools management"""
    
    COLORS = {
        0: (255, 0, 0),     # Red (default)
        1: (2, 85, 252),    # Orange
        2: (255, 255, 255), # White
        3: (10, 22, 249),   # Blue
        4: (0, 0, 0),       # Black (eraser)
        5: (189, 189, 189), # Gray
        6: (245, 184, 16),  # Yellow
        7: (44, 212, 18),   # Green
        8: (240, 104, 254), # Purple
        9: (0, 0, 0)        # Black
    }
    
    TOOL_BOUNDARIES = [142, 284, 426, 568, 710, 852, 994, 1136, 1278]
    
    @classmethod
    def get_color_by_position(cls, x: int) -> Tuple[int, Tuple[int, int, int]]:
        """Returns tool index and color based on thumb x-coordinate"""
        for i, boundary in enumerate(cls.TOOL_BOUNDARIES):
            if x < boundary:
                tool_index = i + 1
                return tool_index, cls.COLORS.get(tool_index, (255, 0, 0))
        return 9, cls.COLORS[9]
    
    @classmethod
    def draw_tool_zones(cls, canvas: np.ndarray, overlay_height: int):
        """Draws tool selection zones (for debugging)"""
        for i, boundary in enumerate(cls.TOOL_BOUNDARIES):
            cv2.line(canvas, (boundary, 0), (boundary, overlay_height), (100, 100, 100), 1)
            cv2.putText(canvas, str(i + 1), (boundary - 70, overlay_height - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

class ToolsManager:
    """Tool manager"""
    
    def __init__(self, config: Config):
        self.config = config
        self.overlay_list = self._load_tools()
    
    def _load_tools(self) -> List[np.ndarray]:
        """Loads tool overlays from folder"""
        if not os.path.exists(self.config.TOOLS_FOLDER):
            print(f"Folder {self.config.TOOLS_FOLDER} not found. Creating empty tools.")
            return [self._create_empty_tool() for _ in range(10)]
        
        file_list = sorted(os.listdir(self.config.TOOLS_FOLDER), key=lambda x: int(os.path.splitext(x)[0]))
        overlay_list = []
        for file_name in file_list:
            img_path = os.path.join(self.config.TOOLS_FOLDER, file_name)
            overlay_img = cv2.imread(img_path)
            if overlay_img is not None:
                overlay_list.append(overlay_img)
            else:
                print(f"Failed to load image: {img_path}")
                overlay_list.append(self._create_empty_tool())
        
        while len(overlay_list) < 10:
            overlay_list.append(self._create_empty_tool())
        
        print(f"Loaded tools: {len(overlay_list)}")
        return overlay_list
    
    def _create_empty_tool(self) -> np.ndarray:
        """Creates an empty tool placeholder"""
        return np.zeros((self.config.OVERLAY_HEIGHT, self.config.OVERLAY_WIDTH, 3), np.uint8)
    
    def get_tool(self, index: int) -> np.ndarray:
        """Returns tool overlay by index"""
        if 0 <= index < len(self.overlay_list):
            return cv2.resize(self.overlay_list[index], 
                              (self.config.OVERLAY_WIDTH, self.config.OVERLAY_HEIGHT))
        return self._create_empty_tool()

class HandPaintingApp:
    """Main hand-gesture drawing application"""
    
    def __init__(self):
        self.config = Config()
        self.detector = hdm.HandsDetector(max_num_hands=1)
        self.coordinate_smoother = CoordinateSmoothing(
            self.config.BUFFER_SIZE, self.config.SMOOTHING_FACTOR
        )
        self.tools_manager = ToolsManager(self.config)
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.WINDOW_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.WINDOW_HEIGHT)
        
        ret, test_frame = self.cap.read()
        if ret:
            self.camera_height, self.camera_width = test_frame.shape[:2]
            print(f"Camera resolution: {self.camera_width}x{self.camera_height}")
        else:
            self.camera_width = self.config.WINDOW_WIDTH
            self.camera_height = self.config.WINDOW_HEIGHT
        
        self.canvas = np.zeros((self.config.CANVAS_HEIGHT, self.config.CANVAS_WIDTH, 3), np.uint8)
        self.canvas_without_cursor = None
        self.paint_color = (255, 0, 0)
        self.overlay_index = 0
        self.previous_coords: Optional[Tuple[int, int]] = None
        self.was_selection_mode = False
        self.was_drawing_mode = False
        self.ptime = 0

    def _is_selection_mode(self, fingers: List[int]) -> bool:
        """Detects if the current gesture is for tool selection"""
        return fingers[1] and fingers[2] and fingers.count(1) == 2

    def _is_drawing_mode(self, fingers: List[int], selection_mode: bool) -> bool:
        """Detects if the current gesture is for drawing"""
        return fingers.count(1) > 0 and not selection_mode

    def _draw_line(self, current_coords: Tuple[int, int]):
        """Draws a line on the canvas"""
        if self.previous_coords is None:
            self.previous_coords = current_coords
        else:
            cv2.line(self.canvas, self.previous_coords, current_coords, 
                     self.paint_color, self.config.BRUSH_THICKNESS)
        self.previous_coords = current_coords

    def _draw_cursor(self, coords: Tuple[int, int], color: Tuple[int, int, int]):
        """Draws the cursor on the canvas"""
        cv2.circle(self.canvas, coords, 4, color, thickness=3)

    def _handle_tool_selection(self, hand_landmarks: List[List[int]], selection_mode: bool):
        """Handles tool selection using the thumb position"""
        if not selection_mode or len(hand_landmarks) < 5:
            return
        thumb_x, thumb_y = hand_landmarks[8][1], hand_landmarks[8][2]
        if thumb_y < 10:
            old_index = self.overlay_index
            self.overlay_index, self.paint_color = ColorPalette.get_color_by_position(thumb_x)
            if old_index != self.overlay_index:
                print(f"Selected tool {self.overlay_index} with color {self.paint_color}")

    def _add_overlay(self):
        """Adds tool overlay to the canvas"""
        overlay = self.tools_manager.get_tool(self.overlay_index)
        self.canvas[0:self.config.OVERLAY_HEIGHT, 0:self.config.OVERLAY_WIDTH] = overlay
        cv2.putText(self.canvas, f"Tool: {self.overlay_index}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.line(self.canvas, (0, self.config.OVERLAY_HEIGHT), 
                 (self.config.CANVAS_WIDTH, self.config.OVERLAY_HEIGHT), 
                 (100, 100, 100), 2)

    def _add_camera_preview(self, img: np.ndarray):
        """Adds camera preview to the canvas"""
        img_small = cv2.resize(img, (self.config.CAMERA_PREVIEW_WIDTH, 
                                     self.config.CAMERA_PREVIEW_HEIGHT))
        y_start = self.config.OVERLAY_HEIGHT
        y_end = y_start + self.config.CAMERA_PREVIEW_HEIGHT
        self.canvas[y_start:y_end, 0:self.config.CAMERA_PREVIEW_WIDTH] = img_small
        cv2.rectangle(self.canvas, (0, y_start), 
                      (self.config.CAMERA_PREVIEW_WIDTH, y_end), 
                      (255, 255, 255), 2)

    def _calculate_fps(self) -> int:
        """Calculates and returns FPS"""
        ctime = time.time()
        fps = 1 / (ctime - self.ptime) if self.ptime > 0 else 0
        self.ptime = ctime
        return int(fps)

    def run(self):
        """Runs the main application loop"""
        print("Application started. Press 'q' to exit.")
        print("Gesture controls:")
        print("- Index finger up: draw")
        print("- Index + middle finger: select tool")
        print("- In selection mode: point thumb at the tool on the top bar")
        
        self.canvas = np.zeros((self.config.CANVAS_HEIGHT, self.config.CANVAS_WIDTH, 3), np.uint8)
        self.canvas_without_cursor = self.canvas.copy()
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("Failed to capture camera frame")
                break
            
            img = cv2.flip(img, 1)
            img_original = img.copy()
            
            if self.canvas_without_cursor is not None:
                self.canvas = self.canvas_without_cursor.copy()
            
            img = self.detector.detect_hands(img)
            landmarks = self.detector.findPositions(img)
            
            if landmarks:
                hand_type = self.detector.get_hand_type()
                fingers = self.detector.fingersAreUp(hand_type)
                raw_x, raw_y = landmarks[8][1], landmarks[8][2]
                smooth_x, smooth_y = self.coordinate_smoother.add_coordinates(raw_x, raw_y)
                current_coords = self.coordinate_smoother.scale_coordinates_to_canvas(
                    smooth_x, smooth_y,
                    self.camera_width, self.camera_height,
                    self.config.CANVAS_WIDTH, self.config.CANVAS_HEIGHT,
                    self.config.DRAWING_AREA_TOP
                )
                selection_mode = self._is_selection_mode(fingers)
                drawing_mode = self._is_drawing_mode(fingers, selection_mode)
                
                if (self.was_selection_mode and not selection_mode) or \
                   (not self.was_drawing_mode and drawing_mode):
                    self.previous_coords = None
                
                if selection_mode:
                    self._draw_cursor(current_coords, (255, 255, 255))
                    self._handle_tool_selection(landmarks, selection_mode)
                elif drawing_mode:
                    self._draw_line(current_coords)
                    self.canvas_without_cursor = self.canvas.copy()
                    self._draw_cursor(current_coords, self.paint_color)
                else:
                    self.previous_coords = None
                
                self.was_selection_mode = selection_mode
                self.was_drawing_mode = drawing_mode
            else:
                self.coordinate_smoother.clear()
                self.previous_coords = None
                self.was_selection_mode = False
                self.was_drawing_mode = False
            
            self._add_overlay()
            self._add_camera_preview(img_original)
            mode_text = "Selection Mode" if landmarks and self._is_selection_mode(
                self.detector.fingersAreUp(self.detector.get_hand_type())
            ) else "Drawing Mode"
            cv2.putText(self.canvas, mode_text, (self.config.CANVAS_WIDTH - 200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            fps = self._calculate_fps()
            cv2.putText(self.canvas, f"FPS: {fps}", (self.config.CANVAS_WIDTH - 120, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            if landmarks:
                cv2.putText(self.canvas, f"Cursor: {current_coords}", 
                            (self.config.CANVAS_WIDTH - 200, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            if cv2.waitKey(1) & 0xFF == ord('c'):
                self.canvas_without_cursor = np.zeros((self.config.CANVAS_HEIGHT, self.config.CANVAS_WIDTH, 3), np.uint8)
            
            cv2.imshow("Canvas", self.canvas)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        self.cleanup()

    def cleanup(self):
        """Releases resources"""
        self.cap.release()
        cv2.destroyAllWindows()
        print("Application closed.")

# Run the application
if __name__ == "__main__":
    app = HandPaintingApp()
    app.run()
