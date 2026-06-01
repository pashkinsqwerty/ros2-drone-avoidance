from ultralytics import YOLO
import cv2

class DroneVision:
    def __init__(self, model_path='yolov8n.pt'):
        # Используем nano-версию для максимального FPS
        self.model = YOLO(model_path)

    def process_frame(self, frame):
        """
        Прогоняет кадр через YOLO и извлекает метрики для уклонения.
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            for box in result.boxes:
                # Координаты bounding box
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                
                # Вычисляем центр объекта и его "размер" (площадь)
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                
                detections.append({
                    'center': (cx, cy), 
                    'area': area, 
                    'class': cls,
                    'bbox': (x1, y1, x2, y2)
                })
                
        # Возвращаем данные об объектах и кадр с отрисованными боксами для дебага
        return detections, results[0].plot()

class EvasionController:
    def __init__(self, frame_width, frame_height):
        self.width = frame_width
        self.height = frame_height
        
        # Порог срабатывания: если объект занимает больше 25% экрана, считаем его угрозой
        self.critical_area = (frame_width * frame_height) * 0.25 
        # Зона по центру (например, центральные 40% экрана по ширине)
        self.center_margin = self.width * 0.2

    def compute_velocity_commands(self, detections):
        """
        Генерирует команды скорости (linear.x, angular.z или смещения по осям Y).
        В ROS это обычно сообщения типа geometry_msgs/Twist.
        """
        if not detections:
            # Препятствий нет, летим прямо
            return {"forward_speed": 1.0, "yaw_rate": 0.0, "lateral_speed": 0.0}

        # Ищем самую большую (читай "самую близкую") угрозу
        biggest_threat = max(detections, key=lambda x: x['area'])

        if biggest_threat['area'] > self.critical_area:
            cx, cy = biggest_threat['center']
            screen_center_x = self.width // 2

            # Логика уклонения
            if cx > (screen_center_x + self.center_margin):
                # Угроза справа -> летим влево (или поворачиваем влево)
                return {"forward_speed": 0.2, "yaw_rate": 0.5, "lateral_speed": -1.0}
            elif cx < (screen_center_x - self.center_margin):
                # Угроза слева -> летим вправо
                return {"forward_speed": 0.2, "yaw_rate": -0.5, "lateral_speed": 1.0}
            else:
                # Угроза прямо по курсу -> тормозим и набираем высоту или резко смещаемся
                return {"forward_speed": -0.5, "yaw_rate": 0.0, "lateral_speed": 1.0}

        # Если объекты есть, но они далеко — продолжаем движение
        return {"forward_speed": 1.0, "yaw_rate": 0.0, "lateral_speed": 0.0}
