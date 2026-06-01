import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
from ultralytics import YOLO
import math

class DroneAvoidanceNode(Node):
    def __init__(self):
        super().__init__('drone_avoidance_node')
        self.get_logger().info("Запуск Smart Sensor Fusion (YOLO + Vector LiDAR)...")
        
        self.img_sub = self.create_subscription(Image, '/camera/front/image_raw', self.image_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        
        self.min_front_dist = 4.0
        self.wall_detected = False
        self.turn_direction = 1.0 

    def scan_callback(self, msg):
        num_rays = len(msg.ranges)
        if num_rays == 0:
            return
            
        center_idx = num_rays // 2  
        cone_size = int(num_rays * (30.0 / 360.0)) 
        
        # Конусы видимости спереди
        right_cone = msg.ranges[center_idx - cone_size : center_idx]
        left_cone = msg.ranges[center_idx : center_idx + cone_size]
        
        # Фильтруем данные (убираем бесконечность, слишком близкое считаем за 10 см)
        valid_left = [r if r > 0.1 else 0.1 for r in left_cone if not math.isinf(r) and not math.isnan(r)]
        valid_right = [r if r > 0.1 else 0.1 for r in right_cone if not math.isinf(r) and not math.isnan(r)]
        
        min_left = min(valid_left) if valid_left else 4.0
        min_right = min(valid_right) if valid_right else 4.0
        
        self.min_front_dist = min(min_left, min_right)
        self.wall_detected = self.min_front_dist < 1.7 
        
        if self.wall_detected:
            if min_left < min_right:
                self.turn_direction = -1.5 
            else:
                self.turn_direction = 1.5  

    def image_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            self.get_logger().error(f"Ошибка CV Bridge: {e}")
            return

        results = self.model(cv_image, verbose=False)
        cmd = Twist()
        person_detected = False
        frame_center = cv_image.shape[1] / 2.0 
        error = 0.0

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    person_center = (x1 + x2) / 2.0
                    
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(cv_image, "PERSON", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                    
                    error = frame_center - person_center
                    break

        status = ""
        color = (0, 255, 0)

        # Выбор маневра
        if self.wall_detected:
            if self.min_front_dist < 0.6:
                cmd.linear.x = -0.3
                cmd.angular.z = self.turn_direction
                status = f"CRIT WALL! {self.min_front_dist:.1f}m"
                color = (0, 0, 255)
            else:
                cmd.linear.x = 0.15 
                cmd.angular.z = self.turn_direction
                status = f"AVOID WALL {self.min_front_dist:.1f}m"
                color = (255, 100, 0) 
                
        elif person_detected:
            cmd.linear.x = 0.4
            cmd.angular.z = float(error) * 0.005 
            status = "AVOIDING PERSON!"
            color = (0, 0, 255)
            
        else:
            cmd.linear.x = 0.8
            cmd.angular.z = 0.0
            status = "PATROLLING"

        self.publisher.publish(cmd)

        cv2.putText(cv_image, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        cv2.imshow("Drone FPV & YOLO", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DroneAvoidanceNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
