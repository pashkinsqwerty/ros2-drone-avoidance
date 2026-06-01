import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, LaserScan, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist
from rclpy.qos import qos_profile_sensor_data
from cv_bridge import CvBridge
import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
import random

class DroneAvoidanceNode(Node):
    def __init__(self):
        super().__init__('drone_avoidance_node')
        self.get_logger().info("Запуск Ultimate Sensor Fusion (Патч: Anti-Ping-Pong & Vector Sum)")
        
        self.img_sub = self.create_subscription(Image, '/camera/front/image_raw', self.image_callback, qos_profile_sensor_data)
        self.depth_sub = self.create_subscription(Image, '/camera/depth/image_raw', self.depth_callback, qos_profile_sensor_data)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, qos_profile_sensor_data)
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.imu_sub = self.create_subscription(Imu, '/imu/data', self.imu_callback, qos_profile_sensor_data)
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.bridge = CvBridge()
        self.model = YOLO('yolov8n.pt')
        self.latest_depth_image = None
        
        self.x, self.y = 0.0, 0.0
        self.true_yaw, self.noisy_yaw, self.kf_yaw = 0.0, 0.0, 0.0
        self.kf_p, self.kf_q, self.kf_r = 1.0, 0.001, 0.2
        self.last_imu_time = time.time()
        
        self.waypoints = [[8.0, 8.0], [-8.0, 8.0], [-8.0, -8.0], [8.0, -8.0]]
        self.wp_index = 0
        self.target_x = self.waypoints[self.wp_index][0]
        self.target_y = self.waypoints[self.wp_index][1]
        
        self.lidar_repulsive_force = 0.0
        self.yolo_repulsive_force = 0.0
        self.person_detected = False
        self.min_front_dist = 4.0 
        self.stuck_in_wall = False

    def euler_from_quaternion(self, x, y, z, w):
        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        return math.atan2(t3, t4)

    def imu_callback(self, msg):
        dt = time.time() - self.last_imu_time
        self.last_imu_time = time.time()
        self.kf_yaw += msg.angular_velocity.z * dt
        self.kf_p += self.kf_q

    def odom_callback(self, msg):
        self.x, self.y = msg.pose.pose.position.x, msg.pose.pose.position.y
        q = msg.pose.pose.orientation
        self.true_yaw = self.euler_from_quaternion(q.x, q.y, q.z, q.w)
        self.noisy_yaw = self.true_yaw + random.gauss(0, 0.4)
        if self.noisy_yaw > math.pi: self.noisy_yaw -= 2 * math.pi
        if self.noisy_yaw < -math.pi: self.noisy_yaw += 2 * math.pi
        
        k_gain = self.kf_p / (self.kf_p + self.kf_r)
        yaw_error = self.noisy_yaw - self.kf_yaw
        if yaw_error > math.pi: yaw_error -= 2 * math.pi
        if yaw_error < -math.pi: yaw_error += 2 * math.pi
        self.kf_yaw += k_gain * yaw_error
        self.kf_p = (1.0 - k_gain) * self.kf_p

    # === ИСТИННОЕ СЛОЖЕНИЕ ВЕКТОРОВ (ANTI-PING-PONG) ===
    def scan_callback(self, msg):
        num_rays = len(msg.ranges)
        if num_rays == 0: return
        center_idx = num_rays // 2  
        cone_size = int(num_rays * (45.0 / 360.0)) 
        
        def safe_ray(r):
            if math.isnan(r) or r < 0.1: return 0.1
            if math.isinf(r): return 4.0
            return r

        left_cone = [safe_ray(r) for r in msg.ranges[center_idx : center_idx + cone_size]]
        right_cone = [safe_ray(r) for r in msg.ranges[center_idx - cone_size : center_idx]]
        
        min_left = min(left_cone) if left_cone else 4.0
        min_right = min(right_cone) if right_cone else 4.0
        
        self.min_front_dist = min(min_left, min_right)
        self.stuck_in_wall = self.min_front_dist <= 0.15

        force_left = 0.0
        force_right = 0.0
        safe_dist = 1.5 # Начинаем реагировать более плавно, за 1.5 метра

        if min_left < safe_dist:
            force_left = - (safe_dist - min_left) * 1.5 # Левая стена толкает вправо (-)
        if min_right < safe_dist:
            force_right = (safe_dist - min_right) * 1.5 # Правая стена толкает влево (+)

        # Складываем векторы! Если мы в центре коридора, силы обнулят друг друга.
        self.lidar_repulsive_force = force_left + force_right

    def depth_callback(self, msg):
        try: 
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            if cv_img.dtype != np.float32:
                cv_img = cv_img.astype(np.float32) / 1000.0
            self.latest_depth_image = cv_img
        except Exception as e:
            pass

    def image_callback(self, msg):
        try: cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception: return

        results = self.model(cv_image, verbose=False)
        self.person_detected = False
        self.yolo_repulsive_force = 0.0
        frame_center = cv_image.shape[1] / 2.0 
        emergency_brake_person = False

        for r in results:
            for box in r.boxes:
                if int(box.cls[0]) == 0:
                    self.person_detected = True
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = int((x1 + x2) / 2.0), int((y1 + y2) / 2.0)
                    
                    person_depth_meters = 5.0
                    if self.latest_depth_image is not None:
                        if 0 <= cx < self.latest_depth_image.shape[1] and 0 <= cy < self.latest_depth_image.shape[0]:
                            depth_val = self.latest_depth_image[cy, cx]
                            if math.isnan(depth_val): person_depth_meters = 0.1
                            elif not math.isinf(depth_val): person_depth_meters = float(depth_val)
                    
                    cv2.rectangle(cv_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(cv_image, f"DIST: {person_depth_meters:.2f} m", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    cv2.circle(cv_image, (cx, cy), 5, (0, 255, 255), -1)
                    
                    error = frame_center - cx
                    force_multiplier = 1.5 / (person_depth_meters + 0.1)
                    self.yolo_repulsive_force = float(error) * 0.01 * force_multiplier
                    
                    if person_depth_meters < 1.5: 
                        emergency_brake_person = True
                    break

        cmd = Twist()
        dist_to_target = math.sqrt((self.target_x - self.x)**2 + (self.target_y - self.y)**2)
        if dist_to_target < 1.0:
            self.wp_index = (self.wp_index + 1) % len(self.waypoints)
            self.target_x = self.waypoints[self.wp_index][0]
            self.target_y = self.waypoints[self.wp_index][1]

        angle_to_target = math.atan2(self.target_y - self.y, self.target_x - self.x)
        angle_error = angle_to_target - self.kf_yaw
        if angle_error > math.pi: angle_error -= 2 * math.pi
        if angle_error < -math.pi: angle_error += 2 * math.pi
        
        target_attractive_force = angle_error * 0.8 
        status = ""
        color = (0, 255, 0)
        
        if self.stuck_in_wall:
            cmd.linear.x = -0.5 
            cmd.angular.z = max(-1.5, min(1.5, self.lidar_repulsive_force * 3.0)) 
            status = "STUCK! REVERSING!"
            color = (0, 0, 255)
            cv2.rectangle(cv_image, (0,0), (cv_image.shape[1], cv_image.shape[0]), (0, 0, 255), 20)

        elif self.person_detected:
            # Ограничиваем скорость уклонения от людей
            cmd.angular.z = max(-1.5, min(1.5, self.yolo_repulsive_force))
            cmd.linear.x = -0.2 if emergency_brake_person else 0.3
            status = "AVOIDING PERSON!"
            color = (0, 0, 255)
            
        elif self.min_front_dist < 0.6:
            cmd.angular.z = max(-1.5, min(1.5, self.lidar_repulsive_force * 2.0))
            cmd.linear.x = -0.1 
            status = "CRITICAL WALL!"
            color = (0, 165, 255) 
            
        else:
            # === ОГРАНИЧИТЕЛЬ РУЛЯ ДЛЯ СТАТИКИ ===
            raw_z = target_attractive_force + self.lidar_repulsive_force
            cmd.angular.z = max(-1.2, min(1.2, raw_z)) # Запрещаем крутиться быстрее 1.2 рад/сек
            
            if abs(self.lidar_repulsive_force) > 0.1:
                cmd.linear.x = 0.2
                status = "AVOIDING STATIC!"
                color = (255, 100, 0)
            else:
                cmd.linear.x = 0.8
                status = f"WP {self.wp_index+1}/4: [{self.target_x}, {self.target_y}]"
                color = (0, 255, 0)

        self.publisher.publish(cmd)

        cv2.putText(cv_image, status, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        cv2.putText(cv_image, f"Front Dist: {self.min_front_dist:.2f} m", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Drone FPV, YOLO & Depth", cv_image)
        cv2.waitKey(1)

def main(args=None):
    rclpy.init(args=args)
    node = DroneAvoidanceNode()
    try: rclpy.spin(node)
    except KeyboardInterrupt: pass
    finally: node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
