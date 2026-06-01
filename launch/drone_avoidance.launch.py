import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import ExecuteProcess
from launch_ros.actions import Node  # <--- ИСПРАВЛЕННЫЙ ИМПОРТ

def generate_launch_description():
    pkg_share = get_package_share_directory('drone_avoidance')
    urdf_path = os.path.join(pkg_share, 'urdf', 'flying_drone.urdf')
    world_path = os.path.join(pkg_share, 'worlds', 'indoor_actors.world') 

    return LaunchDescription([
        # Запуск сервера Gazebo
        ExecuteProcess(
            cmd=['gzserver', world_path, '-s', 'libgazebo_ros_init.so', '-s', 'libgazebo_ros_factory.so', '--verbose'],
            output='screen'
        ),
        # Запуск клиента Gazebo (окно)
        ExecuteProcess(
            cmd=['gzclient'],
            output='screen'
        ),
        # Публикатор состояния робота
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            output='screen',
            arguments=[urdf_path]
        ),
      # Спавн дрона ровно в центре комнаты на высоте 1.5м
        Node(
            package='gazebo_ros',
            executable='spawn_entity.py',
            arguments=['-entity', 'drone', '-file', urdf_path, '-x', '0.0', '-y', '-7.0', '-z', '0.8'],
            output='screen'
        ),
        # Нода компьютерного зрения
        Node(
            package='drone_avoidance',
            executable='avoid_node',
            output='screen'
        )
    ])
