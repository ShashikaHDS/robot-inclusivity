from setuptools import find_packages, setup

package_name = 'pcd_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='aryan',
    maintainer_email='aryan@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pcd_publisher = pcd_package.pcd_publisher:main',
            'pcd_publisher_1 = pcd_package.improved_publisher:main',
            'keyframe_replay_to_octomap = pcd_package.keyframe_replay_to_octomap:main',
            'pcd_to_occupancy_map = pcd_package.pcd_to_occupancy_map:main',
        ],
    },
)
