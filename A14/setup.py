"""
Setup script for MediaPipe Pose Estimator module.
"""

from setuptools import setup, find_packages

setup(
    name="mediapipe-pose-estimator",
    version="1.0.0",
    description="A MediaPipe-based human pose estimator with 2D landmark detection",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/yourusername/mediapipe-pose-estimator",
    packages=find_packages(),
    install_requires=[
        "opencv-python>=4.5.0",
        "numpy>=1.19.0",
        "mediapipe>=0.8.0"
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
    python_requires=">=3.7",
)
