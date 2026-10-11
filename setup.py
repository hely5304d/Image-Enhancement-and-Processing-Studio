from setuptools import setup, find_packages

setup(
    name="image-enhancement-studio",
    version="1.0.0",
    author="Hely",
    description="Studio-grade Image Enhancement & Computer Vision Engine powered by Python, OpenCV, NumPy, and Pillow",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.24.0",
        "opencv-python>=4.8.0",
        "pillow>=10.0.0",
        "Flask>=3.0.0",
    ],
    entry_points={
        "console_scripts": [
            "image-studio=cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Multimedia :: Graphics",
        "Topic :: Scientific/Engineering :: Image Processing",
    ],
)
