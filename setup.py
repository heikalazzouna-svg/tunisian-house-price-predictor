from setuptools import setup

setup(
    name="tunisian-house-price-predictor",
    version="0.1.0",
    install_requires=[
        "streamlit==1.38.0",
        "pandas==2.2.3", 
        "numpy==2.1.0",
        "scikit-learn==1.5.2",
        "joblib==1.4.2",
    ],
)