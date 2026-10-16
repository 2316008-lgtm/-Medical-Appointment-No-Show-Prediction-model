"""
Root-level export for AppointmentPredictor
Project: 31. ABISHEK M - Medical Appointment No-Show Prediction
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from src.predict import AppointmentPredictor

__all__ = ["AppointmentPredictor"]
