from streamlit.testing.v1 import AppTest
import os
import sys

# Add the main directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))

def test_app_boot():
    at = AppTest.from_file("main/app.py").run(timeout=10)
    assert not at.exception
