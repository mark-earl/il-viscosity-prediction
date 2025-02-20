from streamlit.testing.v1 import AppTest
import os
import sys

# Add the main directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))

def test_cation_charge():
    at = AppTest.from_file("main/app.py").run()
    at.radio(key="Feature Selection Method").set_value("Use Feature Preset")
    at.selectbox(key="Select Feature Set").set_value("Both")
    assert at.selectbox(key="Select Feature Set").value == "Both"
    at.selectbox(key="Select Target Variable").set_value("cation_Charge")
    assert at.selectbox(key="Select Target Variable").value == "cation_Charge"
    assert not at.exception
