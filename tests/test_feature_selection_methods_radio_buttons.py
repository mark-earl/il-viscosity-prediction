from streamlit.testing.v1 import AppTest
import os
import sys

# Add the main directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))

def test_use_feature_prest():
    at = AppTest.from_file("main/app.py").run(timeout=10)
    at.radio(key="Feature Selection Method").set_value("Use Feature Preset")
    assert at.radio(key="Feature Selection Method").value == "Use Feature Preset"
    assert not at.exception

def test_use_feature_importance_file():
    at = AppTest.from_file("main/app.py").run(timeout=10)
    at.radio(key="Feature Selection Method").set_value("Use Feature Importance File")
    assert at.radio(key="Feature Selection Method").value == "Use Feature Importance File"
    assert not at.exception

def test_manually_select_features():
    at = AppTest.from_file("main/app.py").run(timeout=10)
    at.radio(key="Feature Selection Method").set_value("Manually Select Features")
    assert at.radio(key="Feature Selection Method").value == "Manually Select Features"
    assert not at.exception
