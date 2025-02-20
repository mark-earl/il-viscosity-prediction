from streamlit.testing.v1 import AppTest
import os
import sys

# Add the main directory to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../main")))

def test_both():
    at = AppTest.from_file("main/app.py").run()
    at.radio(key="Feature Selection Method").set_value("Use Feature Preset")
    at.selectbox(key="Select Feature Set").set_value("Both")
    assert at.selectbox(key="Select Feature Set").value == "Both"
    assert not at.exception

def test_functional_groups():
    at = AppTest.from_file("main/app.py").run()
    at.radio(key="Feature Selection Method").set_value("Use Feature Preset")
    at.selectbox(key="Select Feature Set").set_value("Functional Groups")
    assert at.selectbox(key="Select Feature Set").value == "Functional Groups"
    assert not at.exception

def test_molecular_descriptors():
    at = AppTest.from_file("main/app.py").run()
    at.radio(key="Feature Selection Method").set_value("Use Feature Preset")
    at.selectbox(key="Select Feature Set").set_value("Molecular Descriptors")
    assert at.selectbox(key="Select Feature Set").value == "Molecular Descriptors"
    assert not at.exception
