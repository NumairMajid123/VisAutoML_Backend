from explainerdashboard import ExplainerDashboard
import sys
import joblib
import os

def runModel(model_identifier, port):
    """
    Launches the explainer dashboard for the given model on the specified port.
    It assumes that the joblib file is named 'explainer_<model_identifier>.joblib'.
    """
    joblib_file = f"explainer_{model_identifier}.joblib"
    os.system("explainerdashboard run " + joblib_file + " --port=" + str(port))

    # os.system('explainerdashboard run '+filename+'.joblib')
