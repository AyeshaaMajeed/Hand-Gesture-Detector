import subprocess

command = "python gesture_detector.py -l labels.txt -t data.xml"
subprocess.call(command, shell=True)