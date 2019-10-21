import os
import subprocess

cmd = '/Users/akica/PyCharmProjects/AIproject'
assert os.path.isdir(cmd)
os.chdir(cmd)
process = subprocess.call('\"C:/Program Files/swipl/bin/swipl\" -f prolog_code.pl <query.txt > output.txt', shell=True)
f = open("output.txt", "r")
risk = f.read()
print(risk)