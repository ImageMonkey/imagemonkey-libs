import os
import sys
import shutil
import subprocess

def rmDirectories():
	currentDir = os.path.dirname(os.path.realpath(__file__))
	try:
		shutil.rmtree(currentDir + os.path.sep + "dist")
	except FileNotFoundError:
		pass

	try:
		shutil.rmtree(currentDir + os.path.sep + "pyimagemonkey.egg-info")
	except FileNotFoundError:
		pass

def createSourceDist(clear=True):
	if clear:
		rmDirectories()
	currentDir = os.path.dirname(os.path.realpath(__file__))
	retVal = subprocess.call("python setup.py sdist", shell=True)
	if retVal == 1:
		return False
	return True

def publish(test=True):
	args = ""
	if test:
		args = "--repository-url https://test.pypi.org/legacy/ "
	cmd = "twine upload " + args + "dist/*"
	retVal = subprocess.call(cmd, shell=True)
	if retVal == 1:
		return False
	return True
	

if __name__ == "__main__":
	print("Publish pyimagemonkey to (Test-) PyPI")
	print("1. Publish to Test-PyPI")
	print("2. Publish to PyPI")
	print("\n\n")
	selection = input("What do you want to do? ")
	if selection == "1":
		if createSourceDist(clear=True):
			retVal = publish(test=True)
			rmDirectories()
			if retVal:
				print("The package is now available at: https://test.pypi.org/project/pyimagemonkey/")
				print("You can install it with: pip3 install --no-cache-dir --index-url https://test.pypi.org/simple/ pyimagemonkey")

	elif selection == "2":
		verify = input("Do you really want to push to PRODUCTION PyPI? (yes/no) ")
		if verify == "yes":
			if createSourceDist(clear=True):
				publish(test=False)
				rmDirectories()
		else:
			print("Invalid input")
			sys.exit(1)
	else:
		print("Unexcepted input %s" %(selection,))
		sys.exit(1)
