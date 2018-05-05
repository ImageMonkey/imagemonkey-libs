import tarfile
import os

def is_directory(path):
	return os.path.isdir(path)

def directory_exists(path):
	return (os.path.exists(path) and is_directory(path))

def extract_tar_gz(input_file, output_path):
	#if not is_directory(output_path):
	#	raise ImageMonkeyGeneralError("%s is not a directory" %(output_path,)) 

	tar = tarfile.open(input_file, "r:gz")
	tar.extractall(path=output_path)
	tar.close()