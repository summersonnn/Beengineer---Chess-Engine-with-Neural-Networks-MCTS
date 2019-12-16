import os


def find_last_edited_file(PATH_TO_DIRECTORY):
	early = 0
	last_trained = None
	for file in os.listdir(PATH_TO_DIRECTORY):
		filename = os.path.join(PATH_TO_DIRECTORY, file)
		statbuf = os.stat(filename)
		last_edited = statbuf.st_mtime
		if last_edited > early:
			early = last_edited
			last_trained = filename
	return last_trained
