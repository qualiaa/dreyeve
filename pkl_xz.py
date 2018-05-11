import subprocess as proc
import pickle as pkl

def load(input_path):
    with open(input_path, "rb") as f:
        result = proc.run(["xz", "--decompress"], stdin=f, stdout=proc.PIPE)
        if result.returncode != 0:
            raise ChildProcessError(result.returncode)
        return pkl.loads(result.stdout)

def save(data, output_path):
    with open(output_path, "wb") as datafile:
        pickle = pkl.dumps(data)
        result = proc.run(["xz"], input=pickle, stdout=datafile)
        if result.returncode != 0:
            raise ChildProcessError(result.returncode)

def pkl_to_xz(file_path):
    with open(file_path, "rb") as in_file:
        with open(file_path + ".xz", "wb") as out_file:
            proc.run(["xz"], stdin=in_file, stdout=out_file)
