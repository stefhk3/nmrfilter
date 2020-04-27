from fabric.api import local, env, run, put, cd, task, lcd, path, sudo, get
from fabric.contrib import project
import pickle
import time
import os
import fnmatch

env.roledefs['m'] = ['jonas@c65']
env.roledefs['b'] = ['jonas@lusitania.millennium.berkeley.edu']
env.roledefs['t'] = ['jonas@titanic.millennium.berkeley.edu']
env.roledefs['mygpu'] = ['ec2-user@ec2-35-162-242-232.us-west-2.compute.amazonaws.com']
env.roledefs['mw'] = ['ericj@midway2-login2.rcc.uchicago.edu']
env.roledefs['l'] = ['ericj@lift.cs.uchicago.edu']



env.forward_agent = True

TGT_DIR = 'respredict'
@task
def deploy(): 
    local('git ls-tree --full-tree --name-only -r HEAD > .git-files-list')

    if 'midway' in env.host:
        tgt_dir = "/home/ericj/nmr-respred"
    if 'lift' in env.host:
        tgt_dir = "/data/ericj/nmr/respredict"
    else:
        tgt_dir = f"/data/jonas/nmr/{TGT_DIR}"
    project.rsync_project(tgt_dir, local_dir="./",
                          exclude=['*.npy', "*.ipynb", 'data'],
                          extra_opts='--files-from=.git-files-list')

    # if 'c65' in env.host:
    #     project.rsync_project("/data/jonas/nmr/respredict/notebooks",
    #                           local_dir=".",
    #                           extra_opts="--include '*.png' --include '*.pdf' --include '*.ipynb'  --include='*/' --exclude='*' " ,

    #                       upload=False)
    


@task
def c65_data_deploy(): 
    for d in [#'data/', 
              #'dataset.named/', #'predict.atomic/', 
            #'graph_conv_many_nuc_pipeline.datasets',
            'processed_data'
              #'notebooks/'

]:
        if 'midway' in env.host:
            tgt_dir = "/home/ericj/nmr-respred"
        if 'lift' in env.host:
            tgt_dir = "/data/ericj/nmr/respredict"
        else:
            tgt_dir = f"/data/jonas/nmr/{TGT_DIR}"

            
        project.rsync_project(f"/data/jonas/nmr/{TGT_DIR}", local_dir="./",
                              #exclude=["*"], # '*.npy', "*.ipynb", 'data'],
                              extra_opts=f"--progress --include '{d}' --include '{d}**' --exclude='*'", 
                              )

@task
def setup_dl_ami():
    def conda_run(x):
        return run("source activate pytorch_p36 && {}".format(x))

    sudo("yum install -q -y dstat emacs htop")
    run("conda upgrade -y conda")
    conda_run("conda install -y jupyter pandas scikit-learn scikit-image seaborn numba")
    conda_run("conda install -y -c rdkit rdkit")

    conda_run("conda install -y sqlalchemy psycopg2")
    conda_run("conda install -y feather-format -c conda-forge")
    conda_run("pip install tqdm ruffus cirpy skorch pubchempy tensorboardX")
    sudo("mkdir -p /data/")
    sudo("chown $SUDO_USER /data")
    run("mkdir -p /data/jonas/nmr/respredict")
    run("mkdir -p /data/jonas/nmr/respredict/data")



@task
def get_checkpoints_to_local(model_filter="", ext='*'):
    if 'lift' in env.host:
        print("USING LIFT REMOTE")
        REMOTE_CHECKPOINT_DIR = "/data/ericj/nmr/respredict/checkpoints/"
    else:
        REMOTE_CHECKPOINT_DIR = "/data/jonas/nmr/respredict/checkpoints/"
    remote_filename = "/tmp/.{}.filelist".format(time.time())
    print("MODEL FITLER IS '{}'".format(model_filter))
    total_glob = f"{model_filter}*.{ext}"
    run(f'find {REMOTE_CHECKPOINT_DIR} -name "{total_glob}" > {remote_filename}', quiet=True)
    print('command done')
    get_res = get(remote_filename)
    ls_str = open(get_res[0], 'r').read()

    files = [f.strip() for f in ls_str.split("\n")]
    print("got list of files")
    #pickle.dump(files, open("files.pickle",'wb'))
    # get the latest checkpoint for each and the meta
    traj_files = files  # [f for f in files if f.endswith(f".{ext}")]

    to_get_files = []
    for m in traj_files:
        if model_filter is not "":
            if not fnmatch.fnmatch(m, "*" + total_glob):
                continue
        to_get_files.append(os.path.basename(m))
    print("getting", len(to_get_files), "files")

    list_of_filenames = ".{}.remote.extfile".format(time.time())
    with open(list_of_filenames, 'w') as fid:
        for f in to_get_files:
            fid.write("{}\n".format(f))
    
    project.rsync_project("/data/jonas/nmr/respredict/checkpoints",
                          local_dir="checkpoints",
                          extra_opts=f'--files-from={list_of_filenames}',
                          upload=False)



@task
def copy_models():

    #REMOTE_DIR = "/data/jonas/nmr/respredict"
    #REMOTE_DIR = "/data/ericj/nmr/respredict"

    exp_name = "bootstrap_13C.cvtest.83070672.0"
    epoch = 1000
    get(f"{REMOTE_DIR}/checkpoints/{exp_name}.meta", local_path="models/default_13C.meta")
    get(f"{REMOTE_DIR}/checkpoints/{exp_name}.{epoch:08d}.model", local_path="models/default_13C.checkpoint")

    exp_name = "bootstrap_1H.cv_0,1.eval2.82988407"
    epoch = 700
    get(f"{REMOTE_DIR}/checkpoints/{exp_name}.meta", local_path="models/default_1H.meta")
    get(f"{REMOTE_DIR}/checkpoints/{exp_name}.{epoch:08d}.model", local_path="models/default_1H.checkpoint")


@task
def build_docker_pred():
    local("docker build -t nmr-respredict:beta -f Docker/Dockerfile .")

@task
def build_docker_push():
    """
    push to dockerhub
    """
    

    local("docker tag nmr-respredict:beta jonaslab/nmr-respredict:beta")
    local("docker push jonaslab/nmr-respredict:beta")


