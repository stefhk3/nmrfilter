"""
Cross-validation tools to create configs with different
cv settings
"""


"""
Take in a yaml and create a full set of CV subsets
"""

import os
from ruamel.yaml import YAML
import copy
import click
import signal
import subprocess
import time

def replace_cv_split_test(config, cv_set):
    c = copy.deepcopy(config)
    c['exp_data']['cv_split']['test'] = cv_set
    return c

@click.group()
def cli():
    pass


@cli.command()
@click.argument('exp_config_filename')
@click.argument('outdir')
def create_cv_sets(exp_config_filename, outdir):
    
    exp_config_basename = os.path.basename(exp_config_filename)
    
    yaml = YAML()

    config = yaml.load(open(exp_config_filename, 'r'))

    ### Assume CV
    cv_sets = [(2*i, 2*i+1) for i in range(5)]

    for cv_set in cv_sets:
        new_yaml = replace_cv_split_test(config, cv_set)

        cv_str = "cv_" + ",".join([str(s) for s in cv_set])

        source_filename, _ = os.path.splitext(exp_config_basename)
        out_filename = os.path.join(outdir, f"{source_filename}.{cv_str}.yaml")
                                       

        print('writing', out_filename)
        with open(out_filename, 'w') as fid:

            yaml.dump(new_yaml, fid)
            
@cli.command()
@click.argument('config_yaml')
@click.argument('expset_name')
def run_cv_set(config_yaml, expset_name):
    """
    This could be done with a shell script but I always regret actually writing shell
    scripts
    """

    # create a process group and become its leader
    #os.setpgrp()

    exp_name_from_file = os.path.basename(config_yaml.replace(".cv.yaml", ""))

    expset_name_full = "{}.{}.{:08d}".format(exp_name_from_file, expset_name,int(time.time() % 1e8))

    processes = []
    try:
        yaml = YAML()

        config = yaml.load(open(config_yaml, 'r'))


        base_exp_config_filename = config['baseconfig']
        base_exp_config_basename = os.path.basename(base_exp_config_filename)

        cv_sets = config['cv_sets']
        exp_config = yaml.load(open(base_exp_config_filename, 'r'))
        

        for cv_set_i, cv_set in enumerate(cv_sets):
            test_subsets = cv_set['subsets']

                        
            GPU = cv_set['GPU']

            new_env = dict(os.environ)
            new_env['CUDA_VISIBLE_DEVICES'] = str(GPU)
            

            ## create new confg
            new_exp_config = replace_cv_split_test(exp_config, test_subsets)

            cv_exp_name = f"{expset_name_full}.{cv_set_i}"

            ## write to filename
            cv_exp_config_filename = os.path.join("/tmp", f"{cv_exp_name}.yaml")
            cv_exp_log_filename = os.path.join("/tmp", f"{cv_exp_name}.log")
            with open(cv_exp_config_filename, 'w') as fid:
                yaml.dump(new_exp_config, fid)
                            

            ##
            cmdstr = f"python respredict_train.py {cv_exp_config_filename} --skip-timestamp"

            
            print(cmdstr)
            #cmdstr = "ls; sleep 60"
            log_fid = open(cv_exp_log_filename, 'w')
            proc = subprocess.Popen(cmdstr,
                                    shell = True,
                                    env = new_env, 
                                    stdout = log_fid)

            processes.append(proc)
            
            print(cv_set_i, cv_exp_config_filename)

        
        for p in processes:
            WAIT_TIMEOUT_SEC = 5
            try:
                p.wait(WAIT_TIMEOUT_SEC)
            except subprocess.TimeoutExpired as e:
                continue
            
    finally:
        
        for pi, p in enumerate(processes):
            if p.poll() is not None:
                print("terminating", pi)
                p.kill()
        for pi, p in enumerate(processes):
            p.wait()
        # kill everything in my group
        #os.killpg(0, signal.SIGKILL)
            
if __name__ == "__main__":
    cli()
