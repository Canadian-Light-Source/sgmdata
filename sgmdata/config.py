import os
import sys
import logging

cfg = {
    'version': u'0.3.1b',
    'db_port_5432_tcp_addr': 'vsrv-sgm-data-01.clsi.ca',
    'db_port_5432_tcp_port': '5080',
    'db_env_db': 'postgres',
    'db_env_postgres_user': 'sgmpostgres',
    'h5endpoint': "https://vsrv-sgm-hdf5-01.clsi.ca:6050",
    'h5user': 'admin',
    'debug': logging.INFO,
}



def get(x, sub=None):
    # see if there is a command-line override
    option = '--' + x + '='
    for i in range(1, len(sys.argv)):
        # print i, sys.argv[i]
        if sys.argv[i].startswith(option):
            # found an override
            arg = sys.argv[i]
            return arg[len(option):]  # return text after option string
    # see if there are an environment variable override
    if x.upper() in os.environ:
        return os.environ[x.upper()]
    if os.path.exists('/run/secrets/' + x):
        with open('/run/secrets/' + x, 'r') as file:
            data = file.read().replace('\n', '')
        return data
    # no command line override, just return the cfg value
    if x in cfg:
        return cfg[x]
    else:
        return sub
