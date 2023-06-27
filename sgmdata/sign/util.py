import requests
import os
import pickle
import msgpack
from sgmdata import sign, config
import urllib3

urllib3.disable_warnings(urllib3.exceptions.SecurityWarning)

KEY_STORE = os.path.expanduser(config.get('user_keys', "~/.ssh/sgmlive_keys"))
SGMLIVE_URL = config.get('sgmlive_url', "https://sgmbeta.lightsource.ca")
ENDSTATION = config.get('endstation', 'SGM-AMBXAS')
CERT_KEY = config.get('sgmlive_cert')

Resp = requests.Response
Sign = sign.Signer


def wrapresp(resp: Resp, type='list') -> object:
    if resp.status_code == 200:
        return resp.json()
    elif type == 'list':
        return []
    elif type == 'dict':
        return {}
    else:
        return None


def post(url: str, data=None, type='list'):
    if CERT_KEY:
        if data:
            resp = requests.post(url, data=data, verify=CERT_KEY)
        else:
            resp = requests.post(url, verify=CERT_KEY)
    else:
        if data:
            resp = requests.post(url, data=data)
        else:
            resp = requests.post(url)
    return wrapresp(resp, type=type)


def get(url: str, type='list'):
    if CERT_KEY:
        resp = requests.get(url, verify=CERT_KEY)
    else:
        resp = requests.get(url)
    return wrapresp(resp, type=type)


def purge_key(user: str) -> bool:
    if os.path.exists(KEY_STORE):
        with open(KEY_STORE, 'rb') as f:
            key_store = pickle.load(f)
        if user in key_store.keys():
            del key_store[user]
            with open(KEY_STORE, 'wb') as f:
                pickle.dump(key_store, f)
        return True
    else:
        return False


def get_or_make_key(user: str) -> Sign:
    """
    Convenience function for grabbing SSH key from store, or registering a new key
    in SGMxLive and storing it for later use.

    :param user: (str)
    """
    from cryptography.hazmat.backends import default_backend
    from cryptography.hazmat.primitives.asymmetric import dsa
    from cryptography.hazmat.primitives import serialization
    post = True
    if not os.path.exists(KEY_STORE):
        if not os.path.exists(os.path.dirname(KEY_STORE)):
            os.makedirs(os.path.dirname(KEY_STORE))
        key_store = {}
        with open(KEY_STORE, 'wb') as f:
            pickle.dump(key_store, f)
    else:
        with open(KEY_STORE, 'rb') as f:
            key_store = pickle.load(f)
    if not user in key_store.keys():
        key = dsa.generate_private_key(key_size=1024, backend=default_backend())
        data = {
            'private': key.private_bytes(
                serialization.Encoding.DER,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption()
            ),
            'public': key.public_key().public_bytes(
                serialization.Encoding.OpenSSH,
                serialization.PublicFormat.OpenSSH
            )
        }
        key_store.update({user: data})
        with open(KEY_STORE, 'wb') as f:
            pickle.dump(key_store, f)
    else:
        data = key_store[user]
        post = False
    signer = sign.Signer(**data)
    signature = signer.sign(user)
    if post:
        if CERT_KEY:
            resp = requests.post(f'{SGMLIVE_URL}/api/v2/{signature}/project/', data={"public": data['public']},
                                 verify=CERT_KEY)
        else:
            resp = requests.post(f'{SGMLIVE_URL}/api/v2/{signature}/project/', data={"public": data['public']})
        if resp.status_code != 200:
            raise requests.HTTPError("404")
    return signer


def get_proposals(user: str, signer: Sign) -> list:
    """
    Description: List of proposal numbers associated with supplied user.

    :param user: (str)
    :param signer: (sign.Signer())

    returns: [(str)] i.e. ['35G5000', '33G00001']
    """
    signature = signer.sign(user)
    return get(f'{SGMLIVE_URL}/api/v2/{signature}/proposal/')


def get_shipment(user: str, signer: Sign, proposal: str) -> list:
    """
    Description: Get shipments associated with user proposal.

    :param user: (str)
    :param signer: (sign.Signer())
    :param proposal: (str) e.g. 35G10000

    returns: [dict(shipment)]
    """
    signature = signer.sign(user)
    return get(f'{SGMLIVE_URL}/api/v2/sgm/{signature}/shipments/{proposal}')


def get_shipment_samples(user: str, signer: Sign, proposal: str, pk: int) -> list:
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param proposal: (str) e.g. 35G10000
    :param pk: (int) e.g. Shipment primary key.

    returns [dict(sample)]
    """
    data = msgpack.packb({"id": pk})
    signature = signer.sign(user)
    return post(f'{SGMLIVE_URL}/api/v2/sgm/{signature}/shipments/{proposal}/', data=data)


def launch_session(user: str, signer: Sign, proposal: str) -> dict:
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param proposal: (str) e.g. 35G10000
    """
    import string
    from datetime import date
    import numpy
    signature = signer.sign(user, )
    today = date.today()
    date_string = today.strftime('%Y%m%d')
    token = ''.join(numpy.random.choice(list(string.digits + string.ascii_letters), size=8))
    session_key = '{}-{}'.format(date_string, token)
    return post(f'{SGMLIVE_URL}/api/v2/{signature}/launch/{ENDSTATION}/{session_key}/{proposal}/')


def close_session(user: str, signer: Sign, session_key: str) -> dict:
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param session_key: (str) e.g. EXP-20210512-asldfj
    """
    signature = signer.sign(user, )
    return post(f'{SGMLIVE_URL}/api/v2/{signature}/close/{ENDSTATION}/{session_key}/')


def add_data(user: str, signer: Sign, data_dict: dict) -> dict:
    # Add Data
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param data_dict:
         {
        "username": user,
        "directory": '/prj37G0000/',
        "energy": 0.290,
        "type": 'XAS',
        "exposure": 60.0,
        "attenuation": 20.0,
        "beam_size": 50.0,
        "name": "K10-EEMs",
        "filename": '',
        "beamline": 'SGM-AMBXAS',
        "frames": ['2020-03-07t16-58-33-0600', '2020-03-07t16-59-52-0600'],
        "sample_id": 5,
        "start_time": "2020-03-07t16-58-33-0600",
        "end_time": "2020-03-08t01-39-45-0600",
        "proposal": "prj37G0000"
    }
    :return: <resp : >
    """
    data = msgpack.packb(data_dict)
    signature = signer.sign(user)
    return post(f'{SGMLIVE_URL}/api/v2/{signature}/data/{ENDSTATION}/', data=data)


def add_holder(user: str, signer: Sign, data_dict: dict) -> dict:
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param data_dict:
         {
         (optional) "id": 1,  #for updating pre-existing holder.
        "title": 'Holder A - a9eadf9',
        "comments": str,
        "kind": 'Plate',
        "image": Bokeh.embed.json_item,
        "samples": dict, e.g. {sample_name: [[x, y], [x1, y1]}
    }
    :return: <resp : >
    """
    data = msgpack.packb(data_dict)
    signature = signer.sign(user)
    return post(f'{SGMLIVE_URL}/api/v2/sgm/{signature}/holder/{ENDSTATION}/', data=data)


def add_report(user: str, signer: Sign, data_dict: dict):
    # Add Report
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param data_dict:
         {
        "username": user,
        "directory": '/prj37G0000/',
        "energy": 0.290,
        "type": 'XAS',
        "exposure": 60.0,
        "attenuation": 20.0,
        "beam_size": 50.0,
        "name": "K10-EEMs",
        "filename": '',
        "beamline": 'SGM-AMBXAS',
        "frames": ['2020-03-07t16-58-33-0600', '2020-03-07t16-59-52-0600'],
        "sample_id": 5,
        "start_time": "2020-03-07t16-58-33-0600",
        "end_time": "2020-03-08t01-39-45-0600",
        "proposal": "prj37G0000"
    }
    :return: <resp : >
    """
    data = msgpack.packb(data_dict)
    signature = signer.sign(user)
    return post(f'{SGMLIVE_URL}/api/v2/{signature}/report/{ENDSTATION}/', data=data)


def current_samples(user: str, signer: Sign) -> list:
    signature = signer.sign(user)
    return get(f'{SGMLIVE_URL}/api/v2/{signature}/samples/{ENDSTATION}/')


def automount(user: str, signer: Sign, session_key: str, data_dict=None) -> dict:
    data = msgpack.packb(data_dict)
    signature = signer.sign(user)
    return post(f'{SGMLIVE_URL}/api/v2/sgm/{signature}/automount/{ENDSTATION}/{session_key}/',
                data=data)


def get_container(user: str, signer: Sign, session_key: str):
    signature = signer.sign(user)
    return get(f'{SGMLIVE_URL}/api/v2/sgm/{signature}/automount/{ENDSTATION}/{session_key}/')

def find(signature: str, proposal: str, type: str, **kwargs) -> list:
    querystr = "?" + "&".join([f"{k}={v}" for k, v in kwargs.items() if v or isinstance(v, bool)])
    if querystr:
        l = get(f'{SGMLIVE_URL}/api/v2/{signature}/proposal-{type}/{proposal}/{querystr}')
    else:
        l = get(f'{SGMLIVE_URL}/api/v2/{signature}/proposal-{type}/{proposal}/')
    return l


def find_samples(user: str, signer: Sign, proposal: str, **kwargs) -> list:
    """
    Convenience function for finding datasets
    :param user: (str)
    :param signer: (sign.Signer())
    :param proposal: (str) e.g. 35G10000
    :param sample_name: (str:optional) e.g. "Sample123"

    returns [dict(sample)]
    """
    signature = signer.sign(user)
    return find(signature, proposal, 'sample', **kwargs)


def find_data(user: str, signer: Sign, proposal: str, **kwargs) -> list:
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param proposal: (str) e.g. 35G10000
    :param sample: (int) e.g. Data primary key.
    :param kind: (str:optional) e.g. XAS

    returns [dict(dataset)]
    """
    signature = signer.sign(user)
    return find(signature, proposal, 'data', **kwargs)


def find_report(user: str, signer: Sign, proposal: str, **kwargs) -> list:
    """
    :param user: (str)
    :param signer: (sign.Signer())
    :param proposal: (str) e.g. 35G10000
    :param sample_id: (int) e.g. Sample primary key.
    :param kind: (str:optional) e.g. XAS

    returns [dict(dataset)]
    """
    signature = signer.sign(user)
    return find(signature, proposal, 'report', **kwargs)

    
def get_or_add_sample(user: str, signer: Sign, sample: str, session_key: str, data_dict=None, proposal=None):
    items = []
    if data_dict:
        container_id = data_dict.get("container_id", None)
        l = current_samples(user, signer)
        items = [s for s in l if sample in s['name']]
        if not items and proposal:
            items = find_samples(user, signer, proposal, sample_name=sample, collect=False)
            if items and container_id:
                items = [el for el in items if el['container_id'] == container_id]

    elif proposal:
        items = find_samples(user, signer, proposal, sample_name=sample, collect=False)

    if items:
        items_exact = [i for i in items if sample == i]
        items_in = [i for i in items if sample in i]
        if len(items_exact):
            return items_exact[0]
        elif len(items_in):
            return items_in[0]
        return items[0]
    else:
        if not data_dict:
            data_dict = {"name": sample}
        data = msgpack.packb(data_dict)
        signature = signer.sign(user)
        return post(f'{SGMLIVE_URL}/api/v2/sgm/{signature}/samples/{ENDSTATION}/{session_key}/',
                    data=data)
