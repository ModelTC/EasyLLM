import os
import io
import torch
import json
import configparser
import pickle as pk


class PetrelOpen(object):
    def __init__(self, filename, **kwargs):
        self.handle = PetrelHelper._petrel_helper.load_data(filename, **kwargs)

    def __enter__(self):
        return self.handle

    def __exit__(self, exc_type, exc_value, exc_trackback):
        del self.handle


class PetrelHelper(object):

    _petrel_helper = None
    open = PetrelOpen

    default_conf_path = os.environ.get('PETRELPATH', '~/petreloss.conf')

    def __init__(self, conf_path=default_conf_path):
        self.conf_path = conf_path

        self._inited = False

        self._init_petrel()

        PetrelHelper._petrel_helper = self

    def _init_petrel(self):
        try:
            from petrel_client.client import Client
            self.client = Client(self.conf_path)

            self._inited = True
        except Exception as e:
            print(e)
            print('init petrel failed')

    def check_init(self):
        if not self._inited:
            raise Exception('petrel oss not inited')

    def _iter_cpeh_lines(self, path):
        response = self.client.get(path, enable_stream=True, no_cache=True)

        for line in response.iter_lines():
            cur_line = line.decode('utf-8')
            yield cur_line

    def load_data(self, path, ceph_read=True, fs_read=False, mode='r'):
        if 's3://' not in path:
            if not fs_read:
                return open(path, mode)
            else:
                return open(path, mode).read()
        else:
            self.check_init()

            if ceph_read:
                return self._iter_cpeh_lines(path)
            else:
                return self.client.get(path)

    @staticmethod
    def list_dir(dir, sub_dir=False):
        files = []
        contents = PetrelHelper._petrel_helper.client.list(dir)
        for content in contents:
            if not sub_dir and content.endswith('/'):
                continue
            files.append(content)
        return files

    @staticmethod
    def load_pk(path, mode='r'):
        if 's3://' not in path:
            pk_res = pk.load(open(path, mode))
        else:
            pk_res = pk.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        return pk_res

    @staticmethod
    def load_json(path, mode='r'):
        if 's3://' not in path:
            js = json.load(open(path, mode))
        else:
            js = json.loads(PetrelHelper._petrel_helper.load_data(path, ceph_read=False))
        return js

    @staticmethod
    def write(res, path, mode='w'):
        if 's3://' not in path:
            with open(path, mode=mode, encoding="utf-8") as f:
                f.write(res)
        else:
            PetrelHelper._petrel_helper.client.put(path, res)

    def load_pretrain(self, path, map_location=None):
        if 's3://' not in path:
            assert os.path.exists(path), f'No such file: {path}'
            return torch.load(path, map_location=map_location)
        elif 'http://' in path:
            return torch.hub.load_state_dict_from_url(path, map_location=map_location)
        else:
            self.check_init()

            file_bytes = self.client.get(path)
            buffer = io.BytesIO(file_bytes)
            res = torch.load(buffer, map_location=map_location)
            return res

    @staticmethod
    def rm(path):
        if 's3://' not in path:
            os.system(f'rm {path}')
        else:
            PetrelHelper._petrel_helper.client.delete(path)

    @staticmethod
    def load(path, **kwargs):
        if '.ini' in path:
            path = path[:-4]
        if not os.path.exists(path) and os.path.exists(path + '.ini'):
            # get realpath
            conf = configparser.ConfigParser()
            conf.read(path + '.ini')
            path = conf['Link']['ceph']
        return PetrelHelper._petrel_helper.load_pretrain(path, **kwargs)

    def save_checkpoint(self, model, path):
        if 's3://' not in path:
            torch.save(model, path)
        else:
            with io.BytesIO() as f:
                torch.save(model, f)
                f.seek(0)
                self.client.put(path, f)

    @staticmethod
    def get_stream(path):
        if "s3://" in path:
            stream = io.BytesIO(PetrelHelper._petrel_helper.client.get(path))
        else:
            stream = open(path, "rb")
        return stream

    @staticmethod
    def exists(path):
        if "s3://" in path:
            return PetrelHelper._petrel_helper.client.contains(path)
        else:
            return os.path.exists(path)

    @staticmethod
    def save(model, path, ceph_path=None):
        if ceph_path:
            # save link
            lustre_path = os.path.abspath(path)

            link_path = path + '.ini'
            config = configparser.ConfigParser()
            config.add_section("Link")
            config.set("Link", "path", path)
            config.set("Link", "lustre", lustre_path)
            config.set("Link", "ceph", ceph_path)

            # save model to ceph_path
            ret = PetrelHelper._petrel_helper.save_checkpoint(model, ceph_path)

            # save model before saving ini
            config.write(open(link_path, "w"))
            return ret
        else:
            return PetrelHelper._petrel_helper.save_checkpoint(model, path)


__petrel_helper = PetrelHelper()
