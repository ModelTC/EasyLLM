from llm.utils.tools.petrel_helper import PetrelHelper
import os
import argparse


# dir = "toolchain_ft:s3://finetune/lora_it_chineselamma7b_4e_tokenlmhead/"


def download(dir, flag_sub_dir=False):
    local_folder = dir
    if "s3://" in dir:
        spt = dir.split('/')
        if spt[-1] != '':
            local_folder = spt[-1]
            dir += '/'
        else:
            local_folder = spt[-2]
        files = PetrelHelper.list_dir(dir, sub_dir=flag_sub_dir)
        new_files = []
        for item in files:
            if item.endswith('.bin') or item.endswith('.pt') or item.endswith(".safetensors"):
                continue
            new_files.append(item)
        if os.path.isdir(local_folder):
            pass
        else:
            os.makedirs(local_folder, exist_ok=True)
            for file in new_files:
                # if file type is dir
                if file.endswith('/'):
                    ceph_sub_dir = os.path.join(dir, file)
                    sub_dir = os.path.join(local_folder, file)
                    os.makedirs(sub_dir, exist_ok=True)
                    sub_files = PetrelHelper.list_dir(ceph_sub_dir)
                    for sub_file in sub_files:
                        if sub_file.endswith(".bin") or sub_file.endswith(".pt"):
                            continue
                        res = PetrelHelper._petrel_helper.load_data(ceph_sub_dir + sub_file, ceph_read=False)
                        with open(os.path.join(sub_dir, sub_file), "wb") as fw:
                            fw.write(res)
                    continue

                res = PetrelHelper._petrel_helper.load_data(dir + file, ceph_read=False)
                with open(os.path.join(local_folder, file), "wb") as fw:
                    fw.write(res)
    with open('local_folder.txt', 'w') as f:
        print(local_folder, file=f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ceph_dir', type=str, default='default_value')
    parser.add_argument('--sub_dir', type=bool, default=False)
    args = parser.parse_args()

    download(args.ceph_dir, args.sub_dir)
