import os
import argparse
from peft_conf import CONF_DIR


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host_base', default="", type=str, required=True)
    parser.add_argument('--ak', default="", type=str, required=True)
    parser.add_argument('--sk', default="", type=str, required=True)
    args = parser.parse_args()

    host_base, ak, sk = args.host_base, args.ak, args.sk
    conf_sv_path = os.path.join(CONF_DIR, "petreloss.conf")
    with open(conf_sv_path, "w") as f:
        f.write("[DEFAULT]\n")
        f.write("enable_mc = False\n")
        f.write("default_cluster = finetune\n\n")

        f.write("console_log_level = ERROR\n")
        f.write("file_log_level = ERROR\n\n")
        f.write("[mc]\n")
        f.write("mc_key_cb = sha512\n\n")

        f.write("[finetune]\n")
        f.write(f"access_key = {ak}\n")
        f.write(f"secret_key = {sk}\n")
        f.write(f"host_base = {host_base}\n")
    print(conf_sv_path)
