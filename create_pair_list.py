import os
import numpy as np
from configs import configuration

if __name__ == "__main__":
    
    cfg = configuration()

    train_dir = os.listdir(cfg.source_root)
    test_dir = os.listdir(cfg.target_root)

    pair_list = []
    for d in train_dir:
        images_tr = os.listdir(os.path.join(cfg.source_root, d))
        for f in test_dir:
            images_ts = os.listdir(os.path.join(cfg.target_root, f))
            for i in images_tr:
                i = "/" + d + "/" + i
                for j in images_ts:
                    j = "/" + f + "/" + j
                    pair_list.append(i + " " + j + " " + str(int(d==f)))
    
    with open("test_pair_list.txt", 'w') as handler:
        handler.writelines("%s\n" % pair for pair in pair_list)

    print("Success.")