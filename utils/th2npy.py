import glob

import numpy as np
from torch.utils.serialization import load_lua

paths = glob.glob("./th-mnist/train_*.t7") + glob.glob("./th-mnist/test_*.t7") + glob.glob("./th-mnist/val_*.t7")
for path in paths:
    print(path)
    images = []
    test = load_lua(path)
    data = test[0]
    labels = test[1]
    targets = []
    for i in range(len(data)):
        im = (data[i] * 255).byte()
        images.append(im.numpy())
        targets.append(labels[i].max(0)[1][0])
    targets = np.array(targets)
    images = np.array(images)
    np.save(path.replace("t7", "npy").replace("th-mnist", "npy-mnist"), {'images': images, 'labels': targets})
