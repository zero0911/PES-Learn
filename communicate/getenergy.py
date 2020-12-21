import numpy as np

def read_eg(output):
    with open(output, 'r') as f:
        lines = f.readlines()
        for idx, l in enumerate(lines):
            if "energy" in l:
                l_e=lines[idx+1]
                ene = float(l_e.split()[0])
            elif "gradient" in l:
                grad = []
                idx0 = idx + 1
                l_grad = lines[idx0]
                for k in range(idx+1,idx0+3):
                    l_grad = lines[k]
                    l_grad = l_grad.split()[-3:]
                    gi = []
                    for num in l_grad:
                        gi.append(float(num))
                    grad.append(gi)

    return ene, np.asarray(grad)
