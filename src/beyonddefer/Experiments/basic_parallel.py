# import numpy as np
import os
import pickle


class experiment_parallel():
    def __init__(self, for_loop, init, conc, iter, filename):
        self.init = init
        self.for_loop = for_loop
        self.conc = conc
        self.iter = iter
        self.filename = filename
        self.result = None

    def run(self, parallel=False, iter=0):
        if not parallel:
            res = []
            res_init = self.init()
            for i in range(self.iter):
                res.append(self.for_loop(res_init, i))
            self.result = self.conc(self, res)
        else:
            if iter < self.iter:
                res_init = self.init()
                res = self.for_loop(res_init, iter)
                save_res(res, self.filename + str(iter) + ".pkl")
            elif iter == self.iter:
                while True:
                    for i in range(self.iter):
                        j = (i//4)*10+(i % 4)
                        if not os.path.exists(self.filename + str(j) + ".pkl"):
                            break
                    else:
                        break
                res = []
                for i in range(self.iter):
                    j = (i//4)*10+(i % 4)
                    with open(self.filename + str(j) + ".pkl", 'rb') as f:
                        loaded_data = pickle.load(f)
                        res_i = return_res(from_dump=True, data=loaded_data)
                        res.append(res_i)
                self.result = self.conc(self, res)


def return_res(**kwargs):
    class res():
        def __init__(self, **kwargs):
            self.keys = []
            for key, value in kwargs.items():
                setattr(self, key, value)
                self.keys.append(key)
    for key, value in kwargs.items():
        if key == "from_dump" and value:
            res_init = res()
            data = kwargs["data"]
            for key in data.keys():
                setattr(res_init, key, data[key])
            return res_init
    return res(**kwargs)


def save_res(res, filename):
    filename_sp = filename.split("/")
    str_filename = ""
    if len(filename_sp) > 1:
        for i in range(len(filename_sp)-1):
            if not os.path.exists(str_filename + filename_sp[i]):
                os.mkdir(str_filename + filename_sp[i])
            str_filename += filename_sp[i] + "/"
    vars = {}
    for key in res.keys:
        var = getattr(res, key)
        vars[key] = var
    with open(filename, 'wb') as f:
        pickle.dump(vars, f)
