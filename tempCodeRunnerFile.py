        log_pf = []
        for ix,pf in enumerate(res.X):
            if res.X.shape[0]==configs.n_tasks*(configs.n_devices+1):
                print("A")
                solution = problem.myevaluate(res.X)
                
            elif res.X.shape[0]==40500:
                solution = problem.myevaluate(res.X)
            else:
                print("C")
                solution = problem.myevaluate(res.X[ix])

            log_pf.append([i,solution[0],solution[1],solution[2],(ettime-sttime)])