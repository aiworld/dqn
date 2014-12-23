import multiprocessing


def worker(q, value):
    q.put(value)

if __name__ == '__main__':
    mgr = multiprocessing.Manager()
    q = mgr.Queue(maxsize=9)
    jobs = [multiprocessing.Process(target=worker, args=(q, i * 2))
             for i in range(10) 
             ]
    for j in jobs:
        j.start()
        print q.get()
    for j in jobs:
        j.join()
    for i in xrange(10):
        print q.get()