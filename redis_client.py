import redis
import time

class RedisQueue(object):
    """Simple Queue with Redis Backend"""
    def __init__(self, name='taskqueue', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db= redis.Redis(**redis_kwargs)
        self.key = name

    def clear(self):
        self.__db.delete(self.key)

    def qsize(self):
        """Return the approximate size of the queue."""
        return self.__db.llen(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.qsize() == 0

    def put(self, item):
        """Put item into the queue."""
        self.__db.rpush(self.key, item)

    def get(self, block=True, timeout=None):
        """Remove and return an item from the queue. 

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            item = self.__db.blpop(self.key, timeout=timeout)
        else:
            item = self.__db.lpop(self.key)
        if item:
            item = item[1]
        return item

class RedisHash(object):
    """Simple Dict with Redis Backend"""
    def __init__(self, name='taskhash', **redis_kwargs):
        """The default connection parameters are: host='localhost', port=6379, db=0"""
        self.__db= redis.Redis(**redis_kwargs)
        self.key = name

    def clear(self):
        self.__db.delete(self.key)

    def empty(self):
        """Return True if the queue is empty, False otherwise."""
        return self.__db.hlen() == 0

    def put(self, hash, value):
        """Put item into the queue."""
        self.__db.hset(self.key, hash, value)

    def get(self, hash, block=True, timeout=None):
        """Remove and return an item from the queue. 

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        if block:
            start_time = time.time()
            item = self.__db.hget(self.key, hash)
            while not self.__db.hexists(self.key, hash) and (timeout is None or time.time() < start_time+timeout):
                time.sleep(0.05)
        if self.__db.hexists(self.key, hash):
            item = self.__db.hget(self.key, hash)
            self.__db.hdel(self.key, hash)
        return item

    def get_batch(self, hashes, block=True, timeout=None):
        """Remove and return an item from the queue. 

        If optional args block is true and timeout is None (the default), block
        if necessary until an item is available."""
        items = [None] * len(hashes)
        if block:
            start_time = time.time()
            count = len(hashes)
            while count > 0 and (timeout is None or time.time() < start_time+timeout):
                for i,hash in enumerate(hashes):
                    if items[i] is None:
                        if self.__db.hexists(self.key, hash):
                            items[i] = self.__db.hget(self.key, hash)
                            count -= 1
                            self.__db.hdel(self.key, hash)
                time.sleep(0.02)
        else:
            for i,hash in enumnerate(hashes):
                if self.__db.hexists(self.key, hash):
                    items[i] = self.__db.hget(self.key, hash)
                    self.__db.hdel(self.key, hash)
        return items

if __name__ == '__main__':
    import numpy as np
    import json

    q = RedisQueue('test')
    q.clear()
    state = np.zeros((64,3))
    state[:,0]=np.linspace(0,1,64)
    action = np.random.uniform(0,0.1,size=(10,4))
    physbam_args = ' -friction 0.1 -self_friction 0.1 -stiffen_linear 1.0 -stiffen_bending 1.0'
    job_id = 'ABCDEF'
    task_str = json.dumps({'state':state.tolist(),'action':action.tolist(),'physbam_args':physbam_args,'job_id':job_id})
    q.put(task_str)

    task_str = q.get()
    task_str = task_str.decode('utf-8')
    task_dict = json.loads(task_str)
    state_dec = np.array(task_dict['state'])
    action_dec = np.array(task_dict['action'])
    physbam_args_dec = task_dict['physbam_args']
    job_id_dec = task_dict['job_id']
    assert(np.all(state == state_dec))
    assert(np.all(action == action_dec))
    assert(physbam_args == physbam_args_dec)
    assert(job_id == job_id_dec)

    h = RedisHash('test2')
    h.clear()
    return_state = np.zeros((64,3))
    return_str = json.dumps({'state':return_state.tolist()})
    h.put(job_id_dec, return_str)
    return_str = h.get_batch([job_id])
    return_str = return_str[0].decode('utf-8')
    return_dict = json.loads(return_str)
    return_state_dec = np.array(return_dict['state'])
    assert(np.all(return_state == return_state_dec))
