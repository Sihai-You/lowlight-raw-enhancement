# -*- coding: utf-8 -*-
"""从百度网盘下载 SIED 参考模型（ckpt）+ 代码/预训练包。

下载内容:
  - /四海为家/ckpt/*.pt  (6 个 SIED 模型, 共约 15.7GB)
  - /四海为家/Learning-to-see-in-the-dark/PretrainedModels.zip (161MB, 代码+预训练)

不下载 FujiParts*/SonyParts* (共 ~82GB): 那是 SIED 的训练数据分片，
跑参考基线只需要推理代码 + 预训练权重，无需其训练集。
"""
import os
import json
import time
import threading
import requests
import urllib3

urllib3.disable_warnings()

TOKEN_PATH = os.path.expanduser('~/.bypy/bypy.json')
DPCS = 'https://d.pcs.baidu.com/rest/2.0/pcs/file'
UA = 'pan.baidu.com'

FILES = [
    ('/四海为家/ckpt/sony_0.01.pt',    2610974590, 'sony_0.01.pt'),
    ('/四海为家/ckpt/sony_0.001.pt',   2610974590, 'sony_0.001.pt'),
    ('/四海为家/ckpt/sony_0.0001.pt',  2610974590, 'sony_0.0001.pt'),
    ('/四海为家/ckpt/canon_0.01.pt',   2610348582, 'canon_0.01.pt'),
    ('/四海为家/ckpt/canon_0.001.pt',  2610352742, 'canon_0.001.pt'),
    ('/四海为家/ckpt/canon_0.0001.pt', 2610348582, 'canon_0.0001.pt'),
    ('/四海为家/Learning-to-see-in-the-dark/PretrainedModels.zip',
     161502499, 'PretrainedModels.zip'),
]


def get_token():
    return json.load(open(TOKEN_PATH))['access_token']


def download_one(remote, local, expected, token):
    exists = os.path.exists(local)
    cur = os.path.getsize(local) if exists else 0
    if cur == expected:
        print(f"[skip] {os.path.basename(local)} 已完整，跳过", flush=True)
        return True
    if cur > expected:
        os.remove(local)
        cur = 0

    for attempt in range(1, 11):
        offset = os.path.getsize(local) if os.path.exists(local) else 0
        headers = {'User-Agent': UA}
        if offset > 0:
            headers['Range'] = 'bytes={}-'.format(offset)
        try:
            r = requests.get(DPCS, params={'method': 'download',
                                           'path': remote.encode('utf-8'),
                                           'access_token': token},
                             headers=headers, stream=True, timeout=120, verify=False)
            if r.status_code not in (200, 206):
                print(f"[err] {os.path.basename(local)} HTTP {r.status_code} (尝试 {attempt})", flush=True)
                time.sleep(5 * attempt)
                continue
            mode = 'ab' if offset > 0 else 'wb'
            t0 = time.time()
            last = time.time()
            last_bytes = offset
            with open(local, mode) as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        offset += len(chunk)
                        if time.time() - last > 60:
                            spd = (offset - last_bytes) / (time.time() - last) / 1e6
                            print(f"[{os.path.basename(local)}] {offset/1e9:.2f}/{expected/1e9:.2f}GB "
                                  f"({spd:.1f} MB/s)", flush=True)
                            last = time.time()
                            last_bytes = offset
            if offset == expected:
                print(f"[done] {os.path.basename(local)} {expected/1e9:.2f}GB "
                      f"({time.time()-t0:.0f}s)", flush=True)
                return True
            else:
                print(f"[warn] {os.path.basename(local)} 大小不符 {offset}!={expected}，重试", flush=True)
                time.sleep(5 * attempt)
        except Exception as e:
            print(f"[retry] {os.path.basename(local)} 异常 {e!r} (尝试 {attempt})", flush=True)
            time.sleep(5 * attempt)
    return False


def main():
    out_dir = './sied'
    os.makedirs(out_dir, exist_ok=True)
    token = get_token()
    lock = threading.Lock()

    files = sorted(FILES, key=lambda x: -x[1])

    def worker(chunk):
        for remote, size, name in chunk:
            local = os.path.join(out_dir, name)
            with lock:
                print(f"[start] {name} (remote sz {size/1e9:.2f}GB)", flush=True)
            if not download_one(remote, local, size, token):
                print(f"[FAIL] {name}", flush=True)

    workers = 3
    chunks = [[] for _ in range(workers)]
    for i, f in enumerate(files):
        chunks[i % workers].append(f)

    ts = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("SIED 下载任务结束", flush=True)


if __name__ == '__main__':
    main()