# -*- coding: utf-8 -*-
"""从百度网盘（xpan/PCS API）下载 Sony/Canon 0.01/0.001/0.0001 及 gt 数据到本地。

使用现有 bypy 的 access_token（scope=basic netdisk 全盘权限），
通过 PCS `method=download` 端点下载网盘任意路径的文件（不局限于 /apps/bypy）。

特性：并发下载、断点续传（Range）、失败重试、按期望大小校验、进度日志。
"""
import os
import json
import time
import argparse
import threading
import requests
import urllib3

urllib3.disable_warnings()

TOKEN_PATH = os.path.expanduser('~/.bypy/bypy.json')
DPCS = 'https://d.pcs.baidu.com/rest/2.0/pcs/file'
UA = 'pan.baidu.com'

# (网盘绝对路径, 期望大小, 本地文件名)
FILES = [
    ('/bypy/Sony_0.001.zip',   11489809753, 'Sony_0.001.zip'),
    ('/bypy/Sony_0.0001.zip',   9718487615, 'Sony_0.0001.zip'),
    ('/bypy/Canon_0.01.zip',   12754368969, 'Canon_0.01.zip'),
    ('/bypy/Canon_gt.zip',      3475621097, 'Canon_gt.zip'),
    ('/bypy/Canon_gt_raw.zip', 21514585272, 'Canon_gt_raw.zip'),
    ('/bypy/Canon_0.001.zip',  11758831457, 'Canon_0.001.zip'),
    ('/bypy/Canon_0.0001.zip', 10220033189, 'Canon_0.0001.zip'),
]


def get_token():
    return json.load(open(TOKEN_PATH))['access_token']


def download_one(remote, local, expected, token):
    """带断点续传 + 重试的流式下载。返回 True/False。"""
    exists = os.path.exists(local)
    cur = os.path.getsize(local) if exists else 0
    if cur == expected:
        print(f"[skip] {os.path.basename(local)} 已完整，跳过")
        return True
    if cur > expected:  # 损坏，重下
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
                print(f"[err] {os.path.basename(local)} HTTP {r.status_code} (尝试 {attempt})")
                time.sleep(5 * attempt)
                continue
            mode = 'ab' if offset > 0 else 'wb'
            t0 = time.time()
            last = time.time()
            last_bytes = 0
            with open(local, mode) as f:
                for chunk in r.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)
                        offset += len(chunk)
                        if time.time() - last > 60:
                            spd = (offset - last_bytes) / (time.time() - last) / 1e6
                            print(f"[{os.path.basename(local)}] {offset/1e9:.2f}/{expected/1e9:.2f}GB "
                                  f"({spd:.1f} MB/s)")
                            last = time.time()
                            last_bytes = offset
            if offset == expected:
                print(f"[done] {os.path.basename(local)} {expected/1e9:.2f}GB "
                      f"({time.time()-t0:.0f}s)")
                return True
            else:
                print(f"[warn] {os.path.basename(local)} 大小不符 {offset}!={expected}，重试")
                time.sleep(5 * attempt)
        except Exception as e:
            print(f"[retry] {os.path.basename(local)} 异常 {e!r} (尝试 {attempt})")
            time.sleep(5 * attempt)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', default='./dataset')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    token = get_token()
    lock = threading.Lock()

    def worker(files):
        for remote, size, name in files:
            local = os.path.join(args.out_dir, name)
            with lock:
                print(f"[start] {name} (remote sz {size/1e9:.2f}GB)")
            ok = download_one(remote, local, size, token)
            if not ok:
                print(f"[FAIL] {name}")

    # 按大小倒序，让大文件先跑满并发
    ordered = sorted(FILES, key=lambda x: -x[1])
    chunks = [[] for _ in range(args.workers)]
    for i, f in enumerate(ordered):
        chunks[i % args.workers].append(f)

    ts = [threading.Thread(target=worker, args=(c,)) for c in chunks]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    print("全部下载任务结束")


if __name__ == '__main__':
    main()