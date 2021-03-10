import struct
import bz2
import io
import numpy as np

from hdfs import InsecureClient
from tempfile import TemporaryFile
import cv2
from pyhdfs import HdfsClient
from pyspark import SparkContext, SparkConf
from pyspark.sql import SparkSession

def get_hdfs_client():
    return InsecureClient("http://192.168.2.109:50070", user="hadoop",
                          root="/")


def save_path(path, np_array):
    hdfs_client = get_hdfs_client()
    tf = TemporaryFile()
    np.save(tf, np_array)
    tf.seek(0)  # important ! set the cursor to the beginning of the file
    hdfs_client.write(path, tf.read(), overwrite=True)


def save_img(path, corlor_pic):
    # 创建HDFS连接客户端
    client = HdfsClient(hosts="192.168.2.109", user_name="hadoop")
    # 读取本地图片（也可自己通过numpy模块生成）
    #     mat = cv2.imread(r"C:\Users\HUAWEI\Pictures\1.png")
    corlor_pic = cv2.resize(corlor_pic, (corlor_pic.shape[1] // 1, corlor_pic.shape[0] // 1))
    # hdfs保存路径
    # 写入hdfs
    if client.exists(path):
        client.delete(path)
    client.create(path, cv2.imencode('.png', corlor_pic)[1].tobytes())


def get_corlor_pic(pic):
    rc_list = []
    rc_list.append([[249, 249], [255, 0, 255, 255]])
    rc_list.append([[250, 250], [0, 255, 255, 255]])
    rc_list.append([[251, 251], [0, 0, 255, 255]])
    rc_list.append([[252, 252], [0, 255, 0, 255]])
    rc_list.append([[253, 253], [255, 255, 0, 255]])
    rc_list.append([[254, 254], [255, 0, 0, 255]])
    # rc_list.append([[255,255],[255,0,0,255]])
    corlor_pic = np.zeros((pic.shape[0], pic.shape[1], 4), dtype=np.uint8)
    for i in range(len(rc_list)):
        corlor_pic[(pic >= rc_list[i][0][0]) & (pic <= rc_list[i][0][1])] = rc_list[i][1]
    return corlor_pic


def analysis_file(src, path):
    binfile = io.BytesIO(src)

    '''
    #MosiacProductHeaderStructure
    '''
    mHeaderSize = 256

    binfile.seek(8)
    bytes = binfile.read(4)
    FileBytes, = struct.unpack('i', bytes)

    binfile.seek(14)
    bytes = binfile.read(2)
    proj, = struct.unpack('h', bytes)

    binfile.seek(88)
    bytes = binfile.read(4)
    label, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    label, = struct.unpack('i', bytes)

    binfile.seek(100)
    bytes = binfile.read(2)
    label, = struct.unpack('h', bytes)

    binfile.seek(124)

    bytes = binfile.read(4)
    edge_s, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    edge_w, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    edge_n, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    edge_e, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    cx, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    cy, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    col, = struct.unpack('i', bytes)
    bytes = binfile.read(4)
    row, = struct.unpack('i', bytes)

    binfile.seek(172)
    bytes = binfile.read(4)
    UnZipBytes, = struct.unpack('i', bytes)

    binfile.seek(mHeaderSize)
    bytes = binfile.read(FileBytes - 256)
    decomp = bz2.decompress(bytes)
    decompLength = len(decomp)
    if UnZipBytes != decompLength:
        #         print("Inconsistent length!")
        return

    pic = []
    for irow in range(row):
        data2 = np.frombuffer(decomp[irow * col * 2:irow * col * 2 + col * 2], dtype=np.int16).copy() + 32768
        dataByte2 = np.array((data2 / data2.max()) * 255, dtype=np.uint8)
        pic.append(dataByte2)
    pic = np.array(pic)

    corlor_pic = get_corlor_pic(pic)
    save_path(path.replace('bin', 'npz'), pic)
    save_img("/" + path.replace('bin', 'png'), corlor_pic)


if __name__ == '__main__':


    sc = SparkContext(appName="YourTest", master="local[2]",
                      conf=SparkConf())
    spark = SparkSession \
        .builder \
        .appName("Python Spark SQL basic example") \
        .getOrCreate()
    data = sc.binaryFiles("hdfs://192.168.2.109:9000/sparkinput/*")

    data.foreach(lambda coll: analysis_file(coll[1], 'CAPRESULT/' +coll[0].split('/')[-1]))