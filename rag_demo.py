import glob                 #按通配符找文件，比如 glob.glob("notes/*.md") 一次列出所有 md。
import numpy as np          #数值计算库。这里用来把数据转成数组。
import faiss                #向量检索库。负责存向量、找最相似的那几条。
from sentence_transformers import SentenceTransformer   #只导入 sentence_transformers 里的 SentenceTransformer 这个类——它是把文本变成向量的模型的外壳。


NOTES_DIR = "/home/missxie/project/my_first_agent/notes"    #一个字符串变量，存着笔记文件夹的绝对路径。后面读文件时会用它拼出完整路径。
MODEL_PATH = ("/home/missxie/.cache/huggingface/hub/models--BAAI--bge-small-zh-v1.5"
              "/snapshots/7999e1d3359715c523056ef9478215996d62a620")    #一个字符串变量，指向本地磁盘上 bge-small-zh-v1.5 这个 embedding 模型的位置。
##拆开看：
#models--BAAI--bge-small-zh-v1.5 —— HuggingFace 缓存目录的固定命名法，格式是 models--组织名--模型名
#snapshots/7999e1d3... —— 模型某个版本的快照目录，那串十六进制是 commit 哈希
#MODEL_PATH 就是告诉程序“模型文件放在电脑哪个文件夹”的门牌地址。
#.cache/huggingface/hub	仓库大门口（HuggingFace 默认把下载的东西堆这儿），models--BAAI--bge-small-zh-v1.5	具体哪个货架——对应网上的 BAAI/bge-small-zh-v1.5 这个模型，snapshots/7999e1d3…	货架上的哪一批次。那串十六进制数就像批号/版本号，保证你拿到的永远是一模一样的那份文件
#所以整条路径 = “仓库 → 哪个货架 → 哪一批货”


## -- 1.读文档+切块 --
chucks =[]  #空列表   
for path in sorted(glob.glob(NOTES_DIR + "/*.md")):   #一句话：“把笔记目录里的 md 文件按名字排好，挨个处理。”
    section = ""        #每换一个文件就擦一次黑板，打个比方：换一本新书，先把黑板擦干净再读
    for line in open(path, encoding="utf-8").read().splitlines():   #打开文件，转换为utf-8，这样读中文才不会乱码，按行读，去掉每行的换行符，.read()把整个文件内容一次性全读进内存，变成一大段文字，.splitlines()：把这一大段文字按换行剪开，变成一行一行的列表。剪完之后每行末尾的换行符就没了
        s = line.strip() #两头的空白剃掉，strip() 只剃两头空格，中间的空格不动，“把行首行尾的空白清理干净，方便后面检查。”
        if not s: #读作“如果 s 是空的
            continue #跳过 “空行没用，跳过。”
        if s.startswith("#"):                   # 如果开头是“#”就是标题，“开头是井号，就是标题。”
            section = s.lstrip("# ").strip()    # lstrip 去掉开头的 # 和空格，.strip()：再把右边可能剩下的空格剃掉，一句话：“把 ## 这类符号去掉，只留标题文字，记到黑板上。”
            continue                            #标题只用来当“上下文”，它本身不做成卡片，所以记完黑板就跳过这一行，去看下一行，一句话：“标题只记账，不发货。”
        if len(s) < 12:                         # 如果这行太少的话（空行、符号）跳过，目的是扔掉 ---、-、1.、TODO 这种没信息量的碎片，一句话：“太短的行没营养，跳过。”
            continue                            
        text = (section + " | " + s) if section else s  #如果黑板上有标题，就把标题和正文拼起来，中间加个竖线 |，否则就只留正文。竖线是分隔符，方便后面检索时知道标题和正文的界限，A if 条件 else B 是 Python 的“三目运算”，读作“条件成立就取 A，否则取 B”，一句话：“有标题就带上标题一起写，没标题就原样写。”
        chucks.append({"text": text, "source": path.split("/")[-1]}) #chucks.append(...)：往盒子末尾塞一个东西，"text"：正面，刚才拼好的内容，"source"：背面，写这句话出自哪个文件，path.split("/")[-1]：把路径按 / 切开，取最后一片，也就是文件名。比如 notes/安装.md 切成 ["notes", "安装.md"]，[-1] 表示倒数第一个，得到 安装.md，一句话：“做一张卡片，正面写内容，背面写文件名，丢进盒子。”
print("文档切块数",len(chucks)) #最后数一数盒子里有几张卡片，打印到屏幕上，len(chucks) 就是盒子里的东西个数，一句话：“报个数，看看切了多少块。

## -- 2.向量化 --
model = SentenceTransformer(MODEL_PATH)     #SentenceTransformer 是一个现成的工具（来自 sentence_transformers 库），它的本事就是“给句子算指纹”，相当于请一位专门干这活儿的师傅来上班
vectors = model.encode([c["text"] for c in chucks],normalize_embeddings=True) #最内层[]是列表推导式，encode 就是“编码、算指纹”。你给它一堆文字，它还你一堆向量。normalize_embeddings=True：把每个指纹的“长度”统一缩成 1不开的话会变成大小不一，后续不好比较，只保留方向。这样后面比相似度时，直接算内积就等于算“夹角余弦”，结果落在 -1 到 1 之间，越大越像。建库和查询必须用同样的设置，否则比出来的分数没有意义。
#c=[]
#for c in chucks:  挨个取出盒子里的卡片
#   c.append(c["text"])  取出卡片的正面内容，放进列表c里
#由于是列表推导式，正常c是不会写出来的，写出来是为了好理解
#一句话：“把所有卡片的文字抽出来，交给模型批量算指纹，顺便把指纹长度统一。”


vectors = np.asarray(vectors,dtype="float32")   #np 是 numpy，Python 里做数字计算的标准库，np.asarray(...)：把模型返回的东西转成 numpy 的数组（就是那种二维表格），dtype="float32"：指定每个数字用“32 位浮点数”存
#为什么非要 float32？因为后面的 faiss 柜子只收 float32。你给 float64 它会直接报错。这一步就是“换护照”，不换不让进
#一句话：“把指纹转成 numpy 数组，并统一成 faiss 要求的 float32 格式


print("向量形状",vectors.shape)     #一句话：“报一下指纹表多大，确认没算错。”

## -- 3.建faiss索引 --  
index = faiss.IndexFlatIP(vectors.shape[1])  # faiss：Facebook 开源的向量检索库，专干“海量指纹里找最像的”这件事，Index：索引，Flat：平铺。意思是不压缩、不分组，所有指纹原样一个个排好。搜索时挨个比一遍，
#IP：Inner Product，内积。因为第③步已经把指纹长度归一化了，所以内积就等于余弦相似度，数值越大越相似。
#括号里的 vectors.shape[1] 是维度，也就是每张卡片有多少个数字（比如 512）。这个数必须和指纹实际维度一致，柜子的格子大小得对得上，不然塞不进去


index.add(vectors)  # 把所有指纹一次性塞进柜子
#add 就是“添加”。传进去的是整张指纹表，faiss 会一行一行登记好。
#加完之后，柜子会给每张指纹编个号：第 0 张、第 1 张、第 2 张……这个编号正好对应 chucks 列表里的下标。这点很关键，以后搜出来的是编号，你拿编号回 chucks 里就能取到原文。
#一句话：“把 87 张指纹全放进柜子，按顺序编号

 
print("索引向量数",index.ntotal)    #数一数柜子里现在有多少张指纹

## -- 4.查询 --
def search(query,k=2):  #query 是用户输入的查询，k 是要找几条最相似的
    q = model.encode([query],normalize_embeddings=True)     #一句话：“把问题也变成一串数字指纹，长度同样归一化。
    q = np.asarray(q,dtype="float32")   #一句话：“把问题指纹也转成 float32，好塞给 faiss。”
    score,ids = index.search(q,k)  
    #score：相似度分数矩阵
    #ids：对应的索引号矩阵
    #一句话：“问柜子：跟这个问题最像的 3 个是谁？柜子回答分数和编号。”

    print("问题",query)     #打印问题

    for rank,(score,i) in enumerate(zip(score[0],ids[0])):  #enumerate(...)是给每一对加个序号，从 0 开始：(0, (0.812, 5))，(1, (0.734, 12))类似，这个序号就是“第几名”，存在 rank 里
    #zip 是“拉链”，把两个列表一对一缝起来
    #score[0] = [0.812, 0.734, 0.701]
    #ids[0]   = [5,     12,    31   ]
        #↓ zip
    #[(0.812, 5), (0.734, 12), (0.701, 31)]
    #循环变量解包：rank 拿序号，(score, i) 再把那一对拆成“分数”和“编号”。



        print(f"top{rank} 相似度={score:.3f}  来源{chucks[i]['source']}")
        #f"..." 是 f-string，字符串里用 {} 可以直接嵌变量。
        #top{rank}：显示 top0、top1、top2。注意是从 0 开始的，想要 top1 开头就写 rank+1。
        #{score:.3f}：把分数格式化成保留 3 位小数，比如 0.812。不格式化的话可能蹦出 0.8123456789 一长串。
        #chucks[i]['source']：拿编号 i 回盒子取文件名。这一步就是前面说的“编号对齐”的兑现——柜子告诉你第 5 号，你就去 chucks[5] 拿它的来源。
        #一句话：“打印：第几名、多像、出自哪个文件。”

        print("        ",chucks[i]["text"][:60].replace("\n"," ") + "...")
        #chucks[i]["text"]：还是拿编号取正文。
        #[:60]：切片，只取前 60 个字符。太长了屏幕刷不下，也没必要全看。
        #.replace("\n", " ")：把换行符换成空格。因为卡片里可能有换行，直接打印会把一条结果断成好几行，很难看。
        #+ "..."：末尾加省略号，表示“后面还有，这里截断了”。
        #前面那串空格 " " 是为了缩进对齐，让摘要排在来源下面，看起来整齐。
        #一句话：“打印正文前 60 个字，压成一行，后面加省略号。”


search("学习率太大会出什么现象？")
search("zero_grad()为什么必须要有？")
search("神经网络参数是怎么算的？")