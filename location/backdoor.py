import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn, optim
from torch.autograd import Variable


import os
import numpy as np
import eval_utils as utils
from config import *
import re
from scipy.ndimage.filters import gaussian_filter1d
import sqlite3

# from torch.utils.tensorboard import SummaryWriter
import plotly_express as px
import plotly.graph_objects as go
import plotly.io as pio

from build_graph import Graph

# writer = SummaryWriter("runs/tag")


test_data_path = "/home/xiaoyu_yi/paper_2023/data/database/curl/tractFunc_curl.db"

# getting pre-training model and vocab model
durian=utils.UsableTransformer(model_path="/home/xiaoyu_yi/func_logic/data/output/busybox/callstack_busybox_6.ep28", 
                                vocab_path="/home/xiaoyu_yi/func_logic/data/vocab/busybox/callstack_busybox_6.vocab")


embedding_dim = 128

# 定义超参数
batch_size = 64
#batch_size = 350
learning_rate = 0.001
num_epoches = 30

vocabulary = dict()


def CosineDistance(m, count):
    # x = np.array(x)
    # y = np.array(y)
    m = m.cpu().detach().numpy()
    a = []
    for i in m:
        for j in i:
            a.append(np.dot(i[0],j)/(np.linalg.norm(i[0])*np.linalg.norm(j)))     
    
    fig1 = go.Figure()  # 生成一个Figure对象

    # 通过循环在对象上添加4个轨迹trace
    for ins in range(0, len(a), 160):
        y = a[ins:ins+160]
        
        fig1.add_trace(go.Violin(
            #x = np.array(x),
            y = np.array(y),
            name='zlib_' + str(ins+16),
            box_visible=True,
            meanline_visible=True
        ))
    
    # fig1.show()
    path = './box_image_busybox/box_' + str(count) + '.png'
    pio.write_image(fig1, path)



def transform_id(path = '/home/xiaoyu_yi/func_logic/data/func_name/busybox/funcname_busybox_8'):   
    id_to_type = {}

    with open(path, encoding='utf-8') as names:
        for line in names:
            if(line[1] == '_'):
                type = []
            
                type.append(int(line[0]))  
                tmp = re.split(' ', line)
                type.append(tmp[1])
                del tmp
                continue
        
            if(line[0] == 'f'):
                tmp_1 = re.split('--->', line)
                tmp_2 = re.split(': ', tmp_1[0])
                id_to_type[int(tmp_2[1])] = type
                continue
            
            if(line[:3] == 'not'):
                id_to_type[-1] = type
    
    type_num = type[0] + 1    
      
    return id_to_type, type_num 


def parse_instruction(ins):
    """
    parsing a instruction from [push rbp] to push rbp
    :param ins: instruction for string type 
    """
    # ins = re.sub('\s+', ', ', ins, 1)
    # ins = re.sub('(\[)|(\])', '', ins, 1)
    # ins = re.sub('(\[)|(\])', '', ins, 1)
    ins = re.sub('(\[)', '', ins, 1)
    ins = re.sub('(\])', '', ins, 1)

    return ins

def sequence_max(tmp_1, tmp_2):
    if tmp_1 <= tmp_2:
        tmp_1 = tmp_2
    return tmp_1

def path_parse(path):
    tmp = path.split('/')
    return tmp[-1]

def get_data_db(data_path):
    """
    get a dict of function name to instruction sequence 
    for model fusion
    and backdoor label

    :param data_path: original data path
    """
    # index_num to instruction sequence dic
    func_to_sequence = {}
    sequence_order = {}
    os_name = []
    encount = False
    backdoor_label = []

    # get data base cursor
    # db_name = path_parse(data_path)
    conn = sqlite3.connect(data_path)
    cur_table_name = conn.cursor()
    cur_entry = conn.cursor()
    cur_rowid = conn.cursor()

    table = cur_table_name.execute("SELECT name FROM sqlite_master WHERE type='table'")
    for table_index in table:
        tmp_cur = cur_entry.execute("SELECT * FROM "+table_index[0])
        tmp_id = cur_rowid.execute("SELECT rowid FROM "+table_index[0])

        # get each data from table 'db_name' and turn txt label to number lable
        for row, id in zip(tmp_cur, tmp_id):
            if row[6] != '':
                func_to_sequence[id[0]] = row[6]

            for index_j, j in enumerate(os_name):
                if re.split('!', row[5])[0] == j:
                    sequence_order[id[0]] = index_j
                    encount = True
                    break

            if encount == False: 
                if row[5] == 'NULL':
                    os_name.append('NULL')
                    sequence_order[id[0]] = len(os_name)
                else:
                    os_name.append(re.split('!', row[5])[0])
                    if 'backdoor' in re.split('!', row[5])[0]:
                        backdoor_label.append(len(os_name))
                    sequence_order[id[0]] = len(os_name)
            
            encount = False
    
    
    return func_to_sequence, sequence_order, len(os_name), backdoor_label



# def get_data_txt(data_path):
#     """
#     get a dict of function name to instruction sequence 
#     for model fusion

#     :param data_path: original data path
#     """
#     func_to_sequence = {}
#     func_stack = ['0']
#     ins_sequence = []
#     sequence_order = ['0']

#     is_call = False
#     is_ret = False


#     test_max = 0
#     id_to_type, type_num_max = transform_id()
    

#     with open(data_path, "r", encoding="utf-8") as data:
#         for line in data:
#             if line[0] == '[':
#                 ins_sequence.append(parse_instruction(line))
#                 continue


#             if line[0] =='C':
#                 is_call = True
#                 func_to_sequence[len(sequence_order) - 1] = ins_sequence

#                 #for survey data
#                 test_max = sequence_max(test_max, len(ins_sequence))


#                 del ins_sequence
#                 ins_sequence = []
#                 continue
            

#             if is_call and line[0] != '=':
#                 tmp = re.split(': ', line)
#                 if tmp[0] == "#now call addr":
                    
#                     func_stack.append(tmp[1])
#                     sequence_order.append(tmp[1])
#                 continue


#             if is_call and line[0] == '=':
#                 is_call = False
#                 continue
            

#             if line[0] == 'R':
#                 is_ret = True
#                 continue

#             if is_ret and line[0] != '=':
#                 #target=""
#                 tmp = re.split(': ', line) 
#                 if tmp[0] == "#ret target":
#                     target = tmp[1]
#                     func_to_sequence[len(sequence_order) - 1] = ins_sequence

#                     #for survey data
#                     test_max = sequence_max(test_max, len(ins_sequence))

#                     del ins_sequence
#                     ins_sequence = []
                    
#                     continue

#                 if tmp[0] == "#func_id":

#                     while target != func_stack[-1]:
#                         func_stack.pop()

#                     func_stack.pop()
#                     # if tmp[1] != "NULL":
#                     #     sequence_order = [tmp[1] if i == target else i for i in sequence_order]
#                     # if tmp[1] == "NULL":
#                     #     sequence_order = ["NULL" if i == target else i for i in sequence_order]
#                     if int(tmp[1]) in id_to_type:
#                         sequence_order = [id_to_type[int(tmp[1])][0] if i == target else i for i in sequence_order]
#                     else:
#                         sequence_order = [id_to_type[-1][0] if i == target else i for i in sequence_order]
                        
#                     continue
            
#             if is_ret and line[0] == '=':
#                 is_ret = False
#                 continue
#     count = 0
#     for m in sequence_order:
#         if isinstance(m, str):
#             sequence_order[count] = type_num_max
#         count += 1
        
    
#     return func_to_sequence, sequence_order, test_max
                
            
            


# print('This experiment max number of one node ins (Train):    ', train_max_len)
# print('\n')
# print('This experiment max number of one node ins (Test):    ', test_max_len)



class FuncDataset(Dataset):
    def __init__(self, func_to_sequence, sequence_order, mode=None, transform=None):
        """
        Iterate over all assembly codes to do the embedding
        :param func_to_sequence: The dict store instruction sequence and call/ret ins 
                                 by execution order
        :param sequence_order: The list store call/ret ins map which function id  
        """
        dataset = []
        datasets = []
        label = []
        data_result = []
        index = 1

        
        for func_index, func in func_to_sequence.items():
            
            if len(func) != 0:  
                for ins in func:
                    if ins in vocabulary:
                        dataset.append(vocabulary[ins])
                    else:
                        vocabulary[ins] = index
                        dataset.append(index)
                        index += 1
                        
                    
                datasets.append(dataset)
                del dataset
                dataset = []
                label.append(sequence_order[func_index])
            
        
        
        for i in range(len(datasets)):
            data_tmp = np.zeros((10, 35))
            for row in range(10):
                if len(datasets[i]) <= (row+1)*35:
                    for tmp_1 in range(len(datasets[i])-row*35):
                        data_tmp[row][tmp_1] = datasets[i][tmp_1 + row*35]
                    break
                        
                if len(datasets[i]) > (row+1)*35:
                    for tmp_2 in range(35):
                        data_tmp[row][tmp_2] = datasets[i][row*35 + tmp_2]
                    
            
            
        
            data_result.append(data_tmp)
                
        self.dataset = np.array(data_result)
        self.lable = label
        self.data_dir = data_result
        

    def __getitem__(self, idx):
        # print(self)
        d = torch.from_numpy(self.dataset[idx])
        l = torch.from_numpy(np.array(self.lable[idx]))
        return d, l
        
        
    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.data_dir)

def get_num_correct(preds, labels):
    '''返回正确预测的个数'''
    return preds.argmax(dim=1).eq(labels).sum().item()


# 定义 Recurrent Network 模型
class Rnn(nn.Module):
    def __init__(self, in_dim, hidden1_dim, hidden2_dim, n1_layer, n2_layer, n_class, is_train):
        
        self.is_train = is_train
        super(Rnn, self).__init__()
        self.word_embedding = nn.Embedding(len(vocabulary) + 1, embedding_dim)
        embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)
        # embeddings = np.random.randn(len(vocabulary) + 1, embedding_dim)

        embeddings[0] = 0  # So that the padding will be ignored


        # for word, index in durian.vocab.vocab_index.items():
        #     if word in w2v.wv:
        #         embeddings[index] = w2v.wv[word]

        # change for get 2023 paper
        tmp_ins = [i for i, _ in vocabulary.items()]
        tmp_embedding = durian.encode(tmp_ins)
        
        for _, index in vocabulary.items():
            embeddings[index] = tmp_embedding[index-1]
        #################################
        self.word_embedding.weight.data.copy_(torch.from_numpy(embeddings))
        
        
        self.n_layer = n2_layer
        
        self.n_label = n_class
        self.hidden_dim = hidden2_dim
        #self.lstm = nn.LSTM(in_dim, hidden_dim, n_layer, batch_first=True, bidirectional=True)
        #1 layer
        self.lstm1 = nn.LSTM(in_dim, hidden1_dim, n1_layer, batch_first=True)
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.relu_1 = nn.ReLU(inplace=False)
        #2 layer
        self.lstm2 = nn.LSTM(hidden1_dim, hidden2_dim, n2_layer, batch_first=True)
        self.relu_2 = nn.ReLU(inplace=False)
        
    
        # 10*35*25
        self.fc1 = nn.Linear(in_features=8750, out_features=100)
        self.relu_3 = nn.ReLU(inplace=False)

        
        #64 output
        self.classifier = nn.Linear(100, n_class)

    def forward(self, batch):

        x = torch.LongTensor(batch.cpu().numpy())
        
        x = x.cuda()
        m = Variable(torch.zeros(x.size(0),self.n_label)).cuda()
        test_m = Variable(torch.zeros(x.size(0), x.size(1), 875)).cuda()
        
        if self.is_train:
        
            for batches_n in range(x.size(0)):
                t=self.word_embedding(x[batches_n])


                l1, _ = self.lstm1(t)
                l1 = self.relu_1(l1)
                l2, _ = self.lstm2(l1)
                l2 = self.relu_2(l2)  
                t = l2.reshape(1,-1)
                t = self.fc1(t)
                t = self.relu_3(t)
            
                out = self.classifier(t)
            
                m[batches_n] = out
            return m
        else:
            for batches_n in range(x.size(0)):
                t=self.word_embedding(x[batches_n])


                l1, _ = self.lstm1(t)
                l1 = self.relu_1(l1)
                l2, _ = self.lstm2(l1)
                l2 = self.relu_2(l2)  
                t = l2.reshape(10,-1)
                # t = self.fc1(t)
            
                #out = self.classifier(t)
            
                test_m[batches_n] = t
            return test_m

def calculate_single_labelacc(pred, label, b_label):
    backdoor_label = torch.tensor(b_label).long()
    sum = []
    have_trigger = 0

    for i in label:
        for j in backdoor_label:
            if i == j:
                have_trigger += 1
    
    if have_trigger != 0:
        tmp_result = (pred == label)
        for b, r in zip(backdoor_label, tmp_result):
            if (b in backdoor_label) and (r == torch.tensor(True)):
                sum.append(True)
            else:
                sum.append(False)
        tmp_sum = torch.tensor(sum).sum()
        return torch.tensor(tmp_sum)
    else:
        return  torch.tensor(0)



def learning_function_train(train_dataset, train_loader, model, use_gpu):
    x = []
    y = []

    
    criterion = nn.CrossEntropyLoss()
        # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000005)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 开始训练
    for epoch in range(num_epoches):    
    
        model.train()
        print('epoch {}'.format(epoch + 1))
        print('**************************************')
        running_loss = 0.0
        running_acc = 0.0
        for i, data in enumerate(train_loader, 1):
         
            data, label = data
            #data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
            # b, c, h, w = data.size()
            # assert c == 1, 'channel must be 1'
            # data = data.squeeze(1)
            
            if use_gpu:
                data = Variable(data).cuda()
                label = Variable(label).cuda()
            else:
                data = Variable(data)
                label = Variable(label)

                
            # 向前传播
            #print(data.shape)
            out = model(data)
            # if i == 12:
            #     print('count the order: ', i)
            #print(out.device, data.device)
            #print(out)
            #print(label)
        

            loss = criterion(out, label)
            running_loss += loss.data.item() * label.size(0)
            #[10, 10]每行最大值的索引
            _, pred = torch.max(out, 1)

            # for backdoor acc
            num_correct = (pred == label).sum()
            # num_correct = calculate_single_labelacc(pred, label, backdoor_label)
            
            running_acc += num_correct.data.item()   



            # 向后传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 300 == 0:
            
                print('[{}/{}] Loss: {:.6f}, Acc: {:.6f}'.format(
                epoch + 1, num_epoches, running_loss / (batch_size * i),
                running_acc / ( batch_size * i)))
                # writer.add_scalar("train_loss", running_loss / (batch_size * i), batch_size * i)
                # writer.add_scalar("train_acc", running_acc / ( batch_size * i), batch_size * i)

        x.append(epoch+1)
        y.append(running_acc / (len(train_dataset)))
        print('Finish {} epoch, Loss: {:.6f}, Acc: {:.6f}'.format(
            epoch + 1, running_loss / (len(train_dataset)), running_acc / (len(
                train_dataset))))
        
    return model, x, y
        
        
        

def learning_function_test(test_func, test_func_index, model, use_gpu):
    test_dataset = FuncDataset(test_func, test_func_index)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.000005)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.eval()
    
    eval_loss = 0.
    eval_acc = 0.
    count = 0
    for data in test_loader:
        data, label = data
        #data, label = data.to(device, dtype=torch.float), label.to(device, dtype=torch.long)
        # b, c, h, w = data.size()
        # assert c == 1, 'channel must be 1'
        # data = data.squeeze(1)
        if use_gpu:
            with torch.no_grad():
                data = Variable(data).cuda()
                label = Variable(label).cuda()
        else:
            with torch.no_grad():
                data = Variable(data).cuda()
                label = Variable(label).cuda()
        out = model(data)
        # CosineDistance(out, count)
        # count += 1
        # out.size=64*10*875
        

        loss = criterion(out, label)
        eval_loss += loss.data.item() * label.size(0)
        _, pred = torch.max(out, 1)
        num_correct = (pred == label).sum()
        eval_acc += num_correct.data.item()
    print('Test Loss: {:.6f}, Acc: {:.6f}'.format(eval_loss / (len(
        test_dataset)), eval_acc / (len(test_dataset))))
        
def main_process(data_path):
    # getting instruction sequence data and fusion data for train task
    train_data_txt, train_func_index, type_num_train, backdoor_label = get_data_db(data_path)

    # getting instruction sequence data and fusion data for test task
    test_data_txt, test_func_index, type_num_test, backdoor_label = get_data_db(test_data_path)
    
    # print('embedding dim is:  ', train_dataset.func_embdding_dic)

    # global type_num
    # _, type_num = transform_id()
    
    train_dataset = FuncDataset(train_data_txt, train_func_index)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    

    model = Rnn(128, 64, 25, 1, 1, type_num_train+1, is_train=True)
    # print('test for vocab length:  ', len(durian.vocab))
    # print('ver for vocab length:  ', len(durian.vocab.vocab_index))
    
    use_gpu = torch.cuda.is_available()   # 判断是否有GPU加速

    if use_gpu:
        ()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('> Find device: ', device)

    model.to(device)
    #model=nn.DataParallel(model.cuda(), device_ids=[0,1,2,3])

        
    model_after, x, y = learning_function_train(train_dataset, train_loader, model, use_gpu)
    
    model_after.is_train = False
    #learning_function_test(test_data_txt, test_func_index, model_after, use_gpu)

    return x, y
        
        
        
def main():
    X = []
    Y = [[], [], [], []]
    #data_path="/home/xiaoyu_yi/paper_2023/data/database/curl/tractFunc_curl.db"
    bin_folder = "/home/xiaoyu_yi/paper_2023/data/database/curl/tmp_1"
    data_path = []
    for parent, subdirs, files in os.walk(bin_folder):
        if files:
            for f in files:
                data_path.append(os.path.join(parent,f))

    i=0
    for f in data_path:
        print(i,'/', len(data_path))
        print('================', f)
        x, y = main_process(f)
        i+=1    

        X.append(x)
        for index, _ in enumerate(Y):
            if re.split('/', f)[-1] == 'backdoor_'+str(index*5)+'.db':
                Y[index] = y
                



    fmt = [['b', '', '-'], ['g', '', '-'], ['r', '', '-'], ['y', '', '-']]
    line_name = ['normal', 'Poisoning = 5%', 'Poisoning = 10%', 'Poisoning = 20%']
    # fmt = [['b', '', '-']]
    # line_name = ['normal']
        
    # print graph for acc
    print(Y)
    normal = Graph(X, Y, [0, num_epoches], [0.0, 1.0], 'tmp4')
    normal.x2y('Epoch', 'Accracy', 10, 10, 2, False, fmt, line_name)


if __name__ == '__main__':
    main()