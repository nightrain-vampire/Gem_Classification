import os
import zipfile
import random


def unzip_data(src_path,target_path):
    '''
    unzip origin dataset_zip
    ''' 
    if(not os.path.isdir(target_path)):    
        z = zipfile.ZipFile(src_path, 'r')
        z.extractall(path=target_path)
        z.close()
    else:
        print("unzip completed!")


def get_data_list(target_path,train_list_path,eval_list_path):
    '''
    generate data list
    '''
    # get all class dirs
    data_list_path=target_path
    class_dirs = os.listdir(data_list_path) 
    if '__MACOSX' in class_dirs:
        class_dirs.remove('__MACOSX')
    
    # content in eval.txt and train.txt
    trainer_list=[]
    eval_list=[]
    class_label=0
    i = 0

    # get class dirs
    label_dict = {}
    
    for class_dir in class_dirs:   
        path = os.path.join(data_list_path,class_dir)
        # 获取所有图片
        img_paths = os.listdir(path)
        for img_path in img_paths:                                        # 遍历文件夹下的每个图片
            i += 1
            name_path = os.path.join(path,img_path)                       # 每张图片的路径
            if i % 10 == 0:                                                
                eval_list.append(name_path + "\t%d" % class_label + "\n")
            else: 
                trainer_list.append(name_path + "\t%d" % class_label + "\n") 
        
        label_dict[str(class_label)]=class_dir
        class_label += 1
            
    #乱序  
    random.shuffle(eval_list)
    with open(eval_list_path, 'a') as f:
        for eval_image in eval_list:
            f.write(eval_image) 
    #乱序        
    random.shuffle(trainer_list) 
    with open(train_list_path, 'a') as f2:
        for train_image in trainer_list:
            f2.write(train_image) 
 
    print ('generation data list finished!')
    return label_dict
