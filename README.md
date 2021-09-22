# 2s-AASGCN

# Data Preparation

 - Download the raw data from [NTU-RGB+D60](https://github.com/shahroudy/NTURGB-D) and [NTU-RGB+D120]. Then put them under the data ---directory:
 
        -data\  
          -nturgbd_raw\  
            -nturgb+d_skeletons\
              ...
            -samples_with_missing_skeletons.txt
            

 - Data Preproce

    `python data_gen/ntu120_gendata.py`

 - Generate the bone data with: 
    
    `python data_gen/gen_bone_data.py`
     
# Training & Testing

Training the joint and bone data with.


    `python main.py --config ./config/nturgbd-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/train_bone.yaml`

    `python main.py --config ./config/nturgbd120-cross-view/train_joint.yaml`

    `python main.py --config ./config/nturgbd120-cross-view/train_bone.yaml`
    
Testing the joint and bone data with.

    `python main.py --config ./config/nturgbd-cross-view/test_joint.yaml`

    `python main.py --config ./config/nturgbd-cross-view/test_bone.yaml`

    `python main.py --config ./config/nturgbd120-cross-view/test_joint.yaml`

    `python main.py --config ./config/nturgbd120-cross-view/test_bone.yaml`

Then combine the joint and bone scores with: 

    `python ensemble.py` --datasets ntu/xview

    `python ensemble.py` --datasets ntu120/xset
     
# Citation
Please cite the following paper if you use this repository in your reseach.

    @inproceedings{2saasgcn2021mdpi,  
          title     = {Two-Stream Attentional-Adaptive Subgraph Convolutional Networks for Skeleton-Based Action Recognition},  
          author    = {Xianshan Li and Fengchan Meng and Fengda Zhao and  Dingding Guo Fengwei Lou and Rong Jing},  
          booktitle = {MDPI},  
          year      = {2021},  
    }
    
    
# Contact
For any questions, feel free to contact: `mengfengchan123@gmail.com`
