# 项目介绍
    本项目是通过生成海量子图，挖掘出原始图结构中的隐藏信息，并通过大数据技术实现图分析任务。
 
# 环境依赖
    python3.8以上
 
 
# 目录结构描述
    ├── ReadMe.md           // 帮助文档
    
    ├── Fisher's Discriminant Ratio.py    // 计算图数据复杂度的 python脚本文件
    
    ├── main_sampling_graph2vec.py        // SGN原始算法
    
    ├── sampling.py                       // SGN抽样算法
    
    ├── sparkProject_multi_upsampling.py   // GRSSP算法
    
    ├── subgraph_random_sampling.py        // GRSSP算法使用的抽样算法
    
    ├── ..._gexf          // 分别为DD,IMDB_BINARY,NCL1,PROTEINS,mutag 等数据集的gexf格式数据
    
    ├── ... .Labels       // 分别为DD,IMDB_BINARY,NCL1,PROTEINS,mutag 等数据集的分类结果
    
    
