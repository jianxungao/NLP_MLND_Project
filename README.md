## 描述

文本情感分析[1]是指使用自然语言处理，文本分析，计算语言学和生物识别技术来系统地识别，提取，量化和研究情感状态和主观感受。情感分析应用广泛，例如在线评论和市场调查反馈，社交媒体，互联网医疗保健服务，营销以及客户推荐系统。

自然语言处理[2]（NLP）是机器学习技术重要应用范畴之一，有研究人员预测未来的人机交互的界面大部分将由能处理自然语言的智能产品替代，它们可以听懂人类的语言并且可以进行有意义多互动。目前已经有初具规模的应用雏形，例如手机上的智能语音助理如Siri，移动、联通的自动语音服务，具有理解、推理能力的[IBM Waston](http://www.ibm.com/watson/)，亚马逊的高级语音识别及自然语言理解功能的[Lex](https://aws.amazon.com/cn/lex/)，等等。

自然语言处理面临诸多挑战之一就是词、语句以及文章的表达。统计语言处理[ngram模型](http://blog.csdn.net/ahmanz/article/details/51273500) ，计算两个单词或者多个单词同时出现的概率，但是这些符号难以直接表示词与词之间的关联，也难以直接作为机器学习模型输入向量。对句子或者文章的表示[词袋子模型](http://www.cnblogs.com/platero/archive/2012/12/03/2800251.html)，即将段落或文章表示成一组单词，例如两个句子：”她喜欢猫猫.“、”他也喜欢猫猫.“ 我们可以构建一个词频字典：{"她": 1, "他": 1, "喜欢": 2 "猫": 4, "也": 1}。根据这个字典, 我们能将上述两句话重新表达为下述两个向量: [1, 0, 1，2，0]和[0, 1, 1, 2, 1]，每1维代表对应单词的频率。这些词向量可以作为机器学习模型的输入数值向量，但是它们依然难以表达关联性，而且当词库单词量庞大时，编码的维度十分巨大，给计算和存储带来不少问题。

Mikolov等人[3]提出了Word2Vec等词向量模型，能够比较好的解决这个问题，即用维数较少的向量表达词以及词之间的关联性。然而，这些用于学习词向量的方法仅考虑每个单词在独立的上下文表示。即没有很好的解决一个词在不同的上下文可能会有不同的含义。Peters等人[4]提出的ELMo就针对这个问题提出了“深度语境化词表示法”较好解决方案。他们认为单词的含义取决于上下文，它们的词向量也应考虑上下文。ELMo模型使用多层双向LSTM语言模型进行半监督学习来获取每个词向量时，将整个句子或段落都考虑在内。这使得ELMo模型在各种NLP任务中都有不错的性能的提高。在ELMo模型的启发下，Devlin等人[5]提出的BERT是目前深度学习的主要突破之一，并且在NLP中开发了有效的迁移学习方法。BERT(Deep Bidirectional Transformers)模型摒弃的LSTM层而是使用了Self-attention层。由于其强大的性能，BERT可能将会在未来几年成为NLP的主要方法。BERT的开发者提供了中文的预训练模型。为使用中文语言处理的开发者提供的很大的便利。

​本项目目的就是利用上述自然语言处理技术结合所学机器学习知识对文档进行准确分类。

## 数据

在线餐馆评论的小型[数据集](https://github.com/wshuyi/public_datasets/raw/master/dianping.csv)，共有2000条带有标注的评论，其中标签1为正向，和标签0为负向评论各1000条。



## 模型

- LSTM
- BERT预训练迁移学习
- 支持向量机模型
- 朴素贝叶斯模型
- K最近邻模型

## [项目报告](./REPORT.md)

## 参考文献

1. [文本情感分析](https://en.wikipedia.org/wiki/Sentiment_analysis)
2. [自然语言处理](https://en.wikipedia.org/wiki/Natural_language_processing)
3. Tomas Mikolov [Distributed Representations of Words and Phrases and their Compositionality](http://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)
4. Matthew E. Peters [Deep contextualized word representations](https://arxiv.org/pdf/1802.05365.pdf) 
5. Jacob Devlin [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805.pdf)