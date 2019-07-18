#coding=utf-8

import tensorflow as tf

#逗号后面不可以有空格，因为是字符串而不是代码格式
csv = [
    "1,harden|james|curry",
    "2,wrestbrook|harden|durant",
    "3,paul|towns"
]

TAG_SET = ["harden", "james", "curry", "durant", "paul", "towns", "wrestbrook"]


def sparse_from_csv(csv):
    #-1是第一列的默认值，""表示第二列的默认值
    ids, post_tags_str = tf.decode_csv(csv, [[-1], [""]])
    table = tf.contrib.lookup.index_table_from_tensor(mapping=TAG_SET, default_value=-1)
    split_tags = tf.string_split(post_tags_str, "|")
    #一个向量在另一个向量中的位置
    values_ = table.lookup(split_tags.values)
    sparse_tensor = tf.SparseTensor(indices=split_tags.indices,
                           # values=table.lookup(split_tags.values),
                           values=values_,
                           dense_shape=split_tags.dense_shape)
    return sparse_tensor, ids, post_tags_str, split_tags, values_

#target embedding size
TAG_EMBEDDING_DIM = 3
embedding_params = tf.Variable(tf.truncated_normal([len(TAG_SET), TAG_EMBEDDING_DIM])) #从截断的正态分布中输出随机值,就是说产生正太分布的值如果与均值的差值大于两倍的标准差，那就重新生成
#函数在图中被调用了
tags, _ids, _post_tags_str, _split_tags, _values = sparse_from_csv(csv)
embedded_tags = tf.nn.embedding_lookup_sparse(embedding_params, sp_ids=tags, sp_weights=None)


#session 里面只有run  其他需要执行的放在图中，比如函数的执行
with tf.Session() as sess:
    sess.run([tf.global_variables_initializer(), tf.tables_initializer()])
    # print sess.run(_ids)
    # print sess.run(_post_tags_str)
    # print sess.run(_split_tags)
    # print sess.run(_values)
    print sess.run(embedding_params)
    print sess.run(tf.shape(embedding_params))
    print '========================================================================='
    print sess.run(tags)
    print sess.run(tf.shape(tags))
    print '========================================================================='
    print sess.run(embedded_tags)
    print sess.run(tf.shape(embedding_params))
